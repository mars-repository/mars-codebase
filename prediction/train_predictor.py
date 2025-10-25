"""
Use lora to train OPT-125M on the toolbench prediction dataset.
"""
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from peft import get_peft_model, LoraConfig, PeftType
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from tqdm import tqdm
import os

batch_size = 64
model_name_or_path = "facebook/opt-125m"

context_window = 512 if any(k in model_name_or_path for k in ("bert")) else 2048
use_lora = True
peft_type = PeftType.LORA
device = "cuda"
num_epochs = 40

peft_config = LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=32, lora_alpha=64, lora_dropout=0.1)
lr = 3e-4

if any(k in model_name_or_path for k in ("gpt", "opt", "bloom")):
    padding_side = "left"
else:
    padding_side = "right"

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side=padding_side)
if getattr(tokenizer, "pad_token_id") is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Load the custom dataset
dataset_path = "toolbench_prediction_dataset"
dataset = load_from_disk(dataset_path)

# Define the number of bins and the bin width
num_bins = 50 
bin_width = 10

def bin_labels(label):
    """Bin the continuous label into fixed-width bins."""
    return min(num_bins - 1, int(label / bin_width))

dataset = dataset.map(lambda x: {"label": bin_labels(x["label"])})

def tokenize_function(examples):
    return tokenizer(examples["prompt"], truncation=True, max_length=context_window)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["prompt", "completion", 'api_token', 'api_time', 'api_name'])

tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

def collate_fn(examples):
    return tokenizer.pad(examples, padding="longest", return_tensors="pt")

train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, collate_fn=collate_fn, batch_size=batch_size)
eval_dataloader = DataLoader(tokenized_datasets["test"], shuffle=False, collate_fn=collate_fn, batch_size=batch_size)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name_or_path, num_labels=num_bins, return_dict=True
)
for param in model.base_model.parameters():
    param.requires_grad = False

# Apply LoRA
if use_lora:
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

model.config.pad_token_id = tokenizer.pad_token_id
model.to(device)

optimizer = AdamW(params=model.parameters(), lr=lr)
loss_fn = CrossEntropyLoss()
loss_fn = torch.compile(loss_fn)

writer = SummaryWriter()

lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0.06 * (len(train_dataloader) * num_epochs),
    num_training_steps=(len(train_dataloader) * num_epochs),
)

global_step = 0

output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

all_predictions = []
all_labels = []

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        batch['labels'] = batch['labels'].long()

        outputs = model(**batch)
        predictions = outputs.logits

        loss = loss_fn(predictions, batch['labels'])
        total_train_loss += loss.item()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        global_step += 1
        writer.add_scalar("Loss/Train", loss, global_step)

    avg_train_loss = total_train_loss / len(train_dataloader)
    writer.add_scalar("Loss/Train/Avg", avg_train_loss, epoch)

    model.eval()
    total_eval_loss = 0
    total_accuracy = 0
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        batch['labels'] = batch['labels'].long()

        with torch.no_grad():
            outputs = model(**batch)
            predictions = outputs.logits
            eval_loss = loss_fn(predictions, batch['labels'])
            total_eval_loss += eval_loss.item()

            preds = torch.argmax(predictions, dim=1)
            batch_accuracy = (preds == batch['labels']).float().mean()
            total_accuracy += batch_accuracy.item()

    avg_eval_loss = total_eval_loss / len(eval_dataloader)
    avg_accuracy = total_accuracy / len(eval_dataloader)
    
    writer.add_scalar("Loss/Eval", avg_eval_loss, epoch)
    writer.add_scalar("Accuracy/Eval", avg_accuracy, epoch)

    print(f"epoch {epoch}: Average Evaluation Loss: {avg_eval_loss}, Accuracy: {avg_accuracy:.4f}")
    if epoch % 5 == 0:
        model.save_pretrained(f"runs/model_50_10_full_epoch_{epoch}_focal")

writer.close()