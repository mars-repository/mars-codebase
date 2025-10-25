import os
import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import timeit

model_name_or_path = "facebook/opt-125m"
context_window = 2048
device = "cuda"

# Determine padding side based on the model type
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
num_bins = 50  # Number of bins for classification
bin_width = 10  # Each bin covers 10 units

# Function to bin labels into fixed-width bins
def bin_labels(label):
    """Bin the continuous label into fixed-width bins."""
    return min(num_bins - 1, int(label / bin_width))  # Clamp values to the max bin index

# Apply binning to the labels
dataset = dataset.map(lambda x: {"label": bin_labels(x["label"])})

# Tokenization function for the prompt
def tokenize_function(examples):
    return tokenizer(examples["prompt"], truncation=True, max_length=context_window)

# Tokenize the dataset
tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["prompt", "completion", "api_token", "api_time", "api_name"],
)

# Rename the 'label' column to 'labels'
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

# Data collator for padding
def collate_fn(examples):
    return tokenizer.pad(examples, padding="longest", return_tensors="pt")

# Instantiate the dataloader
eval_dataloader = DataLoader(
    tokenized_datasets["test"],
    shuffle=False,
    collate_fn=collate_fn,
    batch_size=1
)

# Initialize the model for sequence classification
model = AutoModelForSequenceClassification.from_pretrained(
    model_name_or_path, num_labels=num_bins, return_dict=True
)
model.config.pad_token_id = tokenizer.pad_token_id

peft_model_path = "runs/model_50_10_full_epoch_15"
model.load_adapter(peft_model_path)
model.to(device)

# Set up the loss function
loss_fn = CrossEntropyLoss()

# Evaluation loop
model.eval()
total_eval_loss = 0
total_accuracy = 0
total_time = 0
all_predictions = []
all_labels = []
cnt = 0

for step, batch in enumerate(tqdm(eval_dataloader)):
    batch = {k: v.to(device) for k, v in batch.items()}
    batch["labels"] = batch["labels"].long()

    with torch.no_grad():
        a = timeit.default_timer()
        outputs = model(**batch)
        total_time += timeit.default_timer() - a
        cnt += 1
        if cnt % 100 == 0:
            print("Average time: ", total_time / cnt)
        predictions = outputs.logits

        eval_loss = loss_fn(predictions, batch["labels"])
        total_eval_loss += eval_loss.item()

        # Calculate accuracy
        preds = torch.argmax(predictions, dim=1)
        batch_accuracy = (preds == batch["labels"]).float().mean()
        # batch_accuracy = ((preds == batch["labels"]) + (preds == batch["labels"] - 1) + (preds == batch["labels"] + 1)).float().mean()
        total_accuracy += batch_accuracy.item()

        # Collect predictions and labels for the heatmap
        all_predictions.extend(preds.cpu().numpy())
        all_labels.extend(batch["labels"].cpu().numpy())

avg_eval_loss = total_eval_loss / len(eval_dataloader)
avg_accuracy = total_accuracy / len(eval_dataloader)

print(f"Average Evaluation Loss: {avg_eval_loss}, Accuracy: {avg_accuracy:.4f}")

save_dir = "evaluation_results_50_10_epoch15"
os.makedirs(save_dir, exist_ok=True)

# Save the predictions and labels as NumPy arrays
predictions_path = os.path.join(save_dir, "all_predictions.npy")
labels_path = os.path.join(save_dir, "all_labels.npy")

np.save(predictions_path, np.array(all_predictions))
np.save(labels_path, np.array(all_labels))

# Compute the confusion matrix
cm = confusion_matrix(all_labels, all_predictions, labels=range(num_bins))

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(cm, cmap="Blues", square=True)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.savefig("confusion_matrix_heatmap.pdf")
