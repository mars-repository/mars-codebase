"""
Predictor for completion length
"""

import torch
from peft import get_peft_model, LoraConfig, PeftType
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
import json
from tqdm import tqdm

class Predictor:
    def __init__(self):
        self.model_name_or_path = "facebook/opt-125m"
        if any(k in self.model_name_or_path for k in ("gpt", "opt", "bloom")):
            self.padding_side = "left"
        else:
            self.padding_side = "right"

        # Initialize the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, padding_side=self.padding_side)
        if getattr(self.tokenizer, "pad_token_id") is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name_or_path, num_labels=50, device_map="cuda:0")
        peft_model_path = "runs/model_50_10_full_epoch_15"
        self.model.load_adapter(peft_model_path)
        self.bin_width = 10

    def predict(self, input_text):
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1)

        return predicted_class * self.bin_width + self.bin_width // 2

if __name__ == "__main__":
    predictor = Predictor()
    input_text = "This is a test sentence"
    print(predictor.predict(input_text))