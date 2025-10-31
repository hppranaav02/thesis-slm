"""
Fine-tune Qwen2.5-Coder-1.5B for Golang CWE Classification
This script trains the model to identify CWE codes in Golang code.
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import pandas as pd
from typing import Dict, List
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter
import re
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # make physical cuda:1 show up as logical cuda:0
# optional, helps memory fragmentation on long runs:
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B"
MODEL_NAME = "/local/s3905020/notebooks/qwentest/qwen-cwe-classifier/checkpoint-483"
OUTPUT_DIR = "./qwen-cwe-classifier"
MAX_LENGTH = 2048
LEARNING_RATE = 2e-4
BATCH_SIZE = 1
NUM_EPOCHS = 3
GRADIENT_ACCUMULATION_STEPS = 16

class QwenCWEFineTuner:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.cwe_labels = []
        
    def _norm_cwe(self, x):
        # Normalize any input (int/float/str) to "CWE-<digits>" or "UNKNOWN"
        if x is None:
            return "UNKNOWN"
        if isinstance(x, (int, float)) and not pd.isna(x):
            return f"CWE-{int(x)}"
        s = str(x)
        m = re.search(r'(\d+)', s)
        return f"CWE-{m.group(1)}" if m else "UNKNOWN"

    def _label_sort_key(self, lbl):
        # Sort labels by CWE number, put UNKNOWN at the end
        m = re.search(r'(\d+)', lbl)
        return (0, int(m.group(1))) if m else (1, float('inf'))
    
    def load_model_and_tokenizer(self):
        print(f"Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map={'':0},
            trust_remote_code=True,
            # max_memory={1: "20GiB", "cpu": "30GiB"},
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                # llm_int8_enable_fp32_cpu_offload=True
            )
        )
        self.model = prepare_model_for_kbit_training(self.model)
        print("Model and tokenizer loaded successfully!")
        
    def setup_lora(self):
        """Configure LoRA for efficient fine-tuning"""
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
    def format_instruction(self, code: str, cwe_id: str = None) -> str:
        """Format the instruction for training"""
        if cwe_id:
            prompt = f"""### Task: Identify the CWE (Common Weakness Enumeration) ID in the following Golang code.

### Golang Code:
```go
{code}
```

### CWE ID: {cwe_id}"""
        else:
            prompt = f"""### Task: Identify the CWE (Common Weakness Enumeration) ID in the following Golang code.

### Golang Code:
```go
{code}
```

### CWE ID:"""
        return prompt
    
    def preprocess_dataset(self, dataset_path: str) -> Dataset:
        print(f"Loading dataset from: {dataset_path}")
        
        if dataset_path.endswith('.csv'):
            df = pd.read_csv(dataset_path)
        elif dataset_path.endswith('.json'):
            df = pd.read_json(dataset_path)
        elif dataset_path.endswith('.jsonl'):
            df = pd.read_json(dataset_path, lines=True)
        else:
            raise ValueError("Dataset must be CSV, JSON, or JSONL format")
        
        self.cwe_labels = sorted(df['cwe'].unique().tolist())
        print(f"Found {len(self.cwe_labels)} unique CWE labels: {self.cwe_labels}")
        formatted_data = []
        for _, row in df.iterrows():
            text = self.format_instruction(row['code'], row['cwe'])
            formatted_data.append({'text': text, 'cwe': row['cwe']})

        dataset = Dataset.from_pandas(pd.DataFrame(formatted_data))
        
        print(f"Dataset loaded: {len(dataset)} examples")

        cwe_counts = Counter(df['cwe'])
        print("\nLabel Distribution:")
        for cwe, count in cwe_counts.most_common():
            print(f"  {cwe}: {count}")
        
        return dataset
    
    def tokenize_function(self, examples: Dict) -> Dict:
        tokenized = self.tokenizer(
            examples['text'],
            truncation=True,
            max_length=MAX_LENGTH,
            padding='max_length',
            return_tensors=None
        )
        tokenized['labels'] = tokenized['input_ids'].copy()
        return tokenized
    
    def extract_cwe_from_output(self, text: str) -> str:
        m = re.search(r'cwe\s*[-:]?\s*(\d+)', text, flags=re.I)
        if m:
            return f"CWE-{int(m.group(1))}"
        return "UNKNOWN"
    
    def evaluate_model(self, test_data: List[Dict], output_dir: str = OUTPUT_DIR):
        print("\n=== Evaluating Model ===")

        # all_labels = sorted(set(y_true + y_pred), key=self._label_sort_key)
        y_true = []
        y_pred = []
        i = 0
        for item in test_data:
            code = item['code']
            true_cwe = item['cwe']
            pred_cwe = self.classify_code(code)
            y_true.append(true_cwe)
            y_pred.append(pred_cwe)

        # normalize
        print(y_pred[-1])
        y_true = [self._norm_cwe(t) for t in y_true]
        y_pred = [self._norm_cwe(p) for p in y_pred]

        self.plot_confusion_matrix(y_true, y_pred, output_dir)
        print("\n=== Classification Report ===")
        print(classification_report(y_true, y_pred, zero_division=0))
        accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
        print(f"\nOverall Accuracy: {accuracy:.2%}")
        
        return y_true, y_pred
    
    def plot_confusion_matrix(self, y_true: List[str], y_pred: List[str], output_dir: str):
        # all_labels = sorted(list(set(y_true + y_pred)))
        all_labels = sorted(set(y_true + y_pred), key=self._label_sort_key)
        
        cm = confusion_matrix(y_true, y_pred, labels=all_labels)
        plt.figure(figsize=(12, 10))
        
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=all_labels,
            yticklabels=all_labels,
            cbar_kws={'label': 'Count'}
        )
        
        plt.title('CWE Classification Confusion Matrix', fontsize=16, pad=20)
        plt.ylabel('True CWE', fontsize=12)
        plt.xlabel('Predicted CWE', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        save_path = f"{output_dir}/confusion_matrix.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
        
        plt.close()
        
        self.plot_normalized_confusion_matrix(y_true, y_pred, all_labels, output_dir)
    
    def plot_normalized_confusion_matrix(self, y_true: List[str], y_pred: List[str], 
                                        labels: List[str], output_dir: str):
        """Plot and save normalized confusion matrix"""
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(12, 10))
        
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={'label': 'Proportion'},
            vmin=0,
            vmax=1
        )
        
        plt.title('CWE Classification Confusion Matrix (Normalized)', fontsize=16, pad=20)
        plt.ylabel('True CWE', fontsize=12)
        plt.xlabel('Predicted CWE', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        save_path = f"{output_dir}/confusion_matrix_normalized.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Normalized confusion matrix saved to: {save_path}")
        
        plt.close()
    
    def train(self, dataset_path: str):
        dataset = self.preprocess_dataset(dataset_path)
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
        
        # training_args = TrainingArguments(
        #     output_dir=OUTPUT_DIR,
        #     num_train_epochs=NUM_EPOCHS,
        #     per_device_train_batch_size=BATCH_SIZE,
        #     per_device_eval_batch_size=BATCH_SIZE,
        #     gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        #     learning_rate=LEARNING_RATE,
        #     fp16=True,
        #     logging_steps=10,
        #     eval_strategy="steps",
        #     eval_steps=50,
        #     save_strategy="steps",
        #     save_steps=100,
        #     save_total_limit=3,
        #     load_best_model_at_end=True,
        #     warmup_steps=100,
        #     optim="paged_adamw_8bit",
        #     report_to="none"
        # )

        # data_collator = DataCollatorForLanguageModeling(
        #     tokenizer=self.tokenizer,
        #     mlm=False
        # )

        # trainer = Trainer(
        #     model=self.model,
        #     args=training_args,
        #     train_dataset=split_dataset['train'],
        #     eval_dataset=split_dataset['test'],
        #     data_collator=data_collator,
        # )
        
        # print("Starting training")
        # trainer.train()
        # print(f"Saving model to {OUTPUT_DIR}")
        # trainer.save_model(OUTPUT_DIR)
        # self.tokenizer.save_pretrained(OUTPUT_DIR)
        print("Training complete!")
        original_dataset = self.preprocess_dataset(dataset_path)
        test_indices = split_dataset['test'].select(range(min(100, len(split_dataset['test']))))
        test_data = []
        for idx in range(len(test_indices)):
            test_data.append({
                'code': original_dataset[idx]['text'].split('```go')[-1].split('```')[0].strip(),
                'cwe': original_dataset[idx]['cwe']
            })
        self.evaluate_model(test_data, OUTPUT_DIR)
    
    def classify_code(self, code: str) -> str:
        prompt = self.format_instruction(code)
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.1,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract CWE ID
        cwe_id = self.extract_cwe_from_output(result)
        
        return cwe_id


def main():
    fine_tuner = QwenCWEFineTuner()
    fine_tuner.load_model_and_tokenizer()
    fine_tuner.setup_lora()
    fine_tuner.train('/local/s3905020/notebooks/qwentest/cwe_code_before_min100.jsonl')
    
    print("\n=== Testing the fine-tuned model ===")
    test_code = """
package main

import (
    "database/sql"
    "fmt"
)

func getUserData(username string) {
    db, _ := sql.Open("mysql", "user:password@/dbname")
    query := "SELECT * FROM users WHERE username = '" + username + "'"
    rows, _ := db.Query(query)
    fmt.Println(rows)
}
"""
    
    result = fine_tuner.classify_code(test_code)
    print(f"\nPredicted CWE ID: {result}")


if __name__ == "__main__":
    main()