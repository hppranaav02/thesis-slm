import time
import json
import os
import re
from collections import Counter
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    cohen_kappa_score
)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # make physical cuda:1 show up as logical cuda:0
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B"
MODEL_NAME = "/local/s3905020/notebooks/qwentest/qwen-cwe-classifier-metrics/checkpoint-483"
OUTPUT_DIR = "./qwen-cwe-classifier-metrics"
MAX_LENGTH = 2048
LEARNING_RATE = 2e-4
BATCH_SIZE = 1
NUM_EPOCHS = 3
GRADIENT_ACCUMULATION_STEPS = 16
TOP_K_LIST = [1, 3, 5] 
GEN_MAX_NEW_TOKENS = 50
GEN_TEMPERATURE = 0.1 
GEN_TOP_P = 0.9
GEN_DO_SAMPLE = False
TOPK_TEMPERATURE = 0.7
TOPK_TOP_P = 0.95


class QwenCWEFineTuner:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.cwe_labels: List[str] = []
        self.label_encoder = None 

    def load_model_and_tokenizer(self):
        print(f"Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(torch.cuda.current_device())
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map={'': 0},
            trust_remote_code=True,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        )
        self.model = prepare_model_for_kbit_training(self.model)
        print(torch.cuda.current_device())
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
        """Format the instruction for training/inference"""
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

        if 'code' not in df.columns or 'cwe' not in df.columns:
            raise ValueError("Dataset must contain 'code' and 'cwe' columns")

        def _norm(lbl: str) -> str:
            m = re.search(r'(\d+)', str(lbl))
            return f"CWE-{m.group(1)}" if m else str(lbl)

        df['cwe'] = df['cwe'].apply(_norm)
        self.cwe_labels = sorted(df['cwe'].unique().tolist())
        print(f"Found {len(self.cwe_labels)} unique CWE labels.")

        formatted = []
        for _, row in df.iterrows():
            text = self.format_instruction(row['code'], row['cwe'])
            formatted.append({'text': text, 'code': row['code'], 'cwe': row['cwe']})
        dataset = Dataset.from_pandas(pd.DataFrame(formatted))
        print(f"Dataset loaded: {len(dataset)} examples")
        cwe_counts = Counter(df['cwe'])
        print("\nLabel Distribution (top 20):")
        for cwe, count in cwe_counts.most_common(20):
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

    @staticmethod
    def extract_cwe_from_output(text: str) -> str:
        m = re.search(r'CWE-\d+', text)
        if m:
            return m.group(0)

        if "### CWE ID:" in text:
            after = text.split("### CWE ID:")[-1].strip()
            m2 = re.search(r'(\d+)', after)
            if m2:
                return f"CWE-{m2.group(1)}"

        return "UNKNOWN"

    def _generate_sequences(
        self,
        prompt: str,
        num_return_sequences: int = 1,
        temperature: float = GEN_TEMPERATURE,
        top_p: float = GEN_TOP_P,
    ) -> Tuple[List[str], int]:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=5,
                # temperature=temperature,
                do_sample=GEN_DO_SAMPLE,
                # top_p=top_p,
                num_return_sequences=1,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # seqs = outputs.sequences
        # decoded = [self.tokenizer.decode(seqs[i], skip_special_tokens=True) for i in range(seqs.shape[0])]
        # gen_len_first = int(seqs.shape[1] - inputs['input_ids'].shape[1])
        # return decoded, gen_len_first
        seqs = outputs.sequences
        input_len = inputs['input_ids'].shape[1]
        # decode only the generated tail
        decoded_gen = [self.tokenizer.decode(seqs[i][input_len:], skip_special_tokens=True)
                    for i in range(seqs.shape[0])]
        gen_len_first = int(seqs.shape[1] - input_len)
        return decoded_gen, gen_len_first

    def _score_labels(self, code: str, labels=None, batch_size: int = 128):
        if labels is None:
            labels = self.cwe_labels  # freeze to known training labels
        prompt = self.format_instruction(code)
        # Tokenize prompt once
        prompt_ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)["input_ids"][0]

        scores = []
        with torch.no_grad():
            for i in range(0, len(labels), batch_size):
                chunk = labels[i:i+batch_size]
                # Build batch of prompt+label; mask loss on prompt
                texts = [prompt + lbl for lbl in chunk]
                toks = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH).to(self.model.device)
                input_ids = toks["input_ids"]
                attn = toks["attention_mask"]
                # Create labels with -100 on prompt part
                # Find prompt length per row (assumes identical prompt tokens across rows)
                plen = prompt_ids.shape[0]
                labels_mat = input_ids.clone()
                labels_mat[:, :plen] = -100
                out = self.model(input_ids=input_ids, attention_mask=attn, labels=labels_mat)
                # out.loss is mean over tokens per row; collect per example losses
                # HF returns scalar loss; compute token-level nll manually for accuracy:
                logits = out.logits  # [B, T, V]
                shift_logits = logits[:, :-1, :]
                shift_labels = labels_mat[:, 1:]
                # cross-entropy only where labels != -100
                loss_mask = (shift_labels != -100)
                # gather logits at target tokens
                tgt = shift_labels.clone()
                tgt[~loss_mask] = 0
                logp = torch.log_softmax(shift_logits, dim=-1)
                nll = -logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                nll = (nll * loss_mask).sum(dim=1) / loss_mask.sum(dim=1).clamp_min(1)
                scores.extend(nll.tolist())
        # lower is better
        return list(zip(labels, scores))

    def classify_code(self, code: str) -> str:
        scored = self._score_labels(code)
        scored.sort(key=lambda x: x[1])
        return scored[0][0]

    def classify_code_topk(self, code: str, k: int) -> list[str]:
        scored = self._score_labels(code)
        scored.sort(key=lambda x: x[1])
        return [lbl for lbl, _ in scored[:k]]


    def _annotate_cm(self, ax, cm, fmt):
        """Annotate a confusion matrix heatmap with numbers."""
        nrows, ncols = cm.shape
        for i in range(nrows):
            for j in range(ncols):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center")

    def plot_confusion_matrices(self, y_true: List[str], y_pred: List[str], labels: List[str], output_dir: str):
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(cm, aspect='auto')
        ax.set_title('CWE Classification Confusion Matrix', pad=20)
        ax.set_xlabel('Predicted CWE'); ax.set_ylabel('True CWE')
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Count')
        self._annotate_cm(ax, cm, 'd')
        fig.tight_layout()
        save_path = f"{output_dir}/confusion_matrix.png"
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Confusion matrix saved to: {save_path}")

        # Normalized
        cm_norm = cm.astype('float')
        row_sums = cm_norm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm_norm = cm_norm / row_sums

        fig2, ax2 = plt.subplots(figsize=(12, 10))
        im2 = ax2.imshow(cm_norm, vmin=0, vmax=1, aspect='auto')
        ax2.set_title('CWE Classification Confusion Matrix (Normalized)', pad=20)
        ax2.set_xlabel('Predicted CWE'); ax2.set_ylabel('True CWE')
        ax2.set_xticks(range(len(labels))); ax2.set_xticklabels(labels, rotation=45, ha='right')
        ax2.set_yticks(range(len(labels))); ax2.set_yticklabels(labels)
        cbar2 = fig2.colorbar(im2, ax=ax2)
        cbar2.set_label('Proportion')
        self._annotate_cm(ax2, cm_norm, '.2f')
        fig2.tight_layout()
        save_path2 = f"{output_dir}/confusion_matrix_normalized.png"
        fig2.savefig(save_path2, dpi=300, bbox_inches='tight')
        plt.close(fig2)
        print(f"Normalized confusion matrix saved to: {save_path2}")

    def plot_per_class_f1(self, per_class: Dict[str, Dict[str, float]], output_dir: str):
        labels = list(per_class.keys())
        f1s = [per_class[lbl].get('f1-score', 0.0) for lbl in labels]
        order = np.argsort(f1s)[::-1]
        labels_sorted = [labels[i] for i in order]
        f1_sorted = [f1s[i] for i in order]

        fig, ax = plt.subplots(figsize=(10, max(6, 0.35 * len(labels_sorted))))
        ax.barh(labels_sorted, f1_sorted)
        ax.set_xlabel("F1-score"); ax.set_ylabel("CWE")
        ax.set_title("Per-class F1 (sorted)", pad=12)
        ax.invert_yaxis() 
        fig.tight_layout()
        path = f"{output_dir}/per_class_f1_sorted.png"
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Per-class F1 plot saved to: {path}")

    def plot_topk_curve(self, topk_acc: Dict[int, float], output_dir: str):
        ks = sorted(topk_acc.keys())
        vals = [topk_acc[k] for k in ks]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(ks, vals, marker='o')
        ax.set_xlabel("k"); ax.set_ylabel("Top-k Accuracy")
        ax.set_title("Top-k Accuracy Curve", pad=12)
        ax.grid(True, linestyle='--', alpha=0.4)
        fig.tight_layout()
        path = f"{output_dir}/topk_accuracy.png"
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Top-k accuracy curve saved to: {path}")

    def plot_hist(self, values: List[float], title: str, xlabel: str, output_dir: str, filename: str, bins: int = 30):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(values, bins=bins)
        ax.set_title(title, pad=12)
        ax.set_xlabel(xlabel); ax.set_ylabel("Count")
        fig.tight_layout()
        path = f"{output_dir}/{filename}"
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"{title} saved to: {path}")

    def evaluate_model(
        self,
        test_dataset: Dataset,
        output_dir: str = OUTPUT_DIR,
        top_k_list: List[int] = TOP_K_LIST
    ):
        print("\n=== Evaluating Model ===")
        os.makedirs(output_dir, exist_ok=True)
        y_true: List[str] = []
        y_pred_top1: List[str] = []

        topk_hits = {k: 0 for k in top_k_list}
        max_k = max(top_k_list)
        # latency/tokens per sample
        latencies_s: List[float] = []
        gen_tokens_top1: List[int] = []
        # track VRAM
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0_all = time.time()
        rows = []

        for i in range(len(test_dataset)):
            code = test_dataset[i]['code']
            true_cwe = test_dataset[i]['cwe']
            y_true.append(true_cwe)
            start = time.time()
            outs_top1, gen_len_first = self._generate_sequences(
                self.format_instruction(code),
                num_return_sequences=1,
            )
            latency = time.time() - start
            latencies_s.append(latency)
            gen_tokens_top1.append(gen_len_first)
            pred1 = self.extract_cwe_from_output(outs_top1[0])
            y_pred_top1.append(pred1)
            outs_k, _ = self._generate_sequences(
                self.format_instruction(code),
                num_return_sequences=max_k,
                temperature=TOPK_TEMPERATURE,
                top_p=TOPK_TOP_P
            )
            preds_k = []
            for o in outs_k:
                c = self.extract_cwe_from_output(o)
                if c not in preds_k:
                    preds_k.append(c)
                if len(preds_k) >= max_k:
                    break
            for k in top_k_list:
                if true_cwe in preds_k[:k]:
                    topk_hits[k] += 1

            rows.append({
                "index": i,
                "true_cwe": true_cwe,
                "pred_top1": pred1,
                "latency_s": latency,
                "gen_tokens_top1": gen_len_first
            })

        total_time = time.time() - t0_all
        throughput = len(test_dataset) / total_time if total_time > 0 else float('inf')

        max_mem_alloc = None
        max_mem_res = None
        if torch.cuda.is_available():
            max_mem_alloc = torch.cuda.max_memory_allocated() / (1024**2)
            max_mem_res = torch.cuda.max_memory_reserved() / (1024**2)


        per_sample_path = f"{output_dir}/inference_per_sample.csv"
        pd.DataFrame(rows).to_csv(per_sample_path, index=False)
        print(f"Per-sample inference metrics saved to: {per_sample_path}")
        # After you finish collecting y_true and y_pred_top1:
        allowed = sorted(set(self.cwe_labels) | set(y_true))
        if "UNKNOWN" not in allowed:
            allowed.append("UNKNOWN")

        # Map any prediction not in your known training labels to UNKNOWN
        y_pred_top1 = [p if p in self.cwe_labels else "UNKNOWN" for p in y_pred_top1]

        # Fit the encoder ONLY on allowed labels
        self.label_encoder = LabelEncoder().fit(allowed)
        y_true_enc = self.label_encoder.transform(y_true)
        y_pred_enc = self.label_encoder.transform(y_pred_top1)

        # Use the frozen label list for the report
        cls_report = classification_report(
            y_true, y_pred_top1, labels=allowed, output_dict=True, zero_division=0
        )

        all_labels = sorted(list(set(self.cwe_labels) | set(y_true) | set(y_pred_top1)))
        # if "UNKNOWN" not in all_labels:
        #     all_labels.append("UNKNOWN")
        # self.label_encoder = LabelEncoder().fit(all_labels)
        # y_true_enc = self.label_encoder.transform(y_true)
        # y_pred_enc = self.label_encoder.transform([p if p in all_labels else "UNKNOWN" for p in y_pred_top1])
        acc = accuracy_score(y_true, y_pred_top1)
        micro_f1 = f1_score(y_true, y_pred_top1, average='micro', zero_division=0)
        macro_f1 = f1_score(y_true, y_pred_top1, average='macro', zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred_top1, average='weighted', zero_division=0)
        bal_acc = balanced_accuracy_score(y_true, y_pred_top1)
        mcc = matthews_corrcoef(y_true_enc, y_pred_enc)
        kappa = cohen_kappa_score(y_true_enc, y_pred_enc)
        cls_report = classification_report(
            y_true, y_pred_top1, labels=all_labels, output_dict=True, zero_division=0
        )
        per_class = {lbl: cls_report[lbl] for lbl in all_labels if lbl in cls_report}
        topk_acc = {k: topk_hits[k] / len(test_dataset) for k in top_k_list}

        lat_arr = np.array(latencies_s)
        tok_arr = np.array(gen_tokens_top1)
        lat_summary = {
            "mean_s": float(lat_arr.mean()) if len(lat_arr) else None,
            "p50_s": float(np.percentile(lat_arr, 50)) if len(lat_arr) else None,
            "p95_s": float(np.percentile(lat_arr, 95)) if len(lat_arr) else None,
            "p99_s": float(np.percentile(lat_arr, 99)) if len(lat_arr) else None,
            "min_s": float(lat_arr.min()) if len(lat_arr) else None,
            "max_s": float(lat_arr.max()) if len(lat_arr) else None,
        }
        tok_summary = {
            "mean": float(tok_arr.mean()) if len(tok_arr) else None,
            "p50": float(np.percentile(tok_arr, 50)) if len(tok_arr) else None,
            "p95": float(np.percentile(tok_arr, 95)) if len(tok_arr) else None,
            "min": int(tok_arr.min()) if len(tok_arr) else None,
            "max": int(tok_arr.max()) if len(tok_arr) else None,
        }

        run_metrics = {
            "num_samples": len(test_dataset),
            "accuracy": acc,
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "balanced_accuracy": bal_acc,
            "mcc": mcc,
            "cohen_kappa": kappa,
            "topk_accuracy": topk_acc,
            "latency": lat_summary,
            "tokens_generated_per_sample": tok_summary,
            "throughput_samples_per_s": throughput,
            "vram": {
                "max_allocated_mb": float(max_mem_alloc) if max_mem_alloc is not None else None,
                "max_reserved_mb": float(max_mem_res) if max_mem_res is not None else None
            }
        }
        with open(f"{output_dir}/metrics.json", "w") as f:
            json.dump(run_metrics, f, indent=2)
        print(f"Run-level metrics saved to: {output_dir}/metrics.json")

        per_class_rows = []
        for lbl, vals in per_class.items():
            per_class_rows.append({
                "cwe": lbl,
                "precision": vals.get("precision", 0.0),
                "recall": vals.get("recall", 0.0),
                "f1": vals.get("f1-score", 0.0),
                "support": int(vals.get("support", 0))
            })
        per_class_path = f"{output_dir}/per_class_metrics.csv"
        pd.DataFrame(per_class_rows).to_csv(per_class_path, index=False)
        print(f"Per-class metrics saved to: {per_class_path}")

        self.plot_confusion_matrices(y_true, y_pred_top1, all_labels, output_dir)
        self.plot_per_class_f1(per_class, output_dir)
        self.plot_topk_curve(topk_acc, output_dir)
        self.plot_hist(latencies_s, "Latency per sample", "seconds", output_dir, "latency_hist.png")
        self.plot_hist(gen_tokens_top1, "Generated tokens per sample (top-1)", "tokens", output_dir, "tokens_hist.png")
        print("\n=== Classification Report (Top-1) ===")
        print(classification_report(y_true, y_pred_top1, zero_division=0))

        print(f"\nOverall Accuracy (Top-1): {acc:.2%}")
        for k in top_k_list:
            print(f"Top-{k} Accuracy: {topk_acc[k]:.2%}")
        if max_mem_alloc is not None:
            print(f"Peak VRAM allocated: {max_mem_alloc:.1f} MB | reserved: {max_mem_res:.1f} MB")
        print(f"Throughput: {throughput:.2f} samples/sec")

        return y_true, y_pred_top1, run_metrics

    def train(self, dataset_path: str):
        print(torch.cuda.current_device())
        dataset = self.preprocess_dataset(dataset_path)
        split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
        tokenized_split = split_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=split_dataset['train'].column_names
        )
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
        #     train_dataset=tokenized_split['train'],
        #     eval_dataset=tokenized_split['test'],
        #     data_collator=data_collator,
        # )

        print("Starting training...")
        # print(torch.cuda.current_device())
        # trainer.train()
        # print(f"Saving model to {OUTPUT_DIR}")
        # trainer.save_model(OUTPUT_DIR)
        # self.tokenizer.save_pretrained(OUTPUT_DIR)
        # print("Training complete!")
        test_raw = split_dataset['test']
        self.evaluate_model(test_raw, OUTPUT_DIR, TOP_K_LIST)


def main():
    print(torch.cuda.current_device())
    fine_tuner = QwenCWEFineTuner()
    fine_tuner.load_model_and_tokenizer()
    fine_tuner.setup_lora()
    fine_tuner.train('/local/s3905020/notebooks/qwentest/cwe_code_before_min100.jsonl')

    print("\n=== Testing the fine-tuned model (single example) ===")
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
    topk = fine_tuner.classify_code_topk(test_code, k=max(TOP_K_LIST))
    print(f"Top-{max(TOP_K_LIST)} candidates: {topk}")

if __name__ == "__main__":
    main()
