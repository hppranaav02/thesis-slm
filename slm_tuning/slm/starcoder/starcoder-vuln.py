"""
StarCoder-1B Vulnerability Classification for Go Code
======================================================
This script trains StarCoderBase-1B for binary classification of vulnerable Go code.
It includes memory-efficient training with LoRA, gradient checkpointing, and 8-bit optimization.
Architecture: Decoder-only Transformer with Multi-Query Attention (MQA)
"""

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType,
    prepare_model_for_kbit_training
)
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pandas as pd
from tqdm import tqdm

# Force CUDA device to cuda:1 throughout the script
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

CUSTOM_CACHE_DIR = "/local/s3905020/temp/"

# Hugging Face Transformers cache
os.environ["TRANSFORMERS_CACHE"] = CUSTOM_CACHE_DIR
# Hugging Face Datasets cache
os.environ["HF_DATASETS_CACHE"] = CUSTOM_CACHE_DIR
# Tokenizer cache
os.environ["HF_HOME"] = CUSTOM_CACHE_DIR

def stratified_split(jsonl_path, tokenizer, max_length=512, test_size=0.15, val_size=0.15, random_state=42):
    # Load and filter data
    filtered_samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            code = data["code"]
            label = int(data["is_vulnerable"])
            tokens = tokenizer(code, truncation=True, max_length=max_length)["input_ids"]
            if len(tokens) <= max_length:
                filtered_samples.append({"code": code, "is_vulnerable": label})
    # print(f"Filtered to {len(filtered_samples)} samples (<=512 tokens)")

    # Get labels for stratification
    labels = [s["is_vulnerable"] for s in filtered_samples]
    # Stratified split test and remaining
    trainval_samples, test_samples = train_test_split(
        filtered_samples, test_size=test_size, stratify=labels, random_state=random_state
    )
    trainval_labels = [s["is_vulnerable"] for s in trainval_samples]
    # Stratified split train and val from remaining
    val_fraction = val_size / (1 - test_size)
    train_samples, val_samples = train_test_split(
        trainval_samples, test_size=val_fraction, stratify=trainval_labels, random_state=random_state
    )

    return train_samples, val_samples, test_samples

class Config:
    HF_TOKEN = "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"
    MODEL_NAME = "bigcode/starcoderbase-1b"
    MAX_LENGTH = 512 
    BATCH_SIZE = 4  # Small batch size for memory efficiency
    GRADIENT_ACCUMULATION_STEPS = 8
    LEARNING_RATE = 2e-4
    NUM_EPOCHS = 10
    WARMUP_STEPS = 100
    WEIGHT_DECAY = 0.01
    LORA_R = 8
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05
    LORA_TARGET_MODULES = ["c_proj", "c_attn"]
    USE_GRADIENT_CHECKPOINTING = True
    USE_8BIT_OPTIMIZER = True
    FP16_TRAINING = True
    
    # Data configuration
    # TRAIN_DATA_PATH = "train_data.jsonl"
    # VAL_DATA_PATH = "val_data.jsonl"
    # TEST_DATA_PATH = "test_data.jsonl"
    INPUT_PATH = "./ds.jsonl"
    OUTPUT_DIR = "./starcoder_vuln_classifier"
    LOGGING_DIR = "./logs"
    SAVE_STEPS = 500
    EVAL_STEPS = 500
    SEED = 42

class VulnerabilityDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        print(f"Loading data from {file_path}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                data = json.loads(line.strip())
                self.samples.append({
                    'code': data['code'],
                    'label': int(data['is_vulnerable'])
                })
        
        print(f"Loaded {len(self.samples)} samples")
        labels = [s['label'] for s in self.samples]
        print(f"Class distribution - Vulnerable: {sum(labels)}, Non-vulnerable: {len(labels) - sum(labels)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Tokenize the code
        encoding = self.tokenizer(
            sample['code'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(sample['label'], dtype=torch.long)
        }

class StarCoderForClassification(torch.nn.Module):
    
    def __init__(self, model_name, num_labels=2, device='cuda:1'):
        super().__init__()
        self.num_labels = num_labels
        self.device_name = device
        
        print(f"\nInitializing StarCoder model on {device}")
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if Config.FP16_TRAINING else torch.float32,
            device_map={"": device},  # Force all layers to cuda:1
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            token=Config.HF_TOKEN
        )

        self.hidden_size = self.base_model.config.hidden_size
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        print(f"Base model loaded. Hidden size: {self.hidden_size}")

        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.1),
            torch.nn.Linear(self.hidden_size, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, num_labels)
        ).to(device)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing on the base model"""
        if hasattr(self.base_model, 'gradient_checkpointing_enable'):
            self.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
    
    def gradient_checkpointing_disable(self):
        if hasattr(self.base_model, 'gradient_checkpointing_disable'):
            self.base_model.gradient_checkpointing_disable()
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Extract hidden states from the last layer
        hidden_states = outputs.hidden_states[-1]  # Shape: (batch_size, seq_len, hidden_size)
        
        # Pool the hidden states (mean pooling across sequence)
        # Apply attention mask to ignore padding tokens
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_hidden = torch.sum(hidden_states * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        pooled_output = sum_hidden / sum_mask  # Shape: (batch_size, hidden_size)
        logits = self.classifier(pooled_output)  # Shape: (batch_size, num_labels)
        
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return {
            'loss': loss,
            'logits': logits
        }

def setup_training():
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    
    print("\nLoading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME, token=Config.HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("\nSplitting and filtering dataset")
    train_samples, val_samples, test_samples = stratified_split(
        Config.INPUT_PATH, tokenizer, Config.MAX_LENGTH
    )

    def save_jsonl(samples, path):
        with open(path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")

    save_jsonl(train_samples, "train_data.jsonl")
    save_jsonl(val_samples, "val_data.jsonl")
    save_jsonl(test_samples, "test_data.jsonl")

    train_dataset = VulnerabilityDataset("train_data.jsonl", tokenizer, Config.MAX_LENGTH)
    val_dataset = VulnerabilityDataset("val_data.jsonl", tokenizer, Config.MAX_LENGTH)
    test_dataset = VulnerabilityDataset("test_data.jsonl", tokenizer, Config.MAX_LENGTH)
    
    model = StarCoderForClassification(
        Config.MODEL_NAME,
        num_labels=2,
        device=str(DEVICE)
    )
    
    print("\nApplying LoRA (Parameter-Efficient Fine-Tuning)")
    lora_config = LoraConfig(
        r=Config.LORA_R,
        lora_alpha=Config.LORA_ALPHA,
        target_modules=Config.LORA_TARGET_MODULES,
        lora_dropout=Config.LORA_DROPOUT,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION
    )

    model.base_model = get_peft_model(model.base_model, lora_config)
    
    # Print trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")
    
    return model, tokenizer, train_dataset, val_dataset, test_dataset

class VulnerabilityTrainer(Trainer):   
    def create_optimizer(self):
        if Config.USE_8BIT_OPTIMIZER:
            try:
                import bitsandbytes as bnb
                print("Using 8-bit AdamW optimizer for memory efficiency...")
                self.optimizer = bnb.optim.AdamW8bit(
                    self.model.parameters(),
                    lr=self.args.learning_rate,
                    weight_decay=self.args.weight_decay
                )
            except ImportError:
                #TODO: Change this when bnb issue fix
                print("bitsandbytes not available, using standard AdamW...")
                super().create_optimizer()
        else:
            super().create_optimizer()

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Get probabilities for ROC-AUC
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', zero_division=0
    )
    accuracy = accuracy_score(labels, predictions)
    
    # Calculate ROC-AUC
    try:
        roc_auc = roc_auc_score(labels, probs)
    except:
        roc_auc = 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }

def evaluate_model(model, test_dataset, tokenizer, output_dir):
    model.eval()
    model = model.to(DEVICE)
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=Config.BATCH_SIZE,
        shuffle=False
    )
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels']
    
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs['logits']
            
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    roc_auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)
    
    print("CLASSIFICATION METRICS")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    
    print("CONFUSION MATRIX")
    print(f"                Predicted")
    print(f"                Non-Vuln  Vulnerable")
    print(f"Actual Non-Vuln    {cm[0,0]:4d}      {cm[0,1]:4d}")
    print(f"Actual Vulnerable  {cm[1,0]:4d}      {cm[1,1]:4d}")
    
    print("DETAILED CLASSIFICATION REPORT")
    print(classification_report(
        all_labels, all_preds, 
        target_names=['Non-Vulnerable', 'Vulnerable']
    ))
    
    create_visualizations(all_labels, all_preds, all_probs, cm, output_dir)
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'true_label': all_labels,
        'predicted_label': all_preds,
        'vulnerability_probability': all_probs
    })
    results_df.to_csv(f"{output_dir}/test_predictions.csv", index=False)
    print(f"\nPredictions saved to {output_dir}/test_predictions.csv")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm
    }


def create_visualizations(labels, preds, probs, cm, output_dir):
    os.makedirs(f"{output_dir}/visualizations", exist_ok=True)
    
    # Confusion Matrix Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Vulnerable', 'Vulnerable'],
                yticklabels=['Non-Vulnerable', 'Vulnerable'])
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/visualizations/confusion_matrix.png", dpi=300)
    plt.close()
    
    # ROC Curve
    fpr, tpr, thresholds = roc_curve(labels, probs)
    roc_auc = roc_auc_score(labels, probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/visualizations/roc_curve.png", dpi=300)
    plt.close()
    
    # rediction Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Vulnerable samples
    vuln_probs = probs[labels == 1]
    axes[0].hist(vuln_probs, bins=50, color='red', alpha=0.7, edgecolor='black')
    axes[0].axvline(0.5, color='black', linestyle='--', linewidth=2, label='Threshold = 0.5')
    axes[0].set_xlabel('Predicted Vulnerability Probability', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('Distribution for Vulnerable Samples', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Non-vulnerable samples
    non_vuln_probs = probs[labels == 0]
    axes[1].hist(non_vuln_probs, bins=50, color='green', alpha=0.7, edgecolor='black')
    axes[1].axvline(0.5, color='black', linestyle='--', linewidth=2, label='Threshold = 0.5')
    axes[1].set_xlabel('Predicted Vulnerability Probability', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].set_title('Distribution for Non-Vulnerable Samples', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/visualizations/prediction_distributions.png", dpi=300)
    plt.close()
    
    # Metrics Summary Bar Chart
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    accuracy = accuracy_score(labels, preds)
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    values = [accuracy, precision, recall, f1, roc_auc_score(labels, probs)]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, values, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6'],
                   edgecolor='black', linewidth=1.5)
    plt.ylim([0, 1.0])
    plt.ylabel('Score', fontsize=12)
    plt.title('Model Performance Metrics', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/visualizations/metrics_summary.png", dpi=300)
    plt.close()
    
    # 5. Precision-Recall Curve
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    precision_curve, recall_curve, _ = precision_recall_curve(labels, probs)
    avg_precision = average_precision_score(labels, probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall_curve, precision_curve, color='blue', lw=2,
             label=f'PR Curve (AP = {avg_precision:.4f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    plt.savefig(f"{output_dir}/visualizations/precision_recall_curve.png", dpi=300)
    plt.close()

def main():
    print("STARCODER-1B VULNERABILITY CLASSIFICATION TRAINING")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    model, tokenizer, train_dataset, val_dataset, test_dataset = setup_training()
    training_args = TrainingArguments(
        output_dir=Config.OUTPUT_DIR,
        num_train_epochs=Config.NUM_EPOCHS,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE,
        gradient_accumulation_steps=Config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        warmup_steps=Config.WARMUP_STEPS,
        logging_dir=Config.LOGGING_DIR,
        logging_steps=50,
        save_steps=Config.SAVE_STEPS,
        eval_steps=Config.EVAL_STEPS,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=Config.FP16_TRAINING,
        gradient_checkpointing=Config.USE_GRADIENT_CHECKPOINTING,
        report_to="tensorboard",
        seed=Config.SEED,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=False
    )
    
    # Initialize trainer
    trainer = VulnerabilityTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    train_result = trainer.train()
    
    # Save model
    trainer.model.base_model.save_pretrained(Config.OUTPUT_DIR)
    tokenizer.save_pretrained(Config.OUTPUT_DIR)

    torch.save(trainer.model.classifier.state_dict(), f"{Config.OUTPUT_DIR}/classifier_head.pt")
    
    # Print training summary
    print("TRAINING COMPLETED")
    print(f"Total training time: {train_result.metrics['train_runtime']:.2f} seconds")
    print(f"Training loss: {train_result.metrics['train_loss']:.4f}")
    
    test_metrics = evaluate_model(model, test_dataset, tokenizer, Config.OUTPUT_DIR)
    
    final_results = {
        'training_metrics': train_result.metrics,
        'test_metrics': test_metrics,
        'config': {
            'model': Config.MODEL_NAME,
            'max_length': Config.MAX_LENGTH,
            'batch_size': Config.BATCH_SIZE,
            'learning_rate': Config.LEARNING_RATE,
            'lora_r': Config.LORA_R,
            'lora_alpha': Config.LORA_ALPHA
        }
    }
    
    with open(f"{Config.OUTPUT_DIR}/final_results.json", 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nAll results saved to:", Config.OUTPUT_DIR)


if __name__ == "__main__":
    main()
