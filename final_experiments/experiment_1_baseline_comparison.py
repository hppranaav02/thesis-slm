
"""
Experiment 1: Baseline Model Comparison
"""

import os
import json
import random
import warnings
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    RobertaModel, T5EncoderModel,
    get_linear_schedule_with_warmup
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
    average_precision_score, roc_auc_score,
    matthews_corrcoef
)
from imblearn.under_sampling import RandomUnderSampler
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Set device to cuda:1
DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Create output directory
OUTPUT_DIR = Path("./experiment_1_baseline_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Configuration
class Config:
    dataset_path = "./ds.jsonl"
    max_token_length = 512
    window_size = 256
    test_size = 0.2
    val_size = 0.1
    
    models = [
        "microsoft/codebert-base",
        "microsoft/graphcodebert-base",
        "Salesforce/codet5-base"
    ]
    
    batch_size = 8
    learning_rate = 2e-5
    num_epochs = 10
    weight_decay = 0.01
    warmup_steps = 500

config = Config()

# Data loader
class JSONLDataLoader:
    def __init__(self, config):
        self.config = config
    
    def load_jsonl(self, file_path):
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
        return data
    
    def preprocess_data(self, data):
        df = pd.DataFrame(data)
        df = df.dropna(subset=['code', 'is_vulnerable'])
        df['code'] = df['code'].astype(str)
        df['is_vulnerable'] = df['is_vulnerable'].astype(int)
        
        for col in ['cwe_ids', 'nloc', 'complexity']:
            if col not in df.columns:
                df[col] = None

        # Undersample since one class is bigger than the other
        X = df.drop(columns=['is_vulnerable'])
        y = df['is_vulnerable']
        
        if len(y.unique()) > 1:
            sampler = RandomUnderSampler(random_state=42)
            X_res, y_res = sampler.fit_resample(X, y)
            df = pd.concat([X_res, y_res], axis=1)
        
        print(f"Loaded and balanced: {len(df)} samples")
        return df
    
    def filter_large_entries(self, df, tokenizer):
        def get_token_count(code):
            try:
                return len(tokenizer.encode(code, add_special_tokens=True))
            except:
                return float('inf')
        
        df['token_count'] = df['code'].apply(get_token_count)
        filtered_df = df[df['token_count'] <= self.config.max_token_length].copy()
        print(f"After filtering: {len(filtered_df)}/{len(df)} samples")
        return filtered_df.drop('token_count', axis=1)
    
    def stratified_split(self, df):
        X = df.drop('is_vulnerable', axis=1)
        y = df['is_vulnerable']
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=self.config.test_size, stratify=y, random_state=42
        )
        
        val_size_adjusted = self.config.val_size / (1 - self.config.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, stratify=y_temp, random_state=42
        )
        
        return (pd.concat([X_train, y_train], axis=1),
                pd.concat([X_val, y_val], axis=1),
                pd.concat([X_test, y_test], axis=1))

# Dataset
class VulnerabilityDataset(Dataset):
    def __init__(self, df, tokenizer, window_size):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.window_size = window_size
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        code = str(row['code'])
        label = int(row['is_vulnerable'])
        
        if not code or len(code.strip()) == 0:
            return None
        
        try:
            encoding = self.tokenizer(
                code, add_special_tokens=True, max_length=self.window_size,
                padding='max_length', truncation=True, return_tensors='pt'
            )
        except:
            return None
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    return torch.utils.data.default_collate(batch) if batch else None

# Model
class VulnerabilityClassifier(nn.Module):
    def __init__(self, model_name, dropout_rate=0.1):
        super().__init__()
        
        if 'codet5' in model_name.lower():
            self.base_model = T5EncoderModel.from_pretrained(model_name)
            hidden_size = self.base_model.config.d_model
        elif 'roberta' in model_name.lower():
            self.base_model = RobertaModel.from_pretrained(model_name)
            hidden_size = self.base_model.config.hidden_size
        else:
            self.base_model = AutoModel.from_pretrained(model_name)
            hidden_size = self.base_model.config.hidden_size
        
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, 2)
        nn.init.xavier_uniform_(self.classifier.weight)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        return {'logits': self.classifier(self.dropout(pooled))}

# Training and evaluation
def train_epoch(model, dataloader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    predictions, labels = [], []
    
    for batch in tqdm(dataloader, desc="Training"):
        if batch is None:
            continue
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        batch_labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs['logits'], batch_labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        preds = torch.argmax(outputs['logits'], dim=1)
        predictions.extend(preds.cpu().numpy())
        labels.extend(batch_labels.cpu().numpy())
    
    return total_loss / len(dataloader), accuracy_score(labels, predictions)

def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            if batch is None:
                continue
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            probs = F.softmax(outputs['logits'], dim=1)
            preds = torch.argmax(outputs['logits'], dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

# Main experiment
def main():
    print("EXPERIMENT 1: BASELINE MODEL COMPARISON")
    
    # Load data
    data_loader = JSONLDataLoader(config)
    raw_data = data_loader.load_jsonl(config.dataset_path)
    df = data_loader.preprocess_data(raw_data)
    
    results = {}
    
    for model_name in config.models:
        print(f"Training {model_name}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Prepare data
        filtered_df = data_loader.filter_large_entries(df, tokenizer)
        train_df, val_df, test_df = data_loader.stratified_split(filtered_df)
        
        # Create datasets
        train_dataset = VulnerabilityDataset(train_df, tokenizer, config.window_size)
        test_dataset = VulnerabilityDataset(test_df, tokenizer, config.window_size)
        
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                                 shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, 
                                shuffle=False, collate_fn=collate_fn)
        
        # Initialize model
        model = VulnerabilityClassifier(model_name).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(model.parameters(), lr=config.learning_rate, 
                         weight_decay=config.weight_decay)
        
        total_steps = len(train_loader) * config.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=config.warmup_steps, 
            num_training_steps=total_steps
        )
        
        # Training
        for epoch in range(1, config.num_epochs + 1):
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, scheduler, DEVICE
            )
            print(f"Epoch {epoch}: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
        
        # Evaluation
        labels, preds, probs = evaluate(model, test_loader, DEVICE)
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average='weighted'
        )
        accuracy = accuracy_score(labels, preds)
        pr_auc = average_precision_score(labels, probs[:, 1])
        
        try:
            roc_auc = roc_auc_score(labels, probs[:, 1])
        except:
            roc_auc = 0.0
        
        mcc = matthews_corrcoef(labels, preds)
        
        cm = confusion_matrix(labels, preds)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0
        
        # Store results
        results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'pr_auc': pr_auc,
            'roc_auc': roc_auc,
            'mcc': mcc,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            'labels': labels,
            'preds': preds
        }
        
        print(f"\nResults: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
        print(f"Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    
    # Save results to CSV
    results_data = []
    for model_name, result in results.items():
        results_data.append({
            'Model': model_name.split('/')[-1],
            'Accuracy': f"{result['accuracy']:.4f}",
            'Precision': f"{result['precision']:.4f}",
            'Recall': f"{result['recall']:.4f}",
            'F1-Score': f"{result['f1']:.4f}",
            'PR-AUC': f"{result['pr_auc']:.4f}",
            'ROC-AUC': f"{result['roc_auc']:.4f}",
            'MCC': f"{result['mcc']:.4f}",
            'TP': result['tp'],
            'TN': result['tn'],
            'FP': result['fp'],
            'FN': result['fn']
        })
    
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(OUTPUT_DIR / 'baseline_comparison_results.csv', index=False)
    print(f"\nResults saved to {OUTPUT_DIR / 'baseline_comparison_results.csv'}")
    
    # Visualizations
    # Confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, (model_name, result) in enumerate(results.items()):
        cm = confusion_matrix(result['labels'], result['preds'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   xticklabels=['Non-Vuln', 'Vuln'],
                   yticklabels=['Non-Vuln', 'Vuln'])
        axes[idx].set_title(f"{model_name.split('/')[-1]}\nPrec: {result['precision']:.3f} Rec: {result['recall']:.3f}",
                          fontweight='bold')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrices saved")
    
    # Metrics comparison
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    x = np.arange(len(results))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, metric in enumerate(metrics):
        values = [results[m][metric] for m in results.keys()]
        ax.bar(x + i*width, values, width, label=metric.capitalize())
    
    ax.set_xlabel('Models', fontweight='bold')
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Baseline Model Comparison - Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([m.split('/')[-1] for m in results.keys()], rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Metrics comparison saved")
    
    # PR-AUC and ROC-AUC comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    models = [m.split('/')[-1] for m in results.keys()]
    pr_aucs = [results[m]['pr_auc'] for m in results.keys()]
    roc_aucs = [results[m]['roc_auc'] for m in results.keys()]
    
    x = np.arange(len(models))
    width = 0.35
    
    ax.bar(x - width/2, pr_aucs, width, label='PR-AUC', alpha=0.8)
    ax.bar(x + width/2, roc_aucs, width, label='ROC-AUC', alpha=0.8)
    
    ax.set_xlabel('Models', fontweight='bold')
    ax.set_ylabel('AUC Score', fontweight='bold')
    ax.set_title('AUC Metrics Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'auc_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"AUC comparison saved")
    
    print("EXPERIMENT 1 COMPLETE")
    print(f"All results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
