
"""
Ablation Study: Decision Threshold Impact
Tests different classification thresholds to find optimal precision-recall trade-off
"""

import os
import json
import random
import warnings
from pathlib import Path
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
    AutoTokenizer, AutoModel,
    get_linear_schedule_with_warmup
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, precision_recall_curve
)
from imblearn.under_sampling import RandomUnderSampler
from tqdm import tqdm

warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

OUTPUT_DIR = Path("./ablation_threshold")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

class Config:
    dataset_path = "./ds.jsonl"
    model_name = "microsoft/codebert-base"
    max_token_length = 512
    window_size = 256
    test_size = 0.2
    val_size = 0.1
    batch_size = 8
    learning_rate = 2e-5
    num_epochs = 10
    weight_decay = 0.01
    
    # Thresholds to test
    thresholds = np.arange(0.1, 0.96, 0.05)

config = Config()

class AsymmetricFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma_pos=2.0, gamma_neg=4.0):
        super().__init__()
        self.alpha = alpha
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
    
    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        gamma = torch.where(targets == 1,
                           torch.tensor(self.gamma_pos, device=targets.device),
                           torch.tensor(self.gamma_neg, device=targets.device))
        
        alpha_t = torch.where(targets == 1,
                             torch.tensor(self.alpha, device=targets.device),
                             torch.tensor(1 - self.alpha, device=targets.device))
        
        return (alpha_t * (1 - pt) ** gamma * ce_loss).mean()

class JSONLDataLoader:
    def __init__(self, config):
        self.config = config
    
    def load_jsonl(self, file_path):
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data.append(json.loads(line.strip()))
                except:
                    continue
        return data
    
    def preprocess_data(self, data):
        df = pd.DataFrame(data)
        df = df.dropna(subset=['code', 'is_vulnerable'])
        df['code'] = df['code'].astype(str)
        df['is_vulnerable'] = df['is_vulnerable'].astype(int)
        
        X = df.drop(columns=['is_vulnerable'])
        y = df['is_vulnerable']
        
        if len(y.unique()) > 1:
            sampler = RandomUnderSampler(random_state=42)
            X_res, y_res = sampler.fit_resample(X, y)
            df = pd.concat([X_res, y_res], axis=1)
        
        return df
    
    def filter_large_entries(self, df, tokenizer):
        def get_token_count(code):
            try:
                return len(tokenizer.encode(code, add_special_tokens=True))
            except:
                return float('inf')
        
        df['token_count'] = df['code'].apply(get_token_count)
        filtered_df = df[df['token_count'] <= self.config.max_token_length].copy()
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

class VulnerabilityClassifier(nn.Module):
    def __init__(self, model_name, dropout_rate=0.1):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(model_name)
        hidden_size = self.base_model.config.hidden_size
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, 2)
        nn.init.xavier_uniform_(self.classifier.weight)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        return {'logits': self.classifier(self.dropout(pooled))}

def train_model(train_loader, device):
    model = VulnerabilityClassifier(config.model_name).to(device)
    criterion = AsymmetricFocalLoss()
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    total_steps = len(train_loader) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=total_steps)
    
    # Training
    for epoch in range(1, config.num_epochs + 1):
        model.train()
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch}", leave=False):
            if batch is None:
                continue
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs['logits'], labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
    
    return model

def get_probabilities(model, test_loader, device):
    model.eval()
    all_probs, all_labels = [], []
    
    with torch.no_grad():
        for batch in test_loader:
            if batch is None:
                continue
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            probs = F.softmax(outputs['logits'], dim=1)
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_labels), np.array(all_probs)

def main():
    print("ABLATION STUDY: DECISION THRESHOLD IMPACT")
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    data_loader = JSONLDataLoader(config)
    
    raw_data = data_loader.load_jsonl(config.dataset_path)
    df = data_loader.preprocess_data(raw_data)
    filtered_df = data_loader.filter_large_entries(df, tokenizer)
    train_df, val_df, test_df = data_loader.stratified_split(filtered_df)
    
    # Create datasets
    train_dataset = VulnerabilityDataset(train_df, tokenizer, config.window_size)
    test_dataset = VulnerabilityDataset(test_df, tokenizer, config.window_size)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    
    model = train_model(train_loader, DEVICE)
    labels, probs = get_probabilities(model, test_loader, DEVICE)
    
    results = {}
    print("\nTesting different thresholds")
    for threshold in tqdm(config.thresholds, desc="Threshold sweep"):
        preds = (probs[:, 1] > threshold).astype(int)
        
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
        accuracy = accuracy_score(labels, preds)
        
        cm = confusion_matrix(labels, preds)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0
        
        fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        results[threshold] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            'fp_rate': fp_rate,
            'fn_rate': fn_rate
        }
    
    # Find best threshold by F1
    best_threshold = max(results.keys(), key=lambda t: results[t]['f1'])
    print(f"\nBest Threshold: {best_threshold:.2f} (F1={results[best_threshold]['f1']:.4f})")
    
    # Save results
    results_data = []
    for threshold, result in results.items():
        results_data.append({
            'Threshold': f"{threshold:.2f}",
            'Accuracy': f"{result['accuracy']:.4f}",
            'Precision': f"{result['precision']:.4f}",
            'Recall': f"{result['recall']:.4f}",
            'F1-Score': f"{result['f1']:.4f}",
            'FP_Rate': f"{result['fp_rate']:.4f}",
            'FN_Rate': f"{result['fn_rate']:.4f}",
            'TP': result['tp'], 'TN': result['tn'],
            'FP': result['fp'], 'FN': result['fn']
        })
    
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(OUTPUT_DIR / 'threshold_ablation.csv', index=False)
    print(f"Results saved to {OUTPUT_DIR / 'threshold_ablation.csv'}")
    
    # Visualizations
    thresholds = list(results.keys())
    
    # Precision-Recall curve
    fig, ax = plt.subplots(figsize=(10, 8))
    
    precisions = [results[t]['precision'] for t in thresholds]
    recalls = [results[t]['recall'] for t in thresholds]
    
    # precision-recall curve
    ax.plot(recalls, precisions, 'b-', linewidth=2, label='PR Curve')
    
    # Mark best F1 point
    best_prec = results[best_threshold]['precision']
    best_rec = results[best_threshold]['recall']
    ax.scatter([best_rec], [best_prec], color='red', s=200, zorder=5,
              label=f'Best F1 (threshold={best_threshold:.2f})')
    
    #common thresholds
    for t in [0.3, 0.5, 0.7]:
        if t in results:
            p = results[t]['precision']
            r = results[t]['recall']
            ax.scatter([r], [p], s=100, zorder=4)
            ax.annotate(f't={t:.1f}', (r, p), xytext=(5, 5), textcoords='offset points')
    
    ax.set_xlabel('Recall', fontweight='bold', fontsize=12)
    ax.set_ylabel('Precision', fontweight='bold', fontsize=12)
    ax.set_title('Precision-Recall Curve (Threshold Sweep)', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'precision_recall_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Metrics vs Threshold
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = [('precision', 'Precision'), ('recall', 'Recall'), 
              ('f1', 'F1-Score'), ('accuracy', 'Accuracy')]
    
    for idx, (metric, label) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        values = [results[t][metric] for t in thresholds]
        
        ax.plot(thresholds, values, linewidth=2, marker='o', markersize=4)
        ax.axvline(best_threshold, color='red', linestyle='--', linewidth=2, 
                  label=f'Best (t={best_threshold:.2f})')
        ax.set_xlabel('Decision Threshold', fontweight='bold')
        ax.set_ylabel(label, fontweight='bold')
        ax.set_title(f'{label} vs Threshold', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'metrics_vs_threshold.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Combined metrics plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for metric, label in [('precision', 'Precision'), ('recall', 'Recall'), ('f1', 'F1-Score')]:
        values = [results[t][metric] for t in thresholds]
        ax.plot(thresholds, values, linewidth=2, marker='o', markersize=4, label=label)
    
    ax.axvline(best_threshold, color='red', linestyle='--', linewidth=2, alpha=0.5,
              label=f'Best F1 Threshold={best_threshold:.2f}')
    ax.set_xlabel('Decision Threshold', fontweight='bold', fontsize=12)
    ax.set_ylabel('Score', fontweight='bold', fontsize=12)
    ax.set_title('Performance Metrics vs Decision Threshold', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'combined_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # FP vs FN rates
    fig, ax = plt.subplots(figsize=(12, 6))
    
    fp_rates = [results[t]['fp_rate'] for t in thresholds]
    fn_rates = [results[t]['fn_rate'] for t in thresholds]
    
    ax.plot(thresholds, fp_rates, linewidth=2, marker='o', markersize=4, 
           color='red', label='False Positive Rate')
    ax.plot(thresholds, fn_rates, linewidth=2, marker='s', markersize=4, 
           color='orange', label='False Negative Rate')
    ax.axvline(best_threshold, color='green', linestyle='--', linewidth=2, alpha=0.5,
              label=f'Best F1 Threshold={best_threshold:.2f}')
    
    ax.set_xlabel('Decision Threshold', fontweight='bold', fontsize=12)
    ax.set_ylabel('Error Rate', fontweight='bold', fontsize=12)
    ax.set_title('False Positive vs False Negative Rates', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'error_rates_vs_threshold.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Confusion matrix heatmap for selected thresholds
    selected_thresholds = [0.3, 0.5, best_threshold, 0.7]
    selected_thresholds = [t for t in selected_thresholds if t in results]
    
    fig, axes = plt.subplots(1, len(selected_thresholds), figsize=(5*len(selected_thresholds), 5))
    if len(selected_thresholds) == 1:
        axes = [axes]
    
    for idx, threshold in enumerate(selected_thresholds):
        result = results[threshold]
        cm = np.array([[result['tn'], result['fp']], [result['fn'], result['tp']]])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   xticklabels=['Non-Vuln', 'Vuln'],
                   yticklabels=['Non-Vuln', 'Vuln'])
        
        marker = "★ BEST" if threshold == best_threshold else ""
        axes[idx].set_title(f"Threshold = {threshold:.2f} {marker}\nF1={result['f1']:.3f}",
                          fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'confusion_matrices_selected.png', dpi=300, bbox_inches='tight')
    plt.close()
    

    print("THRESHOLD ABLATION COMPLETE")
    print(f"All results saved to {OUTPUT_DIR}")

    print(f"\nSUMMARY:")
    print(f"  Best Threshold: {best_threshold:.2f}")
    print(f"  Precision: {results[best_threshold]['precision']:.4f}")
    print(f"  Recall: {results[best_threshold]['recall']:.4f}")
    print(f"  F1-Score: {results[best_threshold]['f1']:.4f}")
    print(f"  FP Rate: {results[best_threshold]['fp_rate']:.4f}")
    print(f"  FN Rate: {results[best_threshold]['fn_rate']:.4f}")

if __name__ == "__main__":
    main()
