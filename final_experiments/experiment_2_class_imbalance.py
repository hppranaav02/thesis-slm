
"""
Experiment 2: Class Imbalance Handling
Compares different approaches: No balancing, Undersampling, Standard Focal Loss, Asymmetric Focal Loss
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

# Set device to cuda:1
DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

OUTPUT_DIR = Path("./experiment_2_class_imbalance")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# Configuration
class Config:
    dataset_path = "./ds.jsonl"
    model_name = "microsoft/graphcodebert-base"  # Best from Exp 1
    max_token_length = 512
    window_size = 256
    test_size = 0.2
    val_size = 0.1
    batch_size = 8
    learning_rate = 2e-5
    num_epochs = 10
    weight_decay = 0.01

config = Config()

# Loss functions
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

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

# Data components
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
    
    def preprocess_data(self, data, use_undersampling=True):
        df = pd.DataFrame(data)
        df = df.dropna(subset=['code', 'is_vulnerable'])
        df['code'] = df['code'].astype(str)
        df['is_vulnerable'] = df['is_vulnerable'].astype(int)
        
        if use_undersampling:
            X = df.drop(columns=['is_vulnerable'])
            y = df['is_vulnerable']
            
            if len(y.unique()) > 1:
                sampler = RandomUnderSampler(random_state=42)
                X_res, y_res = sampler.fit_resample(X, y)
                df = pd.concat([X_res, y_res], axis=1)
        
        print(f"Dataset size: {len(df)} samples")
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

def train_and_evaluate(train_loader, test_loader, criterion, device):
    model = VulnerabilityClassifier(config.model_name).to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    total_steps = len(train_loader) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=total_steps)
    
    # Training
    for epoch in range(1, config.num_epochs + 1):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
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
            total_loss += loss.item()
        
        print(f"Epoch {epoch}: Loss={total_loss/len(train_loader):.4f}")
    
    # Evaluation
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    
    with torch.no_grad():
        for batch in test_loader:
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

def main():
    print("EXPERIMENT 2: CLASS IMBALANCE HANDLING")
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    data_loader = JSONLDataLoader(config)
    raw_data = data_loader.load_jsonl(config.dataset_path)
    
    # Experiments
    experiments = {
        'No Balancing': {
            'use_undersampling': False,
            'criterion': nn.CrossEntropyLoss()
        },
        'Undersampling': {
            'use_undersampling': True,
            'criterion': nn.CrossEntropyLoss()
        },
        'Standard Focal Loss': {
            'use_undersampling': True,
            'criterion': FocalLoss(alpha=0.25, gamma=2.0)
        },
        'Asymmetric Focal Loss': {
            'use_undersampling': True,
            'criterion': AsymmetricFocalLoss(alpha=0.25, gamma_pos=2.0, gamma_neg=4.0)
        }
    }
    
    results = {}
    
    for exp_name, exp_config in experiments.items():
        print(f"Running: {exp_name}")
        
        # Prepare data
        df = data_loader.preprocess_data(raw_data, use_undersampling=exp_config['use_undersampling'])
        filtered_df = data_loader.filter_large_entries(df, tokenizer)
        train_df, val_df, test_df = data_loader.stratified_split(filtered_df)
        
        train_dataset = VulnerabilityDataset(train_df, tokenizer, config.window_size)
        test_dataset = VulnerabilityDataset(test_df, tokenizer, config.window_size)
        
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
        
        # Train and evaluate
        labels, preds, probs = train_and_evaluate(train_loader, test_loader, exp_config['criterion'], DEVICE)
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
        cm = confusion_matrix(labels, preds)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
        
        fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        results[exp_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            'fp_rate': fp_rate,
            'fn_rate': fn_rate,
            'labels': labels,
            'preds': preds,
            'probs': probs
        }
        
        print(f"Results: P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}")
        print(f"FP Rate={fp_rate:.4f}, FN Rate={fn_rate:.4f}")
    
    # Save results
    results_data = []
    for exp_name, result in results.items():
        results_data.append({
            'Method': exp_name,
            'Precision': f"{result['precision']:.4f}",
            'Recall': f"{result['recall']:.4f}",
            'F1-Score': f"{result['f1']:.4f}",
            'FP Rate': f"{result['fp_rate']:.4f}",
            'FN Rate': f"{result['fn_rate']:.4f}",
            'TP': result['tp'],
            'TN': result['tn'],
            'FP': result['fp'],
            'FN': result['fn']
        })
    
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(OUTPUT_DIR / 'imbalance_handling_results.csv', index=False)
    print(f"\nResults saved to {OUTPUT_DIR / 'imbalance_handling_results.csv'}")
    
    # Visualizations
    # Confusion matrices
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for idx, (exp_name, result) in enumerate(results.items()):
        cm = confusion_matrix(result['labels'], result['preds'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   xticklabels=['Non-Vuln', 'Vuln'],
                   yticklabels=['Non-Vuln', 'Vuln'])
        axes[idx].set_title(f"{exp_name}\nFP Rate: {result['fp_rate']:.3f}", fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'confusion_matrices_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # FP Rate comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    methods = list(results.keys())
    fp_rates = [results[m]['fp_rate'] for m in methods]
    fn_rates = [results[m]['fn_rate'] for m in methods]
    
    x = np.arange(len(methods))
    width = 0.35
    
    ax.bar(x - width/2, fp_rates, width, label='False Positive Rate', alpha=0.8, color='red')
    ax.bar(x + width/2, fn_rates, width, label='False Negative Rate', alpha=0.8, color='orange')
    
    ax.set_xlabel('Methods', fontweight='bold')
    ax.set_ylabel('Error Rate', fontweight='bold')
    ax.set_title('False Positive vs False Negative Rates', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'error_rates_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Precision-Recall curves
    fig, ax = plt.subplots(figsize=(10, 8))
    for exp_name, result in results.items():
        precision_curve, recall_curve, _ = precision_recall_curve(
            result['labels'], result['probs'][:, 1]
        )
        ax.plot(recall_curve, precision_curve, label=exp_name, linewidth=2)
    
    ax.set_xlabel('Recall', fontweight='bold')
    ax.set_ylabel('Precision', fontweight='bold')
    ax.set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'precision_recall_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Metrics comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    metrics = ['precision', 'recall', 'f1']
    x = np.arange(len(results))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        values = [results[m][metric] for m in results.keys()]
        ax.bar(x + i*width, values, width, label=metric.capitalize())
    
    ax.set_xlabel('Methods', fontweight='bold')
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(list(results.keys()), rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("EXPERIMENT 2 COMPLETE")
    print(f"All results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
