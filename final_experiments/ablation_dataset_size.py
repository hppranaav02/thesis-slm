
"""
Ablation Study: Dataset Size Impact
Tests model performance with different training set sizes
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
    confusion_matrix
)
from imblearn.under_sampling import RandomUnderSampler
from tqdm import tqdm

warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

OUTPUT_DIR = Path("./ablation_dataset_size")
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
    
    # Dataset size fractions to test
    dataset_fractions = [0.1, 0.25, 0.5, 0.75, 1.0]

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

def train_and_evaluate(train_loader, test_loader, device):
    model = VulnerabilityClassifier(config.model_name).to(device)
    criterion = AsymmetricFocalLoss()
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    total_steps = len(train_loader) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=total_steps)
    
    # Training
    for epoch in range(1, config.num_epochs + 1):
        model.train()
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
    
    # Evaluation
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in test_loader:
            if batch is None:
                continue
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs['logits'], dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_labels), np.array(all_preds)

def main():
    print("ABLATION STUDY: DATASET SIZE IMPACT")
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    data_loader = JSONLDataLoader(config)
    
    raw_data = data_loader.load_jsonl(config.dataset_path)
    df = data_loader.preprocess_data(raw_data)
    filtered_df = data_loader.filter_large_entries(df, tokenizer)
    
    # Full train/test split
    train_df_full, val_df, test_df = data_loader.stratified_split(filtered_df)
    
    results = {}
    
    for fraction in config.dataset_fractions:
        print(f"Training with {fraction*100:.0f}% of data")
        
        # Sample training data
        if fraction < 1.0:
            train_df = train_df_full.sample(frac=fraction, random_state=42)
        else:
            train_df = train_df_full
        
        print(f"Training samples: {len(train_df)}")
        
        # Create datasets
        train_dataset = VulnerabilityDataset(train_df, tokenizer, config.window_size)
        test_dataset = VulnerabilityDataset(test_df, tokenizer, config.window_size)
        
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
        
        # Train and evaluate
        labels, preds = train_and_evaluate(train_loader, test_loader, DEVICE)
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
        accuracy = accuracy_score(labels, preds)
        cm = confusion_matrix(labels, preds)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
        
        results[fraction] = {
            'train_size': len(train_df),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
        }
        
        print(f"Results: P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}")
    
    # Save results
    results_data = []
    for fraction, result in results.items():
        results_data.append({
            'Dataset_Fraction': f"{fraction*100:.0f}%",
            'Train_Size': result['train_size'],
            'Accuracy': f"{result['accuracy']:.4f}",
            'Precision': f"{result['precision']:.4f}",
            'Recall': f"{result['recall']:.4f}",
            'F1-Score': f"{result['f1']:.4f}",
            'TP': result['tp'], 'TN': result['tn'],
            'FP': result['fp'], 'FN': result['fn']
        })
    
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(OUTPUT_DIR / 'dataset_size_ablation.csv', index=False)
    print(f"\nResults saved to {OUTPUT_DIR / 'dataset_size_ablation.csv'}")
    
    # Visualizations
    # 1. Performance vs Dataset Size
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    fractions = list(results.keys())
    train_sizes = [results[f]['train_size'] for f in fractions]
    
    metrics = ['precision', 'recall', 'f1', 'accuracy']
    titles = ['Precision', 'Recall', 'F1-Score', 'Accuracy']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        values = [results[f][metric] for f in fractions]
        
        ax.plot(train_sizes, values, marker='o', linewidth=2, markersize=8)
        ax.set_xlabel('Training Set Size', fontweight='bold')
        ax.set_ylabel(title, fontweight='bold')
        ax.set_title(f'{title} vs Training Set Size', fontweight='bold')
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'metrics_vs_dataset_size.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Combined plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for metric, label in zip(['precision', 'recall', 'f1'], ['Precision', 'Recall', 'F1-Score']):
        values = [results[f][metric] for f in fractions]
        ax.plot(train_sizes, values, marker='o', linewidth=2, markersize=8, label=label)
    
    ax.set_xlabel('Training Set Size', fontweight='bold')
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Performance Metrics vs Training Set Size', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'combined_metrics_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. TP/TN/FP/FN breakdown
    fig, ax = plt.subplots(figsize=(12, 6))
    
    tp_vals = [results[f]['tp'] for f in fractions]
    tn_vals = [results[f]['tn'] for f in fractions]
    fp_vals = [results[f]['fp'] for f in fractions]
    fn_vals = [results[f]['fn'] for f in fractions]
    
    x = np.arange(len(fractions))
    width = 0.2
    
    ax.bar(x - 1.5*width, tp_vals, width, label='True Positive', alpha=0.8, color='green')
    ax.bar(x - 0.5*width, tn_vals, width, label='True Negative', alpha=0.8, color='blue')
    ax.bar(x + 0.5*width, fp_vals, width, label='False Positive', alpha=0.8, color='red')
    ax.bar(x + 1.5*width, fn_vals, width, label='False Negative', alpha=0.8, color='orange')
    
    ax.set_xlabel('Training Set Fraction', fontweight='bold')
    ax.set_ylabel('Count', fontweight='bold')
    ax.set_title('Confusion Matrix Components vs Training Size', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{f*100:.0f}%" for f in fractions])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'confusion_breakdown.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("DATASET SIZE ABLATION COMPLETE")
    print(f"All results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
