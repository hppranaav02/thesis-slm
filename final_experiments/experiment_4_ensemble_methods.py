
"""
Experiment 4: Ensemble Methods
Compares different ensemble strategies using best models from previous experiments
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
    AutoTokenizer, AutoModel, RobertaModel, T5EncoderModel,
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

OUTPUT_DIR = Path("./experiment_4_ensemble_methods")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

class Config:
    dataset_path = "./ds.jsonl"
    models = [
        "microsoft/codebert-base",
        "microsoft/graphcodebert-base",
        "Salesforce/codet5-base"
    ]
    max_token_length = 512
    window_size = 256
    test_size = 0.2
    val_size = 0.1
    batch_size = 8
    learning_rate = 2e-5
    num_epochs = 10
    weight_decay = 0.01

config = Config()

# Asymmetric Focal Loss (best from Exp 2)
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

def train_model(train_loader, test_loader, model_name, device):
    model = VulnerabilityClassifier(model_name).to(device)
    criterion = AsymmetricFocalLoss(alpha=0.25, gamma_pos=2.0, gamma_neg=4.0)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    total_steps = len(train_loader) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=total_steps)
    
    # Training
    for epoch in range(1, config.num_epochs + 1):
        model.train()
        for batch in tqdm(train_loader, desc=f"Training {model_name.split('/')[-1]} - Epoch {epoch}", leave=False):
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
    print("="*80)
    print("EXPERIMENT 4: ENSEMBLE METHODS")
    print("="*80)
    
    data_loader = JSONLDataLoader(config)
    raw_data = data_loader.load_jsonl(config.dataset_path)
    df = data_loader.preprocess_data(raw_data)
    
    # Train individual models
    individual_results = {}
    model_predictions = {}
    model_probabilities = {}
    test_labels = None
    
    for model_name in config.models:
        print(f"\n{'='*80}")
        print(f"Training {model_name}")
        print(f"{'='*80}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        filtered_df = data_loader.filter_large_entries(df, tokenizer)
        train_df, val_df, test_df = data_loader.stratified_split(filtered_df)
        
        train_dataset = VulnerabilityDataset(train_df, tokenizer, config.window_size)
        test_dataset = VulnerabilityDataset(test_df, tokenizer, config.window_size)
        
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
        
        labels, preds, probs = train_model(train_loader, test_loader, model_name, DEVICE)
        
        if test_labels is None:
            test_labels = labels
        
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
        cm = confusion_matrix(labels, preds)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
        
        individual_results[model_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
        }
        
        model_predictions[model_name] = preds
        model_probabilities[model_name] = probs
        
        print(f"Results: P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}")
    
    # Ensemble strategies
    ensemble_results = {}
    
    # 1. Weighted Voting (based on F1 scores)
    weights = {m: individual_results[m]['f1'] for m in config.models}
    total_weight = sum(weights.values())
    weights = {m: w/total_weight for m, w in weights.items()}
    
    weighted_probs = np.zeros_like(model_probabilities[config.models[0]])
    for model_name, probs in model_probabilities.items():
        weighted_probs += weights[model_name] * probs
    
    weighted_preds = (weighted_probs[:, 1] > 0.6).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(test_labels, weighted_preds, average='weighted')
    cm = confusion_matrix(test_labels, weighted_preds)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    
    ensemble_results['Weighted Voting'] = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'preds': weighted_preds
    }
    
    # 2. Majority Voting (at least 2/3 agree)
    votes = np.stack([model_predictions[m] for m in config.models], axis=0)
    majority_preds = (np.sum(votes, axis=0) >= 2).astype(int)
    
    precision, recall, f1, _ = precision_recall_fscore_support(test_labels, majority_preds, average='weighted')
    cm = confusion_matrix(test_labels, majority_preds)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    
    ensemble_results['Majority Voting'] = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'preds': majority_preds
    }
    
    # 3. Average Probabilities
    avg_probs = np.mean([model_probabilities[m] for m in config.models], axis=0)
    avg_preds = (avg_probs[:, 1] > 0.5).astype(int)
    
    precision, recall, f1, _ = precision_recall_fscore_support(test_labels, avg_preds, average='weighted')
    cm = confusion_matrix(test_labels, avg_preds)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    
    ensemble_results['Average Probs'] = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'preds': avg_preds
    }
    
    # Save results
    results_data = []
    for model_name, result in individual_results.items():
        results_data.append({
            'Method': model_name.split('/')[-1],
            'Type': 'Individual',
            'Precision': f"{result['precision']:.4f}",
            'Recall': f"{result['recall']:.4f}",
            'F1-Score': f"{result['f1']:.4f}",
            'TP': result['tp'], 'TN': result['tn'],
            'FP': result['fp'], 'FN': result['fn']
        })
    
    for ens_name, result in ensemble_results.items():
        results_data.append({
            'Method': ens_name,
            'Type': 'Ensemble',
            'Precision': f"{result['precision']:.4f}",
            'Recall': f"{result['recall']:.4f}",
            'F1-Score': f"{result['f1']:.4f}",
            'TP': result['tp'], 'TN': result['tn'],
            'FP': result['fp'], 'FN': result['fn']
        })
    
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(OUTPUT_DIR / 'ensemble_results.csv', index=False)
    print(f"\nResults saved to {OUTPUT_DIR / 'ensemble_results.csv'}")
    
    # Visualizations
    # 1. Confusion matrices
    n_methods = len(individual_results) + len(ensemble_results)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    idx = 0
    for model_name, result in individual_results.items():
        cm = confusion_matrix(test_labels, model_predictions[model_name])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   xticklabels=['Non-Vuln', 'Vuln'],
                   yticklabels=['Non-Vuln', 'Vuln'])
        axes[idx].set_title(f"{model_name.split('/')[-1]}\nF1: {result['f1']:.3f}", fontweight='bold')
        idx += 1
    
    for ens_name, result in ensemble_results.items():
        cm = confusion_matrix(test_labels, result['preds'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=axes[idx],
                   xticklabels=['Non-Vuln', 'Vuln'],
                   yticklabels=['Non-Vuln', 'Vuln'])
        axes[idx].set_title(f"{ens_name}\nF1: {result['f1']:.3f}", fontweight='bold')
        idx += 1
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'confusion_matrices_all.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Performance comparison
    fig, ax = plt.subplots(figsize=(14, 6))
    
    methods = list(individual_results.keys()) + list(ensemble_results.keys())
    method_labels = [m.split('/')[-1] if '/' in m else m for m in methods]
    
    all_results = {**individual_results, **ensemble_results}
    precisions = [all_results[m]['precision'] for m in methods]
    recalls = [all_results[m]['recall'] for m in methods]
    f1s = [all_results[m]['f1'] for m in methods]
    
    x = np.arange(len(methods))
    width = 0.25
    
    ax.bar(x - width, precisions, width, label='Precision', alpha=0.8)
    ax.bar(x, recalls, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1s, width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Methods', fontweight='bold')
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Individual Models vs Ensemble Methods', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(method_labels, rotation=45, ha='right')
    ax.axvline(x=len(individual_results)-0.5, color='red', linestyle='--', linewidth=2, label='Ensemble Start')
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'ensemble_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Model agreement heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    model_names = [m.split('/')[-1] for m in config.models]
    agreement_matrix = np.zeros((len(config.models), len(config.models)))
    
    for i, model1 in enumerate(config.models):
        for j, model2 in enumerate(config.models):
            if i == j:
                agreement_matrix[i, j] = 1.0
            else:
                preds1 = model_predictions[model1]
                preds2 = model_predictions[model2]
                agreement = np.mean(preds1 == preds2)
                agreement_matrix[i, j] = agreement
    
    sns.heatmap(agreement_matrix, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax,
                xticklabels=model_names, yticklabels=model_names, vmin=0, vmax=1)
    ax.set_title('Model Agreement Matrix (Prediction Overlap)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'model_agreement.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n{'='*80}")
    print("EXPERIMENT 4 COMPLETE")
    print(f"All results saved to {OUTPUT_DIR}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
