
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, T5EncoderModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
from datetime import datetime
import warnings
import argparse
warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class VulnerabilityDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        input_code = str(row['code'])
        label = int(row['is_vulnerable'])

        tokenized = self.tokenizer(
            input_code,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

    def __len__(self):
        return len(self.data)

class CodeT5Classifier(nn.Module):
    def __init__(self, model_name, dropout=0.3, hidden_size=256):
        super().__init__()
        self.encoder = T5EncoderModel.from_pretrained(model_name)
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.encoder.config.d_model, hidden_size)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout * 0.5)
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        encoder_output = self.encoder(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        cls_embedding = encoder_output.last_hidden_state[:, 0, :]
        x = self.dropout1(cls_embedding)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        logits = self.classifier(x)

        return logits

def train_epoch(model, dataloader, criterion, optimizer, scheduler, device, use_amp=False):
    model.train()
    total_loss = 0
    predictions, true_labels = [], []

    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=-1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

        progress_bar.set_postfix({'loss': loss.item()})
    if scheduler is not None:
        scheduler.step()

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, zero_division=0)

    return avg_loss, accuracy, f1

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    predictions, true_labels, probabilities = [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            probabilities.extend(probs[:, 1].cpu().numpy())

    avg_loss = total_loss / len(dataloader)

    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy_score(true_labels, predictions),
        'precision': precision_score(true_labels, predictions, zero_division=0),
        'recall': recall_score(true_labels, predictions, zero_division=0),
        'f1': f1_score(true_labels, predictions, zero_division=0),
        'roc_auc': roc_auc_score(true_labels, probabilities) if len(set(true_labels)) > 1 else 0.0,
        'avg_precision': average_precision_score(true_labels, probabilities) if len(set(true_labels)) > 1 else 0.0
    }

    return metrics, predictions, true_labels, probabilities

def plot_training_history(history, save_path='training_history.png'):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    epochs = range(1, len(history['train_loss']) + 1)
    axes[0, 0].plot(epochs, history['train_loss'], 'o-', label='Train', linewidth=2, markersize=4)
    axes[0, 0].plot(epochs, history['val_loss'], 's-', label='Validation', linewidth=2, markersize=4)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, history['train_acc'], 'o-', label='Train', linewidth=2, markersize=4)
    axes[0, 1].plot(epochs, history['val_acc'], 's-', label='Validation', linewidth=2, markersize=4)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(epochs, history['train_f1'], 'o-', label='Train F1', linewidth=2, markersize=4, color='green')
    axes[0, 2].plot(epochs, history['val_f1'], 's-', label='Val F1', linewidth=2, markersize=4, color='darkgreen')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('F1 Score')
    axes[0, 2].set_title('F1 Score Progression')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs, history['val_precision'], 'o-', label='Precision', linewidth=2, markersize=4)
    axes[1, 0].plot(epochs, history['val_recall'], 's-', label='Recall', linewidth=2, markersize=4)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Precision and Recall')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs, history['val_roc_auc'], 's-', label='ROC-AUC', linewidth=2, markersize=4, color='purple')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('ROC-AUC Score')
    axes[1, 1].set_title('ROC-AUC Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    if 'learning_rate' in history:
        axes[1, 2].plot(epochs, history['learning_rate'], 'o-', linewidth=2, markersize=4, color='red')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Learning Rate')
        axes[1, 2].set_title('Learning Rate Schedule')
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].set_yscale('log')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, save_path='confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Vulnerable', 'Vulnerable'],
                yticklabels=['Not Vulnerable', 'Vulnerable'],
                cbar_kws={'label': 'Count'}, ax=ax)

    for i in range(2):
        for j in range(2):
            ax.text(j + 0.5, i + 0.7, f'({cm_percent[i, j]:.1f}%)', 
                   ha='center', va='center', fontsize=9, color='gray')

    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curve(y_true, y_probs, save_path='roc_curve.png'):
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_precision_recall_curve(y_true, y_probs, save_path='precision_recall_curve.png'):
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    avg_precision = average_precision_score(y_true, y_probs)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2,
             label=f'PR curve (AP = {avg_precision:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_class_distribution(train_df, val_df, test_df, save_path='class_distribution.png'):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, df, title in zip(axes, [train_df, val_df, test_df], ['Train', 'Validation', 'Test']):
        counts = df['is_vulnerable'].value_counts()
        ax.bar(['Not Vulnerable', 'Vulnerable'], 
               [counts.get(0, 0), counts.get(1, 0)],
               color=['#2ecc71', '#e74c3c'])
        ax.set_ylabel('Count')
        ax.set_title(f'{title} Set Distribution')
        ax.grid(True, alpha=0.3, axis='y')
        for i, v in enumerate([counts.get(0, 0), counts.get(1, 0)]):
            ax.text(i, v + max(counts) * 0.02, str(v), ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_per_class_metrics(y_true, y_pred, save_path='per_class_metrics.png'):
    from sklearn.metrics import precision_recall_fscore_support

    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(2)
    width = 0.25

    ax.bar(x - width, precision, width, label='Precision', color='#3498db')
    ax.bar(x, recall, width, label='Recall', color='#e74c3c')
    ax.bar(x + width, f1, width, label='F1 Score', color='#2ecc71')

    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Performance Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(['Not Vulnerable', 'Vulnerable'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.1])

    for i in range(2):
        for j, values in enumerate([precision, recall, f1]):
            ax.text(i + (j - 1) * width, values[i] + 0.02, f'{values[i]:.3f}',
                   ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # argparse use_focal_loss, balance_classes, use_lr_scheduler
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_focal_loss", action="store_true")
    parser.add_argument("--balance_classes", action="store_true")
    parser.add_argument("--use_lr_scheduler", action="store_true")
    args = parser.parse_args()

    CONFIG = {
        'model_name': 'Salesforce/codet5p-220m',
        'data_path': './ds.jsonl',
        'max_length': 512,
        'batch_size': 8,
        'learning_rate': 2e-5,
        'num_epochs': 10,
        'dropout': 0.3,
        'hidden_size': 256,
        'train_split': 0.7,
        'val_split': 0.15,
        'test_split': 0.15,
        'random_seed': 42,

        'use_focal_loss': args.use_focal_loss,
        'focal_alpha': 0.25,
        'focal_gamma': 2.0,
        'use_mixed_precision': True,
        'gradient_checkpointing': True,
        'use_lr_scheduler': args.use_lr_scheduler,
        'balance_classes': args.balance_classes,
    }

    torch.manual_seed(CONFIG['random_seed'])
    np.random.seed(CONFIG['random_seed'])

    if CONFIG['data_path'].endswith('.jsonl') or CONFIG['data_path'].endswith('.json'):
        print("  Detected JSONL format")
        data = []
        with open(CONFIG['data_path'], 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line.strip()))
        df = pd.DataFrame(data)
    else:
        print("  Detected CSV format")
        df = pd.read_csv(CONFIG['data_path'])

    print(f"\nDataset size before filtering: {len(df)} samples")
    print(f"Initial class distribution:\n{df['is_vulnerable'].value_counts()}")

    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    def token_count(code):
        return len(tokenizer(str(code), truncation=False)['input_ids'])

    df['token_count'] = df['code'].apply(token_count)
    filtered_df = df[df['token_count'] <= 512].reset_index(drop=True)

    print(f"Dataset size after filtering: {len(filtered_df)} samples")
    print(f"Class distribution after filtering:\n{filtered_df['is_vulnerable'].value_counts()}")

    if CONFIG['balance_classes']:
        print("\nBalancing classes (undersampling majority class)...")
        vuln_df = filtered_df[filtered_df['is_vulnerable'] == 1]
        non_vuln_df = filtered_df[filtered_df['is_vulnerable'] == 0]

        min_count = min(len(vuln_df), len(non_vuln_df))

        vuln_df_balanced = vuln_df.sample(n=min_count, random_state=CONFIG['random_seed'])
        non_vuln_df_balanced = non_vuln_df.sample(n=min_count, random_state=CONFIG['random_seed'])

        filtered_df = pd.concat([vuln_df_balanced, non_vuln_df_balanced]).sample(frac=1, random_state=CONFIG['random_seed']).reset_index(drop=True)

        print(f"Balanced dataset size: {len(filtered_df)} samples")
        print(f"Balanced class distribution:\n{filtered_df['is_vulnerable'].value_counts()}")

    train_df, temp_df = train_test_split(
        filtered_df,
        test_size=CONFIG['val_split'] + CONFIG['test_split'],
        stratify=filtered_df['is_vulnerable'],
        random_state=CONFIG['random_seed']
    )

    val_size_adjusted = CONFIG['val_split'] / (CONFIG['val_split'] + CONFIG['test_split'])
    val_df, test_df = train_test_split(
        temp_df,
        test_size=1 - val_size_adjusted,
        stratify=temp_df['is_vulnerable'],
        random_state=CONFIG['random_seed']
    )

    print(f"\nTrain: {len(train_df)} samples")
    print(f"  Class 0: {(train_df['is_vulnerable'] == 0).sum()}")
    print(f"  Class 1: {(train_df['is_vulnerable'] == 1).sum()}")
    print(f"Val: {len(val_df)} samples")
    print(f"  Class 0: {(val_df['is_vulnerable'] == 0).sum()}")
    print(f"  Class 1: {(val_df['is_vulnerable'] == 1).sum()}")
    print(f"Test: {len(test_df)} samples")
    print(f"  Class 0: {(test_df['is_vulnerable'] == 0).sum()}")
    print(f"  Class 1: {(test_df['is_vulnerable'] == 1).sum()}")

    plot_class_distribution(train_df, val_df, test_df)
    train_dataset = VulnerabilityDataset(train_df, tokenizer, CONFIG['max_length'])
    val_dataset = VulnerabilityDataset(val_df, tokenizer, CONFIG['max_length'])
    test_dataset = VulnerabilityDataset(test_df, tokenizer, CONFIG['max_length'])

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], 
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], 
                           shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], 
                            shuffle=False, num_workers=4, pin_memory=True)

    model = CodeT5Classifier(CONFIG['model_name'], dropout=CONFIG['dropout'], hidden_size=CONFIG['hidden_size'])

    if CONFIG['gradient_checkpointing']:
        model.encoder.gradient_checkpointing_enable()
        print("✓ Gradient checkpointing enabled")

    model = model.to(DEVICE)

    param_count = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"  Total: {param_count:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Model size (FP16): ~{param_count * 2 / (1024**2):.2f} MB")


    if CONFIG['use_focal_loss']:
        criterion = FocalLoss(alpha=CONFIG['focal_alpha'], gamma=CONFIG['focal_gamma'])
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=0.01)

    scheduler = None
    if CONFIG['use_lr_scheduler']:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=CONFIG['num_epochs'], eta_min=1e-7
        )

    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_precision': [],
        'val_recall': [], 'val_f1': [], 'val_roc_auc': [],
        'learning_rate': []
    }

    best_val_f1 = 0.0
    for epoch in range(CONFIG['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{CONFIG['num_epochs']}")

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning Rate: {current_lr:.2e}")
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, DEVICE,
            use_amp=CONFIG['use_mixed_precision']
        )

        val_metrics, _, _, _ = evaluate(model, val_loader, criterion, DEVICE)

        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_roc_auc'].append(val_metrics['roc_auc'])
        history['learning_rate'].append(current_lr)

        print(f"\nTrain - Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1']:.4f}")
        print(f"Val   - Precision: {val_metrics['precision']:.4f} | Recall: {val_metrics['recall']:.4f} | ROC-AUC: {val_metrics['roc_auc']:.4f}")

        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': best_val_f1,
                'config': CONFIG
            }, 'best_model.pt')

    plot_training_history(history)
    checkpoint = torch.load('best_model.pt', map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_metrics, test_preds, test_labels, test_probs = evaluate(
        model, test_loader, criterion, DEVICE
    )

    print(f"\nTest Results:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1 Score:  {test_metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {test_metrics['roc_auc']:.4f}")
    print(f"  Avg Precision: {test_metrics['avg_precision']:.4f}")

    print(f"\nDetailed Classification Report:")
    print(classification_report(test_labels, test_preds, 
                                target_names=['Not Vulnerable', 'Vulnerable'],
                                digits=4))

    plot_confusion_matrix(test_labels, test_preds)
    plot_roc_curve(test_labels, test_probs)
    plot_precision_recall_curve(test_labels, test_probs)
    plot_per_class_metrics(test_labels, test_preds)

    results = {
        'config': CONFIG,
        'best_val_f1': float(best_val_f1),
        'test_metrics': {k: float(v) for k, v in test_metrics.items()},
        'training_history': {k: [float(x) for x in v] for k, v in history.items()},
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    with open('training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to training_results.json")

    if torch.cuda.is_available():
        print("GPU MEMORY SUMMARY")
        print(f"  Allocated: {torch.cuda.memory_allocated(DEVICE) / 1024**2:.2f} MB")
        print(f"  Reserved:  {torch.cuda.memory_reserved(DEVICE) / 1024**2:.2f} MB")
        print(f"  Max Allocated: {torch.cuda.max_memory_allocated(DEVICE) / 1024**2:.2f} MB")
    print(f"Best Validation F1: {best_val_f1:.4f}")
    print(f"Test F1: {test_metrics['f1']:.4f}")

if __name__ == "__main__":
    main()
