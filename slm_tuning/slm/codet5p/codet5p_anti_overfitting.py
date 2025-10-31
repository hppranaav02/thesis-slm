
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
import copy
warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

class FocalLossWithLabelSmoothing(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, smoothing=0.1, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, inputs, targets):
        n_classes = inputs.size(-1)
        with torch.no_grad():
            smooth_targets = torch.zeros_like(inputs)
            smooth_targets.fill_(self.smoothing / (n_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        log_probs = F.log_softmax(inputs, dim=-1)
        loss = -(smooth_targets * log_probs).sum(dim=-1)
        probs = torch.exp(log_probs)
        focal_weight = (1 - probs.gather(1, targets.unsqueeze(1)).squeeze(1)) ** self.gamma
        loss = self.alpha * focal_weight * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.0001, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, score, epoch):
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        if self.mode == 'min':
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.best_epoch = epoch
                self.counter = 0
            else:
                self.counter += 1
        else:
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.best_epoch = epoch
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            return True
        return False

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
        self.layer_norm = nn.LayerNorm(hidden_size)  # Added layer norm
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
        x = self.layer_norm(x)
        x = self.relu(x)
        x = self.dropout2(x)
        logits = self.classifier(x)

        return logits

    def freeze_encoder_layers(self, num_layers_to_freeze):
        if num_layers_to_freeze > 0:
            for param in self.encoder.encoder.embed_tokens.parameters():
                param.requires_grad = False

            for i in range(num_layers_to_freeze):
                if i < len(self.encoder.encoder.block):
                    for param in self.encoder.encoder.block[i].parameters():
                        param.requires_grad = False

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, initial_lr, min_lr=1e-7):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1

        if self.current_epoch <= self.warmup_epochs:
            lr = self.initial_lr * (self.current_epoch / self.warmup_epochs)
        else:
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.initial_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr

class SWA:
    def __init__(self, model):
        self.swa_model = copy.deepcopy(model)
        self.swa_n = 0

    def update(self, model):
        self.swa_n += 1
        for swa_param, param in zip(self.swa_model.parameters(), model.parameters()):
            swa_param.data = (swa_param.data * (self.swa_n - 1) + param.data) / self.swa_n

    def get_model(self):
        return self.swa_model

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
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=-1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

        progress_bar.set_postfix({'loss': loss.item()})

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

def plot_training_history_enhanced(history, early_stop_epoch=None, save_path='training_history.png'):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    epochs = range(1, len(history['train_loss']) + 1)

    axes[0, 0].plot(epochs, history['train_loss'], 'o-', label='Train', linewidth=2, markersize=4)
    axes[0, 0].plot(epochs, history['val_loss'], 's-', label='Validation', linewidth=2, markersize=4)
    if early_stop_epoch:
        axes[0, 0].axvline(x=early_stop_epoch, color='red', linestyle='--', label='Early Stop')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 1].plot(epochs, history['train_acc'], 'o-', label='Train', linewidth=2, markersize=4)
    axes[0, 1].plot(epochs, history['val_acc'], 's-', label='Validation', linewidth=2, markersize=4)
    if early_stop_epoch:
        axes[0, 1].axvline(x=early_stop_epoch, color='red', linestyle='--', label='Early Stop')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 2].plot(epochs, history['train_f1'], 'o-', label='Train F1', linewidth=2, markersize=4, color='green')
    axes[0, 2].plot(epochs, history['val_f1'], 's-', label='Val F1', linewidth=2, markersize=4, color='darkgreen')
    if early_stop_epoch:
        axes[0, 2].axvline(x=early_stop_epoch, color='red', linestyle='--', label='Early Stop')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('F1 Score')
    axes[0, 2].set_title('F1 Score Progression')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    axes[1, 0].plot(epochs, history['val_precision'], 'o-', label='Precision', linewidth=2, markersize=4)
    axes[1, 0].plot(epochs, history['val_recall'], 's-', label='Recall', linewidth=2, markersize=4)
    if early_stop_epoch:
        axes[1, 0].axvline(x=early_stop_epoch, color='red', linestyle='--')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Precision and Recall')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    train_val_gap = np.array(history['train_loss']) - np.array(history['val_loss'])
    axes[1, 1].plot(epochs, train_val_gap, 'o-', linewidth=2, markersize=4, color='red')
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    if early_stop_epoch:
        axes[1, 1].axvline(x=early_stop_epoch, color='red', linestyle='--', label='Early Stop')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Train Loss - Val Loss')
    axes[1, 1].set_title('Overfitting Indicator (Lower is Better)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    if 'learning_rate' in history:
        axes[1, 2].plot(epochs, history['learning_rate'], 'o-', linewidth=2, markersize=4, color='purple')
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

def main():
    CONFIG = {
        'model_name': 'Salesforce/codet5p-220m',
        'data_path': './ds.jsonl',
        'max_length': 512,
        'batch_size': 8,
        'learning_rate': 1e-5,
        'num_epochs': 30,
        'dropout': 0.3,
        'hidden_size': 256,
        'train_split': 0.7,
        'val_split': 0.15,
        'test_split': 0.15,
        'random_seed': 42,
        'use_focal_loss': True,
        'focal_alpha': 0.25,
        'focal_gamma': 4.0,
        'label_smoothing': 0.1,

        'early_stopping': True,
        'early_stopping_patience': 5,
        'early_stopping_min_delta': 0.0001,

        'lr_warmup_epochs': 2,
        'use_lr_warmup': True, 

        'weight_decay': 0.01,
        'gradient_clip': 1.0,

        'freeze_bottom_layers': 3,

        'use_swa': True,
        'swa_start_epoch': 15,

        'balance_classes': True,
        'use_mixed_precision': True,
        'gradient_checkpointing': True,
    }

    torch.manual_seed(CONFIG['random_seed'])
    np.random.seed(CONFIG['random_seed'])

    if CONFIG['data_path'].endswith('.jsonl') or CONFIG['data_path'].endswith('.json'):
        data = []
        with open(CONFIG['data_path'], 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line.strip()))
        df = pd.DataFrame(data)
    else:
        df = pd.read_csv(CONFIG['data_path'])

    print(f"Dataset size: {len(df)} samples")

    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])

    def token_count(code):
        return len(tokenizer(str(code), truncation=False)['input_ids'])

    df['token_count'] = df['code'].apply(token_count)
    filtered_df = df[df['token_count'] <= 512].reset_index(drop=True)

    print(f"After filtering: {len(filtered_df)} samples")

    if CONFIG['balance_classes']:
        vuln_df = filtered_df[filtered_df['is_vulnerable'] == 1]
        non_vuln_df = filtered_df[filtered_df['is_vulnerable'] == 0]
        min_count = min(len(vuln_df), len(non_vuln_df))
        filtered_df = pd.concat([
            vuln_df.sample(n=min_count, random_state=CONFIG['random_seed']),
            non_vuln_df.sample(n=min_count, random_state=CONFIG['random_seed'])
        ]).sample(frac=1, random_state=CONFIG['random_seed']).reset_index(drop=True)
        print(f"Balanced dataset: {len(filtered_df)} samples")
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

    print(f"\nTrain: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    train_dataset = VulnerabilityDataset(train_df, tokenizer, CONFIG['max_length'])
    val_dataset = VulnerabilityDataset(val_df, tokenizer, CONFIG['max_length'])
    test_dataset = VulnerabilityDataset(test_df, tokenizer, CONFIG['max_length'])

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], 
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], 
                           shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], 
                            shuffle=False, num_workers=4, pin_memory=True)

    model = CodeT5Classifier(CONFIG['model_name'], 
                            dropout=CONFIG['dropout'], 
                            hidden_size=CONFIG['hidden_size'])

    if CONFIG['freeze_bottom_layers'] > 0:
        model.freeze_encoder_layers(CONFIG['freeze_bottom_layers'])

    if CONFIG['gradient_checkpointing']:
        model.encoder.gradient_checkpointing_enable()

    model = model.to(DEVICE)
    if CONFIG['use_focal_loss']:
        criterion = FocalLossWithLabelSmoothing(
            alpha=CONFIG['focal_alpha'], 
            gamma=CONFIG['focal_gamma'],
            smoothing=CONFIG['label_smoothing']
        )
        # print(f"✓ Focal Loss + Label Smoothing (ε={CONFIG['label_smoothing']})")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG['label_smoothing'])
        # print(f"✓ Cross Entropy + Label Smoothing (ε={CONFIG['label_smoothing']})")

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=CONFIG['learning_rate'], 
        weight_decay=CONFIG['weight_decay']
    )
    # print(f"✓ AdamW optimizer (weight_decay={CONFIG['weight_decay']})")

    if CONFIG['use_lr_warmup']:
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_epochs=CONFIG['lr_warmup_epochs'],
            total_epochs=CONFIG['num_epochs'],
            initial_lr=CONFIG['learning_rate']
        )
        # print(f"✓ Warmup + Cosine Annealing ({CONFIG['lr_warmup_epochs']} warmup epochs)")
    else:
        scheduler = None

    if CONFIG['early_stopping']:
        early_stopping = EarlyStopping(
            patience=CONFIG['early_stopping_patience'],
            min_delta=CONFIG['early_stopping_min_delta'],
            mode='max'
        )
        print(f"Early Stopping (patience={CONFIG['early_stopping_patience']})")

    if CONFIG['use_swa']:
        swa = SWA(model)
        print(f"Stochastic Weight Averaging (start epoch {CONFIG['swa_start_epoch']})")

    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_precision': [],
        'val_recall': [], 'val_f1': [], 'val_roc_auc': [],
        'learning_rate': []
    }

    best_val_f1 = 0.0
    early_stopped_epoch = None

    for epoch in range(CONFIG['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{CONFIG['num_epochs']}")

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning Rate: {current_lr:.2e}")
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, DEVICE,
            use_amp=CONFIG['use_mixed_precision']
        )
        if scheduler and CONFIG['use_lr_warmup']:
            current_lr = scheduler.step()
        val_metrics, _, _, _ = evaluate(model, val_loader, criterion, DEVICE)
        if CONFIG['use_swa'] and epoch >= CONFIG['swa_start_epoch']:
            swa.update(model)

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

        # Save best model
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': best_val_f1,
                'config': CONFIG
            }, 'best_model.pt')
            print(f"Best model saved! (F1: {best_val_f1:.4f})")

        if CONFIG['early_stopping']:
            if early_stopping(val_metrics['f1'], epoch):
                print(f"\n Early stopping triggered at epoch {epoch + 1}")
                print(f"   Best epoch was {early_stopping.best_epoch + 1} with F1={early_stopping.best_score:.4f}")
                early_stopped_epoch = early_stopping.best_epoch + 1
                break
    plot_training_history_enhanced(history, early_stopped_epoch)

    checkpoint = torch.load('best_model.pt', map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    if CONFIG['use_swa'] and epoch >= CONFIG['swa_start_epoch']:
        print("\nUsing SWA model for final evaluation...")
        model = swa.get_model().to(DEVICE)

    test_metrics, test_preds, test_labels, test_probs = evaluate(
        model, test_loader, criterion, DEVICE
    )

    print(f"\nTest Results:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1 Score:  {test_metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {test_metrics['roc_auc']:.4f}")

    print(f"\nClassification Report:")
    print(classification_report(test_labels, test_preds, 
                                target_names=['Not Vulnerable', 'Vulnerable'],
                                digits=4))

    plot_confusion_matrix(test_labels, test_preds)
    plot_roc_curve(test_labels, test_probs)
    plot_precision_recall_curve(test_labels, test_probs)

    results = {
        'config': CONFIG,
        'best_val_f1': float(best_val_f1),
        'test_metrics': {k: float(v) for k, v in test_metrics.items()},
        'training_history': {k: [float(x) for x in v] for k, v in history.items()},
        'early_stopped_epoch': early_stopped_epoch,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    with open('training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Best Validation F1: {best_val_f1:.4f}")
    print(f"Test F1: {test_metrics['f1']:.4f}")
    if early_stopped_epoch:
        print(f"Training stopped early at epoch {early_stopped_epoch}")

if __name__ == "__main__":
    main()
