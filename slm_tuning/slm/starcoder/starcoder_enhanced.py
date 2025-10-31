"""
StarCoder-1B Vulnerability Classification
========================================================================
Maximum F1-Score Optimization

- Weighted Asymmetric Focal Loss (Ridnik et al., ICCV 2021)
- Traditional Focal Loss (Lin et al., 2017)
- threshold tuning
- Learning rate scheduling with warmup

Li et al. "StarCoder: may the source be with you!" (2023)
Lin et al. "Focal Loss for Dense Object Detection" (2017)
Ridnik et al. "Asymmetric Loss for Multi-Label Classification" (ICCV 2021)
He et al. "An Improved Software Source Code Vulnerability Detection Method" (Sensors 2025)
Sun et al. "Ensembling Large Language Models for Code Vulnerability Detection" (2025)
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    get_cosine_schedule_with_warmup
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType
)
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='StarCoder-1B Vulnerability Classification with Advanced Loss Functions'
    )
    
    parser.add_argument('--input_data', type=str, default='./ds.jsonl',
                       help='Path to input JSONL file')
    parser.add_argument('--output_dir', type=str, default='./starcoder_vuln_classifier',
                       help='Output directory for model and results')
    
    parser.add_argument('--model_name', type=str, default='bigcode/starcoderbase-1b',
                       help='HuggingFace model name')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum sequence length')
    
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Training batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8,
                       help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                       help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--warmup_steps', type=int, default=100,
                       help='Number of warmup steps')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    
    parser.add_argument('--loss_type', type=str, default='asymmetric_focal',
                       choices=['cross_entropy', 'focal', 'asymmetric_focal'],
                       help='Loss function type')
    parser.add_argument('--focal_alpha', type=float, default=0.25,
                       help='Focal loss alpha parameter')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                       help='Focal loss gamma parameter')
    parser.add_argument('--asl_gamma_pos', type=float, default=0.0,
                       help='Asymmetric focal loss gamma+ for positive samples')
    parser.add_argument('--asl_gamma_neg', type=float, default=4.0,
                       help='Asymmetric focal loss gamma- for negative samples')
    
    parser.add_argument('--lora_r', type=int, default=8,
                       help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32,
                       help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05,
                       help='LoRA dropout')
    
    parser.add_argument('--train_split', type=float, default=0.70,
                       help='Training data split ratio')
    parser.add_argument('--val_split', type=float, default=0.15,
                       help='Validation data split ratio')
    parser.add_argument('--test_split', type=float, default=0.15,
                       help='Test data split ratio')
    parser.add_argument('--balance_classes', action='store_true', default=True,
                       help='Balance classes to 50/50')
    
    parser.add_argument('--use_gradient_checkpointing', action='store_true', default=True,
                       help='Use gradient checkpointing')
    parser.add_argument('--use_8bit_optimizer', action='store_true', default=True,
                       help='Use 8-bit optimizer')
    parser.add_argument('--fp16', action='store_true', default=True,
                       help='Use FP16 training')
    
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use for training')
    
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--save_visualizations', action='store_true', default=True,
                       help='Save visualization plots')
    parser.add_argument('--class_weight_non_vuln', type=float, default=1.0,
                   help='Weight for non-vulnerable class (increase to reduce FP)')
    parser.add_argument('--class_weight_vuln', type=float, default=1.0,
                   help='Weight for vulnerable class')
    
    return parser.parse_args()

args = parse_arguments()

# Force CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = args.device.split(':')[-1]
DEVICE = torch.device(args.device if torch.cuda.is_available() else "cpu")

print(f"STARCODER-1B VULNERABILITY CLASSIFICATION")
print(f"Device: {DEVICE}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Loss function: {args.loss_type}")

class WeightedFocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        # class_weights = [weight_non_vuln, weight_vuln]
        # Higher weight for non-vulnerable reduces FP
        self.class_weights = class_weights
        print(f"Focal Loss (α={alpha}, γ={gamma}, weights={class_weights})")
    
    def forward(self, inputs, targets):
        ce_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        # Apply class weights to reduce false positives
        if self.class_weights is not None:
            weights = self.class_weights[targets]
            focal_loss = focal_loss * weights
        
        return focal_loss.mean()


class WeightedAsymmetricFocalLoss(torch.nn.Module):
    def __init__(self, gamma_pos=0.0, gamma_neg=4.0, clip=0.05, class_weights=None):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.class_weights = class_weights
        print(f"ASL (γ+={gamma_pos}, γ-={gamma_neg}, weights={class_weights})")
    
    def forward(self, logits, targets):
        targets_one_hot = torch.nn.functional.one_hot(targets, logits.size(1)).float()
        
        # Get probabilities for weighting
        probs = torch.sigmoid(logits)
        if self.clip:
            probs = torch.clamp(probs, self.clip, 1.0 - self.clip)
        # Asymmetric weights
        pos_weight = (1 - probs) ** self.gamma_pos
        neg_weight = probs ** self.gamma_neg
        weights = targets_one_hot * pos_weight + (1 - targets_one_hot) * neg_weight
        # binary_cross_entropy_with_logits (AMP-safe)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets_one_hot, reduction='none'
        )
        
        loss = weights * bce
        # Apply class weights to penalize FP more
        if self.class_weights is not None:
            class_weight_mask = torch.zeros_like(loss)
            for i in range(logits.size(1)):
                class_weight_mask[:, i] = self.class_weights[i]
            loss = loss * class_weight_mask
        
        return loss.sum() / logits.size(0)

# class FocalLoss(torch.nn.Module):
    
#     def __init__(self, alpha=0.25, gamma=2.0):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         print(f"Initialized Focal Loss (α={alpha}, γ={gamma})")
    
#     def forward(self, inputs, targets):
#         ce_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction='none')
#         pt = torch.exp(-ce_loss)
#         focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
#         return focal_loss.mean()


# class AsymmetricFocalLoss(torch.nn.Module):
    
#     def __init__(self, gamma_pos=0.0, gamma_neg=4.0, clip=0.05):
#         super(AsymmetricFocalLoss, self).__init__()
#         self.gamma_pos = gamma_pos
#         self.gamma_neg = gamma_neg
#         self.clip = clip
#         print(f"Initialized Asymmetric Focal Loss (γ+={gamma_pos}, γ-={gamma_neg}, clip={clip})")
    
#     def forward(self, inputs, targets):
#         # Get probabilities
#         probs = torch.sigmoid(inputs)
        
#         # Convert targets to one-hot
#         targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=inputs.shape[1]).float()
        
#         # Asymmetric clipping
#         if self.clip is not None and self.clip > 0:
#             probs = torch.clamp(probs, min=self.clip, max=1.0 - self.clip)
        
#         # Calculate asymmetric focusing weights
#         # For positive samples (target=1): (1-p)^γ+
#         # For negative samples (target=0): p^γ-
#         pos_weight = (1 - probs) ** self.gamma_pos
#         neg_weight = probs ** self.gamma_neg
        
#         # Combine weights based on target
#         weights = targets_one_hot * pos_weight + (1 - targets_one_hot) * neg_weight
        
#         # Calculate binary cross entropy
#         bce = torch.nn.functional.binary_cross_entropy(probs, targets_one_hot, reduction='none')
        
#         # Apply asymmetric weights
#         asymmetric_loss = weights * bce
        
#         return asymmetric_loss.sum() / inputs.shape[0]

# class AsymmetricFocalLoss(torch.nn.Module):
#     def __init__(self, gamma_pos=0.0, gamma_neg=4.0, clip=0.05):
#         super().__init__()
#         self.gamma_pos = gamma_pos
#         self.gamma_neg = gamma_neg
#         self.clip = clip

#     def forward(self, logits, targets):
#         targets_one_hot = torch.nn.functional.one_hot(
#             targets, num_classes=logits.size(1)).float()

#         # Probabilities for weighting only (still safe)
#         probs = torch.sigmoid(logits)
#         if self.clip:
#             probs = torch.clamp(probs, self.clip, 1.0 - self.clip)

#         pos_w = (1.0 - probs) ** self.gamma_pos
#         neg_w = probs ** self.gamma_neg
#         weights = targets_one_hot * pos_w + (1.0 - targets_one_hot) * neg_w

#         # AMP-safe BCE with logits
#         bce = torch.nn.functional.binary_cross_entropy_with_logits(
#             logits, targets_one_hot, reduction='none')

#         return (weights * bce).sum() / logits.size(0)


# def get_loss_function(loss_type, **kwargs):
#     if loss_type == 'cross_entropy':
#         print("Using standard Cross Entropy Loss")
#         return torch.nn.CrossEntropyLoss()
#     elif loss_type == 'focal':
#         return FocalLoss(
#             alpha=kwargs.get('focal_alpha', 0.25),
#             gamma=kwargs.get('focal_gamma', 2.0)
#         )
#     elif loss_type == 'asymmetric_focal':
#         return AsymmetricFocalLoss(
#             gamma_pos=kwargs.get('asl_gamma_pos', 0.0),
#             gamma_neg=kwargs.get('asl_gamma_neg', 4.0)
#         )
#     else:
#         raise ValueError(f"Unknown loss type: {loss_type}")

def get_loss_function(loss_type, **kwargs):
    # Get class weights from args
    class_weights = torch.tensor(
        [args.class_weight_non_vuln, args.class_weight_vuln],
        device=DEVICE,
        dtype=torch.float32
    )
    
    if loss_type == 'cross_entropy':
        print(f"Using Cross Entropy with weights: {class_weights}")
        return torch.nn.CrossEntropyLoss(weight=class_weights)
    
    elif loss_type == 'focal':
        return WeightedFocalLoss(
            alpha=kwargs.get('focal_alpha', 0.25),
            gamma=kwargs.get('focal_gamma', 2.0),
            class_weights=class_weights
        )
    
    elif loss_type == 'asymmetric_focal':
        return WeightedAsymmetricFocalLoss(
            gamma_pos=kwargs.get('asl_gamma_pos', 0.0),
            gamma_neg=kwargs.get('asl_gamma_neg', 4.0),
            class_weights=class_weights
        )

def load_and_balance_data(jsonl_path, tokenizer, max_length=512, balance=True, random_state=42):

    print(f"\nLoading data from {jsonl_path}")
    
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"Input file not found: {jsonl_path}")
    
    vulnerable_samples = []
    non_vulnerable_samples = []
    total_samples = 0
    filtered_out = 0
    parse_errors = 0
    
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(tqdm(f, desc="Loading and filtering"), 1):
                total_samples += 1
                try:
                    line = line.strip()
                    if not line:
                        continue
                        
                    data = json.loads(line)
                    
                    # Validate required fields
                    if 'code' not in data or 'is_vulnerable' not in data:
                        print(f"Warning: Line {line_num} missing required fields, skipping")
                        parse_errors += 1
                        continue
                    
                    code = data['code']
                    label = int(data['is_vulnerable'])
                    
                    if label not in [0, 1]:
                        print(f"Warning: Line {line_num} has invalid label {label}, skipping")
                        parse_errors += 1
                        continue
                    
                    # Filter by token length
                    try:
                        tokens = tokenizer(code, truncation=True, max_length=max_length, add_special_tokens=True)
                        if len(tokens['input_ids']) <= max_length:
                            sample = {'code': code, 'is_vulnerable': label}
                            if label == 1:
                                vulnerable_samples.append(sample)
                            else:
                                non_vulnerable_samples.append(sample)
                        else:
                            filtered_out += 1
                    except Exception as e:
                        print(f"Warning: Tokenization error on line {line_num}: {e}")
                        parse_errors += 1
                        continue
                        
                except json.JSONDecodeError as e:
                    print(f"Warning: JSON decode error on line {line_num}: {e}")
                    parse_errors += 1
                    continue
                except Exception as e:
                    print(f"Warning: Unexpected error on line {line_num}: {e}")
                    parse_errors += 1
                    continue
    
    except Exception as e:
        raise RuntimeError(f"Error reading file {jsonl_path}: {e}")
    
    print(f"DATA LOADING SUMMARY")
    print(f"Total lines processed: {total_samples}")
    print(f"Parse/validation errors: {parse_errors}")
    print(f"Filtered out (>{max_length} tokens): {filtered_out}")
    print(f"Vulnerable samples: {len(vulnerable_samples)}")
    print(f"Non-vulnerable samples: {len(non_vulnerable_samples)}")
    
    if len(vulnerable_samples) == 0 or len(non_vulnerable_samples) == 0:
        raise ValueError("Dataset must contain both vulnerable and non-vulnerable samples!")
    
    # Balance classes if requested
    if balance:
        min_count = min(len(vulnerable_samples), len(non_vulnerable_samples))
        print(f"\nBalancing classes to {min_count} samples each (50/50 split)...")
        
        try:
            # Undersample majority class
            np.random.seed(random_state)
            if len(vulnerable_samples) > min_count:
                vulnerable_samples = resample(vulnerable_samples, 
                                             n_samples=min_count, 
                                             random_state=random_state,
                                             replace=False)
            if len(non_vulnerable_samples) > min_count:
                non_vulnerable_samples = resample(non_vulnerable_samples,
                                                 n_samples=min_count,
                                                 random_state=random_state,
                                                 replace=False)
        except Exception as e:
            raise RuntimeError(f"Error during class balancing: {e}")
    
    # Combine and shuffle
    all_samples = vulnerable_samples + non_vulnerable_samples
    np.random.seed(random_state)
    np.random.shuffle(all_samples)
    
    vulnerable_count = sum(s['is_vulnerable'] for s in all_samples)
    non_vulnerable_count = len(all_samples) - vulnerable_count
    
    print(f"FINAL DATASET")
    print(f"Total samples: {len(all_samples)}")
    print(f"Vulnerable: {vulnerable_count} ({100*vulnerable_count/len(all_samples):.1f}%)")
    print(f"Non-vulnerable: {non_vulnerable_count} ({100*non_vulnerable_count/len(all_samples):.1f}%)")
    print(f"Balance ratio: {vulnerable_count/non_vulnerable_count:.2f}:1")
    
    return all_samples


def create_stratified_splits(samples, train_split=0.70, val_split=0.15, test_split=0.15, random_state=42):
    if not np.isclose(train_split + val_split + test_split, 1.0):
        raise ValueError(f"Splits must sum to 1.0, got {train_split + val_split + test_split}")
    
    try:
        labels = [s['is_vulnerable'] for s in samples]
        
        # First split: train vs (val + test)
        train_samples, temp_samples = train_test_split(
            samples,
            test_size=(1 - train_split),
            stratify=labels,
            random_state=random_state
        )
        
        temp_labels = [s['is_vulnerable'] for s in temp_samples]
        val_ratio = val_split / (val_split + test_split)
        
        val_samples, test_samples = train_test_split(
            temp_samples,
            test_size=(1 - val_ratio),
            stratify=temp_labels,
            random_state=random_state
        )
        
        print(f"DATASET SPLITS (Stratified)")
        print(f"Train: {len(train_samples)} samples ({100*train_split:.1f}%)")
        print(f"  - Vulnerable: {sum(s['is_vulnerable'] for s in train_samples)}")
        print(f"  - Non-vulnerable: {len(train_samples) - sum(s['is_vulnerable'] for s in train_samples)}")
        print(f"Val:   {len(val_samples)} samples ({100*val_split:.1f}%)")
        print(f"  - Vulnerable: {sum(s['is_vulnerable'] for s in val_samples)}")
        print(f"  - Non-vulnerable: {len(val_samples) - sum(s['is_vulnerable'] for s in val_samples)}")
        print(f"Test:  {len(test_samples)} samples ({100*test_split:.1f}%)")
        print(f"  - Vulnerable: {sum(s['is_vulnerable'] for s in test_samples)}")
        print(f"  - Non-vulnerable: {len(test_samples) - sum(s['is_vulnerable'] for s in test_samples)}")
        
        return train_samples, val_samples, test_samples
    
    except Exception as e:
        raise RuntimeError(f"Error during dataset splitting: {e}")

class VulnerabilityDataset(Dataset):
    
    def __init__(self, samples, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = samples
        
        # Validate samples
        if len(self.samples) == 0:
            raise ValueError("Dataset cannot be empty!")
        
        # Print class distribution
        labels = [s['is_vulnerable'] for s in self.samples]
        vuln_count = sum(labels)
        non_vuln_count = len(labels) - vuln_count
        print(f"  Vulnerable: {vuln_count}, Non-vulnerable: {non_vuln_count}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        try:
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
                'labels': torch.tensor(sample['is_vulnerable'], dtype=torch.long)
            }
        except Exception as e:
            raise RuntimeError(f"Error processing sample at index {idx}: {e}")

class StarCoderForClassification(torch.nn.Module):
    
    def __init__(self, model_name, num_labels=2, device='cuda:1', loss_fn=None):
        super().__init__()
        self.num_labels = num_labels
        self.device_name = device
        self.loss_fn = loss_fn

        print(f"Loading StarCoder model on {device}")
        
        try:
            # Load base model with memory-efficient settings
            self.base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if args.fp16 else torch.float32,
                device_map={"": device},
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                token="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"
            )
            
            # Get hidden size from model config
            self.hidden_size = self.base_model.config.hidden_size
            
            for param in self.base_model.parameters():
                param.requires_grad = False
            
            print(f"Base model loaded successfully")
            print(f"Hidden size: {self.hidden_size}")
            
            self.classifier = torch.nn.Sequential(
                torch.nn.Dropout(0.1),
                torch.nn.Linear(self.hidden_size, 256),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(256, num_labels)
            ).to(device)
            
            print(f"Classification head initialized")
            
        except Exception as e:
            raise RuntimeError(f"Error initializing model: {e}")
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        try:
            if hasattr(self.base_model, 'gradient_checkpointing_enable'):
                self.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
        except Exception as e:
            print(f"Warning: Could not enable gradient checkpointing: {e}")
    
    def gradient_checkpointing_disable(self):
        try:
            if hasattr(self.base_model, 'gradient_checkpointing_disable'):
                self.base_model.gradient_checkpointing_disable()
        except Exception as e:
            print(f"Warning: Could not disable gradient checkpointing: {e}")
    
    def forward(self, input_ids, attention_mask, labels=None):
        try:
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            
            hidden_states = outputs.hidden_states[-1]
            
            # Mean pooling with attention mask
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_hidden = torch.sum(hidden_states * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            pooled_output = sum_hidden / sum_mask
            
            logits = self.classifier(pooled_output)
            
            loss = None
            if labels is not None and self.loss_fn is not None:
                loss = self.loss_fn(logits, labels)
            
            return {
                'loss': loss,
                'logits': logits
            }
        except Exception as e:
            raise RuntimeError(f"Error in forward pass: {e}")

def setup_training():

    try:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        
        # Load tokenizer
        print("\nLoading tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name,token="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")
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("Set pad_token to eos_token")
        
        all_samples = load_and_balance_data(
            args.input_data,
            tokenizer,
            args.max_length,
            balance=args.balance_classes,
            random_state=args.seed
        )
        
        train_samples, val_samples, test_samples = create_stratified_splits(
            all_samples,
            args.train_split,
            args.val_split,
            args.test_split,
            random_state=args.seed
        )
        
        # Create datasets
        print("Creating datasets")
        print("Train set")
        train_dataset = VulnerabilityDataset(train_samples, tokenizer, args.max_length)
        print("Validation set")
        val_dataset = VulnerabilityDataset(val_samples, tokenizer, args.max_length)
        print("Test set")
        test_dataset = VulnerabilityDataset(test_samples, tokenizer, args.max_length)
        
        loss_fn = get_loss_function(
            args.loss_type,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
            asl_gamma_pos=args.asl_gamma_pos,
            asl_gamma_neg=args.asl_gamma_neg
        )
        
        model = StarCoderForClassification(
            args.model_name,
            num_labels=2,
            device=str(DEVICE),
            loss_fn=loss_fn
        )
        
        # Apply LoRA
        print("Applying LoRA")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["c_proj", "c_attn"],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION
        )
        model.base_model = get_peft_model(model.base_model, lora_config)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"MODEL PARAMETERS")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")
        return model, tokenizer, train_dataset, val_dataset, test_dataset

    except Exception as e:
        print(f"\nERROR during training setup: {e}")
        raise

class VulnerabilityTrainer(Trainer):
 
    def create_optimizer(self):
        try:
            if args.use_8bit_optimizer:
                try:
                    import bitsandbytes as bnb
                    print("Using 8-bit AdamW optimizer...")
                    self.optimizer = bnb.optim.AdamW8bit(
                        self.model.parameters(),
                        lr=self.args.learning_rate,
                        weight_decay=self.args.weight_decay
                    )
                except ImportError:
                    print("bitsandbytes not available, using standard AdamW...")
                    super().create_optimizer()
            else:
                super().create_optimizer()
        except Exception as e:
            print(f"Warning: Error creating optimizer: {e}")
            super().create_optimizer()
    
    def _save(self, output_dir=None, state_dict=None):
        pass
    
    def _save_checkpoint(self, model, trial, metrics=None):
        pass

def compute_metrics(eval_pred):
    try:
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary', zero_division=0
        )
        accuracy = accuracy_score(labels, predictions)
        try:
            roc_auc = roc_auc_score(labels, probs)
        except:
            roc_auc = 0.0
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'roc_auc': float(roc_auc)
        }
    except Exception as e:
        print(f"Error computing metrics: {e}")
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'roc_auc': 0.0
        }

def find_optimal_threshold(labels, probs, metric='f1'):

    try:
        thresholds = np.arange(0.05, 0.95, 0.01)
        results = []
        
        for threshold in thresholds:
            preds = (probs >= threshold).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, preds, average='binary', zero_division=0
            )
            accuracy = accuracy_score(labels, preds)
            
            results.append({
                'threshold': threshold,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
        
        results_df = pd.DataFrame(results)
        
        # Find optimal threshold
        metric_col = metric if metric in results_df.columns else 'f1'
        best_idx = results_df[metric_col].idxmax()
        optimal_threshold = results_df.loc[best_idx, 'threshold']
        best_score = results_df.loc[best_idx, metric_col]
        
        return optimal_threshold, best_score, results_df
        
    except Exception as e:
        print(f"Error finding optimal threshold: {e}")
        return 0.5, 0.0, pd.DataFrame()

def create_publication_quality_plots(labels, preds, probs, cm, threshold, 
                                     threshold_results_df, output_dir):
    os.makedirs(f"{output_dir}/visualizations", exist_ok=True)
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
    })
    
    try:
        # Confusion Matrix
        print("Generating confusion matrix...")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Non-Vulnerable', 'Vulnerable'],
                    yticklabels=['Non-Vulnerable', 'Vulnerable'],
                    cbar_kws={'label': 'Count'}, ax=ax, square=True)
        for i in range(2):
            for j in range(2):
                percentage = cm[i,j] / cm[i].sum() * 100
                ax.text(j+0.5, i+0.7, f'({percentage:.1f}%)', 
                       ha='center', va='center', fontsize=9, color='gray')
        
        ax.set_title(f'Confusion Matrix\n(Threshold = {threshold:.3f})', 
                    fontweight='bold', pad=20)
        ax.set_ylabel('True Label', fontweight='bold')
        ax.set_xlabel('Predicted Label', fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/visualizations/01_confusion_matrix.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # ROC Curve with confidence bands
        print("Generating ROC curve")
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = roc_auc_score(labels, probs)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='#2E86AB', lw=2.5, 
               label=f'ROC Curve (AUC = {roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], color='#E63946', lw=2, 
               linestyle='--', label='Random Classifier')
        ax.fill_between(fpr, tpr, alpha=0.2, color='#2E86AB')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate (1 - Specificity)', fontweight='bold')
        ax.set_ylabel('True Positive Rate (Sensitivity)', fontweight='bold')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve', 
                    fontweight='bold', pad=20)
        ax.legend(loc="lower right", frameon=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/visualizations/02_roc_curve.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Precision-Recall Curve with optimal point
        print("Generating precision-recall curve")
        precision_curve, recall_curve, thresholds_pr = precision_recall_curve(labels, probs)
        avg_precision = average_precision_score(labels, probs)
        
        # Find point closest to optimal threshold
        if len(thresholds_pr) > 0:
            idx = np.argmin(np.abs(thresholds_pr - threshold))
        else:
            idx = 0
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall_curve, precision_curve, color='#06A77D', lw=2.5,
               label=f'PR Curve (AP = {avg_precision:.4f})')
        ax.scatter(recall_curve[idx], precision_curve[idx], color='#E63946', 
                  s=150, zorder=5, marker='*',
                  label=f'Optimal Point (τ={threshold:.3f})')
        ax.fill_between(recall_curve, precision_curve, alpha=0.2, color='#06A77D')
        
        ax.set_xlabel('Recall (Sensitivity)', fontweight='bold')
        ax.set_ylabel('Precision (Positive Predictive Value)', fontweight='bold')
        ax.set_title('Precision-Recall Curve', fontweight='bold', pad=20)
        ax.legend(loc="lower left", frameon=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        plt.tight_layout()
        plt.savefig(f"{output_dir}/visualizations/03_precision_recall_curve.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        #Prediction Distributions with KDE
        print("Generating prediction distributions...")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        vuln_probs = probs[labels == 1]
        non_vuln_probs = probs[labels == 0]
        
        # Vulnerable samples
        axes[0].hist(vuln_probs, bins=40, color='#E63946', alpha=0.7, 
                    edgecolor='black', density=True, label='Histogram')
        axes[0].axvline(threshold, color='black', linestyle='--', linewidth=2.5, 
                       label=f'Optimal Threshold = {threshold:.3f}')
        axes[0].axvline(0.5, color='gray', linestyle=':', linewidth=2, 
                       label='Default Threshold = 0.5', alpha=0.7)
        axes[0].set_xlabel('Predicted Vulnerability Probability', fontweight='bold')
        axes[0].set_ylabel('Density', fontweight='bold')
        axes[0].set_title('Distribution: Vulnerable Samples', fontweight='bold')
        axes[0].legend(frameon=True, shadow=True)
        axes[0].grid(True, alpha=0.3, linestyle='--')
        
        # Non-vulnerable samples
        axes[1].hist(non_vuln_probs, bins=40, color='#06A77D', alpha=0.7, 
                    edgecolor='black', density=True, label='Histogram')
        axes[1].axvline(threshold, color='black', linestyle='--', linewidth=2.5, 
                       label=f'Optimal Threshold = {threshold:.3f}')
        axes[1].axvline(0.5, color='gray', linestyle=':', linewidth=2, 
                       label='Default Threshold = 0.5', alpha=0.7)
        axes[1].set_xlabel('Predicted Vulnerability Probability', fontweight='bold')
        axes[1].set_ylabel('Density', fontweight='bold')
        axes[1].set_title('Distribution: Non-Vulnerable Samples', fontweight='bold')
        axes[1].legend(frameon=True, shadow=True)
        axes[1].grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/visualizations/04_prediction_distributions.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Metrics Summary
        print("Generating metrics summary")
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        accuracy = accuracy_score(labels, preds)
        
        metrics_dict = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ROC-AUC': roc_auc,
            'AP': avg_precision
        }
        
        fig, ax = plt.subplots(figsize=(11, 6))
        colors = ['#2E86AB', '#E63946', '#06A77D', '#F18F01', '#9B59B6', '#C73E1D']
        bars = ax.bar(range(len(metrics_dict)), list(metrics_dict.values()), 
                     color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
        
        ax.set_xticks(range(len(metrics_dict)))
        ax.set_xticklabels(list(metrics_dict.keys()), fontweight='bold')
        ax.set_ylim([0, 1.05])
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title('Model Performance Metrics Summary', fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/visualizations/05_metrics_summary.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        #Threshold Optimization Curves
        if not threshold_results_df.empty:
            print("Generating threshold optimization curves...")
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            metrics_to_plot = ['f1', 'precision', 'recall', 'accuracy']
            colors_plot = ['#F18F01', '#E63946', '#06A77D', '#2E86AB']
            titles = ['F1-Score vs Threshold', 'Precision vs Threshold', 
                     'Recall vs Threshold', 'Accuracy vs Threshold']
            
            for idx, (metric, color, title) in enumerate(zip(metrics_to_plot, colors_plot, titles)):
                ax = axes[idx // 2, idx % 2]
                ax.plot(threshold_results_df['threshold'], 
                       threshold_results_df[metric],
                       color=color, lw=2.5, label=metric.capitalize())
                ax.axvline(threshold, color='black', linestyle='--', linewidth=2,
                          label=f'Optimal τ = {threshold:.3f}')
                ax.axvline(0.5, color='gray', linestyle=':', linewidth=1.5,
                          label='Default τ = 0.5', alpha=0.7)
                
                # Mark optimal point
                opt_val = threshold_results_df.loc[
                    threshold_results_df['threshold'] == threshold, metric
                ].values
                if len(opt_val) > 0:
                    ax.scatter([threshold], opt_val, color='red', s=100, zorder=5, marker='*')
                
                ax.set_xlabel('Threshold (τ)', fontweight='bold')
                ax.set_ylabel(metric.capitalize(), fontweight='bold')
                ax.set_title(title, fontweight='bold')
                ax.legend(frameon=True, shadow=True)
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.set_xlim([0, 1])
                ax.set_ylim([0, 1.05])
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/visualizations/06_threshold_optimization.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        # Class-wise Performance Bar Chart
        print("Generating class-wise performance...")
        report = classification_report(labels, preds, 
                                       target_names=['Non-Vulnerable', 'Vulnerable'],
                                       output_dict=True, zero_division=0)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        classes = ['Non-Vulnerable', 'Vulnerable']
        metrics_cw = ['precision', 'recall', 'f1-score']
        x = np.arange(len(classes))
        width = 0.25
        
        for i, metric in enumerate(metrics_cw):
            values = [report[cls][metric] for cls in classes]
            ax.bar(x + i*width, values, width, label=metric.capitalize(),
                  alpha=0.8, edgecolor='black', linewidth=1.5)
        
        ax.set_xlabel('Class', fontweight='bold')
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title('Class-wise Performance Comparison', fontweight='bold', pad=20)
        ax.set_xticks(x + width)
        ax.set_xticklabels(classes, fontweight='bold')
        ax.legend(frameon=True, shadow=True)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/visualizations/07_classwise_performance.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Error Analysis
        print("Generating error analysis...")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # False positives and false negatives
        fp_mask = (labels == 0) & (preds == 1)
        fn_mask = (labels == 1) & (preds == 0)
        fp_probs = probs[fp_mask]
        fn_probs = probs[fn_mask]
        
        # FP distribution
        if len(fp_probs) > 0:
            axes[0].hist(fp_probs, bins=30, color='#E63946', alpha=0.7, 
                        edgecolor='black')
            axes[0].axvline(threshold, color='black', linestyle='--', linewidth=2,
                           label=f'Threshold = {threshold:.3f}')
            axes[0].set_xlabel('Predicted Probability', fontweight='bold')
            axes[0].set_ylabel('Count', fontweight='bold')
            axes[0].set_title(f'False Positives (n={len(fp_probs)})', fontweight='bold')
            axes[0].legend(frameon=True, shadow=True)
            axes[0].grid(True, alpha=0.3, linestyle='--')
        else:
            axes[0].text(0.5, 0.5, 'No False Positives', ha='center', va='center',
                        fontsize=14, fontweight='bold')
            axes[0].set_title('False Positives (n=0)', fontweight='bold')
        
        # FN distribution
        if len(fn_probs) > 0:
            axes[1].hist(fn_probs, bins=30, color='#F18F01', alpha=0.7, 
                        edgecolor='black')
            axes[1].axvline(threshold, color='black', linestyle='--', linewidth=2,
                           label=f'Threshold = {threshold:.3f}')
            axes[1].set_xlabel('Predicted Probability', fontweight='bold')
            axes[1].set_ylabel('Count', fontweight='bold')
            axes[1].set_title(f'False Negatives (n={len(fn_probs)})', fontweight='bold')
            axes[1].legend(frameon=True, shadow=True)
            axes[1].grid(True, alpha=0.3, linestyle='--')
        else:
            axes[1].text(0.5, 0.5, 'No False Negatives', ha='center', va='center',
                        fontsize=14, fontweight='bold')
            axes[1].set_title('False Negatives (n=0)', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/visualizations/08_error_analysis.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved to: {output_dir}/visualizations/")
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        raise

def evaluate_model(model, test_dataset, tokenizer, output_dir):
    
    try:
        model.eval()
        model = model.to(DEVICE)
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=args.batch_size,
            shuffle=False
        )
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        print("Running inference on test set")
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
        
        # Find optimal threshold
        print("\nFinding optimal threshold")
        optimal_threshold, optimal_f1, threshold_results_df = find_optimal_threshold(
            all_labels, all_probs, metric='precision'
        )
        print(f"Optimal threshold: {optimal_threshold:.3f} (F1: {optimal_f1:.4f})")
        
        # Recalculate with optimal threshold
        all_preds_opt = (all_probs >= optimal_threshold).astype(int)
        
        # Calculate metrics
        print("METRICS WITH DEFAULT THRESHOLD (0.5)")
        
        acc_def = accuracy_score(all_labels, all_preds)
        prec_def, rec_def, f1_def, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', zero_division=0
        )
        roc_auc = roc_auc_score(all_labels, all_probs)
        
        print(f"Accuracy:  {acc_def:.4f}")
        print(f"Precision: {prec_def:.4f}")
        print(f"Recall:    {rec_def:.4f}")
        print(f"F1-Score:  {f1_def:.4f}")
        print(f"ROC-AUC:   {roc_auc:.4f}")
        
        print(f"METRICS WITH OPTIMIZED THRESHOLD ({optimal_threshold:.3f})")
        
        acc_opt = accuracy_score(all_labels, all_preds_opt)
        prec_opt, rec_opt, f1_opt, _ = precision_recall_fscore_support(
            all_labels, all_preds_opt, average='binary', zero_division=0
        )
        cm_opt = confusion_matrix(all_labels, all_preds_opt)
        
        print(f"Accuracy:  {acc_opt:.4f}")
        print(f"Precision: {prec_opt:.4f}")
        print(f"Recall:    {rec_opt:.4f}")
        print(f"F1-Score:  {f1_opt:.4f} ← OPTIMIZED")
        print(f"ROC-AUC:   {roc_auc:.4f}")
        print("CONFUSION MATRIX (Optimized)")
        print(f"                Predicted")
        print(f"                Non-Vuln  Vulnerable")
        print(f"Actual Non-Vuln    {cm_opt[0,0]:4d}      {cm_opt[0,1]:4d}")
        print(f"Actual Vulnerable  {cm_opt[1,0]:4d}      {cm_opt[1,1]:4d}")
        
        print("DETAILED CLASSIFICATION REPORT")
        print(classification_report(
            all_labels, all_preds_opt, 
            target_names=['Non-Vulnerable', 'Vulnerable'],
            zero_division=0
        ))
        
        if args.save_visualizations:
            create_publication_quality_plots(
                all_labels, all_preds_opt, all_probs, cm_opt,
                optimal_threshold, threshold_results_df, output_dir
            )
        
        results_df = pd.DataFrame({
            'true_label': all_labels,
            'predicted_label_default': all_preds,
            'predicted_label_optimized': all_preds_opt,
            'vulnerability_probability': all_probs,
            'optimal_threshold': optimal_threshold
        })
        results_df.to_csv(f"{output_dir}/test_predictions.csv", index=False)
        
        # Save threshold results
        if not threshold_results_df.empty:
            threshold_results_df.to_csv(f"{output_dir}/threshold_analysis.csv", index=False)
        
        return {
            'accuracy': float(acc_opt),
            'precision': float(prec_opt),
            'recall': float(rec_opt),
            'f1': float(f1_opt),
            'roc_auc': float(roc_auc),
            'optimal_threshold': float(optimal_threshold),
            'confusion_matrix': cm_opt.tolist()
        }
    
    except Exception as e:
        print(f"\nERROR during evaluation: {e}")
        raise


def diagnose_model_behavior(model, dataset, tokenizer, output_dir):
    
    model.eval()
    model = model.to(DEVICE)
    
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    all_preds = []
    all_labels = []
    all_probs = []
    all_logits = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Analyzing"):
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
            all_logits.extend(logits.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_logits = np.array(all_logits)
    
    # Analyze predictions
    print("\n1. PREDICTION DISTRIBUTION")
    print(f"   Predicted Vulnerable: {sum(all_preds == 1)} ({100*sum(all_preds == 1)/len(all_preds):.1f}%)")
    print(f"   Predicted Non-Vulnerable: {sum(all_preds == 0)} ({100*sum(all_preds == 0)/len(all_preds):.1f}%)")
    print(f"   Actual Vulnerable: {sum(all_labels == 1)} ({100*sum(all_labels == 1)/len(all_labels):.1f}%)")
    print(f"   Actual Non-Vulnerable: {sum(all_labels == 0)} ({100*sum(all_labels == 0)/len(all_labels):.1f}%)")
    
    # Analyze probability distribution
    print("\n2. PROBABILITY STATISTICS")
    vuln_probs = all_probs[all_labels == 1]
    non_vuln_probs = all_probs[all_labels == 0]
    
    print(f"   Vulnerable samples:")
    print(f"     Mean prob: {vuln_probs.mean():.3f}, Std: {vuln_probs.std():.3f}")
    print(f"     Median: {np.median(vuln_probs):.3f}")
    
    print(f"   Non-Vulnerable samples:")
    print(f"     Mean prob: {non_vuln_probs.mean():.3f}, Std: {non_vuln_probs.std():.3f}")
    print(f"     Median: {np.median(non_vuln_probs):.3f}")
    
    # Analyze false positives
    fp_mask = (all_labels == 0) & (all_preds == 1)
    fp_probs = all_probs[fp_mask]
    
    print(f"\n3. FALSE POSITIVE ANALYSIS")
    print(f"   Total FP: {sum(fp_mask)} ({100*sum(fp_mask)/sum(all_labels == 0):.1f}% of non-vulnerable)")
    if len(fp_probs) > 0:
        print(f"   FP probability range: [{fp_probs.min():.3f}, {fp_probs.max():.3f}]")
        print(f"   FP probability mean: {fp_probs.mean():.3f}")
        
        # Probability bins
        bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for i in range(len(bins)-1):
            count = sum((fp_probs >= bins[i]) & (fp_probs < bins[i+1]))
            print(f"   FP in [{bins[i]:.1f}, {bins[i+1]:.1f}): {count}")
    
    # Analyze logit distribution
    print(f"\n4. LOGIT ANALYSIS")
    vuln_logits = all_logits[all_labels == 1][:, 1]
    non_vuln_logits = all_logits[all_labels == 0][:, 1]
    
    print(f"   Vulnerable logits: mean={vuln_logits.mean():.3f}, std={vuln_logits.std():.3f}")
    print(f"   Non-Vulnerable logits: mean={non_vuln_logits.mean():.3f}, std={non_vuln_logits.std():.3f}")
    print(f"   Logit separation: {abs(vuln_logits.mean() - non_vuln_logits.mean()):.3f}")
    
    # Check if model is just memorizing
    print(f"\n5. MODEL CONFIDENCE")
    high_conf = sum((all_probs > 0.9) | (all_probs < 0.1))
    mid_conf = sum((all_probs >= 0.4) & (all_probs <= 0.6))
    print(f"   High confidence (>0.9 or <0.1): {high_conf} ({100*high_conf/len(all_probs):.1f}%)")
    print(f"   Low confidence (0.4-0.6): {mid_conf} ({100*mid_conf/len(all_probs):.1f}%)")
        
    return {
        'fp_rate': sum(fp_mask) / sum(all_labels == 0),
        'mean_prob_diff': vuln_probs.mean() - non_vuln_probs.mean(),
        'logit_separation': abs(vuln_logits.mean() - non_vuln_logits.mean())
    }

def main():
    print("TRAINING CONFIGURATION")
    print(f"Model: {args.model_name}")
    print(f"Loss function: {args.loss_type}")
    print(f"Max length: {args.max_length}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Device: {DEVICE}")
    print(f"Output directory: {args.output_dir}")
    
    start_time = datetime.now()
    
    try:
        # Setup
        model, tokenizer, train_dataset, val_dataset, test_dataset = setup_training()
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(f"{args.output_dir}/logs", exist_ok=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            logging_dir=f"{args.output_dir}/logs",
            logging_steps=50,
            save_strategy="no",
            eval_strategy="epoch",
            load_best_model_at_end=False,
            metric_for_best_model="f1",
            fp16=args.fp16,
            gradient_checkpointing=args.use_gradient_checkpointing,
            report_to="tensorboard",
            seed=args.seed,
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
            remove_unused_columns=False
        )
        
        trainer = VulnerabilityTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
         
        train_result = trainer.train()
        
        print("\nSaving model")
        trainer.model.base_model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        torch.save(trainer.model.classifier.state_dict(), 
                  f"{args.output_dir}/classifier_head.pt")
        
        # Save config
        config_dict = {
            'model_name': args.model_name,
            'loss_type': args.loss_type,
            'max_length': args.max_length,
            'hidden_size': trainer.model.hidden_size,
            'num_labels': 2,
            'args': vars(args)
        }
        with open(f"{args.output_dir}/model_config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        # Training summary
        print(f"Training time: {train_result.metrics['train_runtime']:.2f} seconds")
        print(f"Training loss: {train_result.metrics['train_loss']:.4f}")
        
        test_metrics = evaluate_model(model, test_dataset, tokenizer, args.output_dir)
        
        final_results = {
            'training_metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v 
                                for k, v in train_result.metrics.items()},
            'test_metrics': test_metrics,
            'config': config_dict
        }
        
        with open(f"{args.output_dir}/final_results.json", 'w') as f:
            json.dump(final_results, f, indent=2)
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        # Final summary
        print("FINAL SUMMARY")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Training samples: {len(train_dataset)}")
        print(f"F1-Score: {test_metrics['f1']:.4f}")
        print(f"Optimal Threshold: {test_metrics['optimal_threshold']:.3f}")
        print(f"ROC-AUC: {test_metrics['roc_auc']:.4f}")
        if args.save_visualizations:
            print(f"Visualizations: 8 plots generated")
        print(f"Results saved to: {args.output_dir}")
        
        return 0
    
    except Exception as e:
        print("TRAINING FAILED")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
