
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_curve, auc, classification_report
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import os

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

MODEL_NAME = "Qwen/Qwen2.5-Coder-0.5B"
TARGET_GPU = 0 
MAX_TOKENS = 512
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 5
LORA_R = 32  
LORA_ALPHA = 64
LORA_DROPOUT = 0.1
WARMUP_RATIO = 0.1
FOCAL_LOSS_GAMMA = 2.0
FOCAL_LOSS_ALPHA = 0.75 

os.makedirs("./qwen_vulnerability_classifier", exist_ok=True)
os.makedirs("./thesis_visualizations", exist_ok=True)
os.makedirs("./logs", exist_ok=True)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits, labels):

        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        alpha_t = torch.where(labels == 1, self.alpha, 1 - self.alpha)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class FocalLossTrainer(Trainer):
    def __init__(self, *args, focal_loss_alpha=0.75, focal_loss_gamma=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_loss = FocalLoss(alpha=focal_loss_alpha, gamma=focal_loss_gamma)
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = self.focal_loss(logits, labels)
        
        return (loss, outputs) if return_outputs else loss

def augment_code(code, num_augmentations=1):
    augmented = []
    
    for _ in range(num_augmentations):
        aug_code = code
        
        if random.random() > 0.5:
            aug_code = ' '.join(aug_code.split())
        if random.random() > 0.5:
            aug_code = aug_code.replace(';', ';\n')
        augmented.append(aug_code)
    
    return augmented

def load_and_filter_jsonl(file_path, tokenizer, max_tokens=512):
    print(f"Loading data from {file_path}")
    codes = []
    labels = []
    
    with open(file_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            code = entry['code']
            is_vulnerable = entry['is_vulnerable']
            tokens = tokenizer(code, truncation=False, add_special_tokens=True)
            token_count = len(tokens['input_ids'])
            
            if token_count <= max_tokens:
                codes.append(code)
                labels.append(int(is_vulnerable))
    
    print(f"Total samples after filtering (≤{max_tokens} tokens): {len(codes)}")
    return codes, labels

def balance_classes_with_augmentation(codes, labels, augment_minority=True):
    codes = np.array(codes)
    labels = np.array(labels)
    idx_vul = np.where(labels == 1)[0]
    idx_nonvul = np.where(labels == 0)[0]
    print(f"Before balancing - Vulnerable: {len(idx_vul)}, Non-vulnerable: {len(idx_nonvul)}")
    
    if len(idx_vul) < len(idx_nonvul):
        minority_idx = idx_vul
        majority_idx = idx_nonvul
        minority_label = 1
        majority_label = 0
    else:
        minority_idx = idx_nonvul
        majority_idx = idx_vul
        minority_label = 0
        majority_label = 1
    
    majority_size = len(majority_idx)
    minority_size = len(minority_idx)
    
    if augment_minority:
        print(f"Augmenting minority class (label={minority_label}) to match majority class...")
        balanced_codes = []
        balanced_labels = []
        
        for idx in majority_idx:
            balanced_codes.append(codes[idx])
            balanced_labels.append(labels[idx])
        
        for idx in minority_idx:
            balanced_codes.append(codes[idx])
            balanced_labels.append(labels[idx])
        
        needed_augmented = majority_size - minority_size
        print(f"Need to create {needed_augmented} augmented samples from {minority_size} minority samples")
        
        if needed_augmented > 0:
            samples_to_augment = np.random.choice(minority_idx, needed_augmented, replace=True)
            for idx in samples_to_augment:
                original_code = codes[idx]
                aug_versions = augment_code(original_code, num_augmentations=1)
                balanced_codes.append(aug_versions[0])
                balanced_labels.append(labels[idx])
        
        combined = list(zip(balanced_codes, balanced_labels))
        random.shuffle(combined)
        balanced_codes, balanced_labels = zip(*combined)
        balanced_codes = list(balanced_codes)
        balanced_labels = list(balanced_labels)
        
    else:
        min_class_size = min(len(idx_vul), len(idx_nonvul))
        selected_vul = np.random.choice(idx_vul, min_class_size, replace=False)
        selected_nonvul = np.random.choice(idx_nonvul, min_class_size, replace=False)
        
        selected_indices = np.concatenate([selected_vul, selected_nonvul])
        np.random.shuffle(selected_indices)
        
        balanced_codes = codes[selected_indices].tolist()
        balanced_labels = labels[selected_indices].tolist()
    
    vul_count = sum(balanced_labels)
    nonvul_count = len(balanced_labels) - vul_count
    
    print(f"After balancing - Total samples: {len(balanced_codes)}, "
          f"Vulnerable: {vul_count}, Non-vulnerable: {nonvul_count}")
    print(f"Ratio - Vulnerable: {vul_count/(vul_count+nonvul_count)*100:.1f}%, "
          f"Non-vulnerable: {nonvul_count/(vul_count+nonvul_count)*100:.1f}%")
    
    return balanced_codes, balanced_labels


# def balance_classes_with_augmentation(codes, labels, augment_minority=True):
#     codes = np.array(codes)
#     labels = np.array(labels)
    
#     idx_vul = np.where(labels == 1)[0]
#     idx_nonvul = np.where(labels == 0)[0]
    
#     print(f"Before balancing - Vulnerable: {len(idx_vul)}, Non-vulnerable: {len(idx_nonvul)}")
    
#     max_class_size = max(len(idx_vul), len(idx_nonvul))
#     min_class_size = min(len(idx_vul), len(idx_nonvul))
    
#     if len(idx_vul) < len(idx_nonvul):
#         minority_idx = idx_vul
#         majority_idx = idx_nonvul
#         minority_label = 1
#     else:
#         minority_idx = idx_nonvul
#         majority_idx = idx_vul
#         minority_label = 0
    
#     if augment_minority and len(minority_idx) > 0:        
#         selected_majority = np.random.choice(majority_idx, min_class_size, replace=False)
#         selected_minority = minority_idx
        
#         augmented_codes = []
#         augmented_labels = []
        
#         for idx in selected_minority:
#             original_code = codes[idx]
#             augmented_codes.append(original_code)
#             augmented_labels.append(labels[idx])
            
#             aug_versions = augment_code(original_code, num_augmentations=1)
#             for aug_code in aug_versions:
#                 augmented_codes.append(aug_code)
#                 augmented_labels.append(labels[idx])
        
#         for idx in selected_majority:
#             augmented_codes.append(codes[idx])
#             augmented_labels.append(labels[idx])
        
#         combined = list(zip(augmented_codes, augmented_labels))
#         random.shuffle(combined)
#         balanced_codes, balanced_labels = zip(*combined)
#         balanced_codes = list(balanced_codes)
#         balanced_labels = list(balanced_labels)
#     else:
#         min_class_size = min(len(idx_vul), len(idx_nonvul))
#         selected_vul = np.random.choice(idx_vul, min_class_size, replace=False)
#         selected_nonvul = np.random.choice(idx_nonvul, min_class_size, replace=False)
        
#         selected_indices = np.concatenate([selected_vul, selected_nonvul])
#         np.random.shuffle(selected_indices)
        
#         balanced_codes = codes[selected_indices].tolist()
#         balanced_labels = labels[selected_indices].tolist()
    
#     print(f"After balancing - Total samples: {len(balanced_codes)}, "
#           f"Vulnerable: {sum(balanced_labels)}, Non-vulnerable: {len(balanced_labels) - sum(balanced_labels)}")
    
#     return balanced_codes, balanced_labels


def stratified_split(codes, labels, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    train_codes, temp_codes, train_labels, temp_labels = train_test_split(
        codes, labels, 
        test_size=(val_ratio + test_ratio), 
        stratify=labels, 
        random_state=SEED
    )
    
    val_size = val_ratio / (val_ratio + test_ratio)
    val_codes, test_codes, val_labels, test_labels = train_test_split(
        temp_codes, temp_labels,
        test_size=(1 - val_size),
        stratify=temp_labels,
        random_state=SEED
    )
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_codes)} samples (Vul: {sum(train_labels)}, Non-vul: {len(train_labels) - sum(train_labels)})")
    print(f"  Val:   {len(val_codes)} samples (Vul: {sum(val_labels)}, Non-vul: {len(val_labels) - sum(val_labels)})")
    print(f"  Test:  {len(test_codes)} samples (Vul: {sum(test_labels)}, Non-vul: {len(test_labels) - sum(test_labels)})")
    
    return train_codes, val_codes, test_codes, train_labels, val_labels, test_labels

def create_dataset(codes, labels, tokenizer, max_length=512):
    encodings = tokenizer(codes, truncation=True, padding=False, max_length=max_length)
    
    dataset_dict = {
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'labels': labels
    }
    
    return Dataset.from_dict(dataset_dict)

def setup_model_and_tokenizer(model_name, target_gpu):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        quantization_config=bnb_config,
        device_map={'': target_gpu},
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    
    model.config.pad_token_id = tokenizer.pad_token_id
    
    print("Preparing model for k-bit training with gradient checkpointing...")
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True
    )

    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        inference_mode=False,
    )
    
    model = get_peft_model(model, lora_config)
    # print(f"Model loaded on device: {model.device}")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable parameters: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.2f}%)")
    
    return tokenizer, model

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', zero_division=0
    )
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def train_model(model, tokenizer, train_dataset, val_dataset, target_gpu):
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    training_args = TrainingArguments(
        output_dir="./qwen_vulnerability_classifier",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        report_to="none",
        seed=SEED,
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=False,
        save_total_limit=2,
    )
    
    trainer = FocalLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        focal_loss_alpha=FOCAL_LOSS_ALPHA,
        focal_loss_gamma=FOCAL_LOSS_GAMMA,
    )
    
    print(f"Using Focal Loss (alpha={FOCAL_LOSS_ALPHA}, gamma={FOCAL_LOSS_GAMMA})")
    print(f"Learning rate warmup: {WARMUP_RATIO*100}% of steps")
    trainer.train()
    
    return trainer

def evaluate_model(trainer, test_dataset, test_labels):
    
    predictions_output = trainer.predict(test_dataset)
    logits = predictions_output.predictions
    predictions = np.argmax(logits, axis=-1)
    probabilities = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
    
    test_results = {
        'accuracy': accuracy_score(test_labels, predictions),
        'confusion_matrix': confusion_matrix(test_labels, predictions),
        'classification_report': classification_report(test_labels, predictions, 
                                                       target_names=['Non-Vulnerable', 'Vulnerable']),
    }
    
    precision, recall, f1, support = precision_recall_fscore_support(
        test_labels, predictions, average='binary', zero_division=0
    )
    test_results['precision'] = precision
    test_results['recall'] = recall
    test_results['f1'] = f1
    
    fpr, tpr, thresholds = roc_curve(test_labels, probabilities)
    roc_auc = auc(fpr, tpr)
    test_results['roc_curve'] = (fpr, tpr, thresholds)
    test_results['roc_auc'] = roc_auc
    
    test_results['predictions'] = predictions
    test_results['probabilities'] = probabilities
    test_results['true_labels'] = test_labels
    
    print("\nTest Results:")
    print(f"  Accuracy:  {test_results['accuracy']:.4f}")
    print(f"  Precision: {test_results['precision']:.4f}")
    print(f"  Recall:    {test_results['recall']:.4f}")
    print(f"  F1 Score:  {test_results['f1']:.4f}")
    print(f"  ROC AUC:   {test_results['roc_auc']:.4f}")
    print("\nClassification Report:")
    print(test_results['classification_report'])
    print("\nConfusion Matrix:")
    print(test_results['confusion_matrix'])
    
    return test_results

def create_thesis_visualizations(test_results, training_history=None):

    output_dir = "./thesis_visualizations"

    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.size'] = 10
    
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = test_results['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Vulnerable', 'Vulnerable'],
                yticklabels=['Non-Vulnerable', 'Vulnerable'])
    plt.title('Confusion Matrix - Vulnerability Classification', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/confusion_matrix.png")
    
    # ROC Curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = test_results['roc_curve']
    roc_auc = test_results['roc_auc']
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/roc_curve.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/roc_curve.png")
    
    # Precision-Recall Curve
    from sklearn.metrics import precision_recall_curve
    plt.figure(figsize=(8, 6))
    precision_curve, recall_curve, _ = precision_recall_curve(
        test_results['true_labels'], 
        test_results['probabilities']
    )
    pr_auc = auc(recall_curve, precision_curve)
    plt.plot(recall_curve, precision_curve, color='blue', lw=2,
             label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    plt.savefig(f"{output_dir}/precision_recall_curve.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/precision_recall_curve.png")
    
    # Metrics Summary Bar Chart
    plt.figure(figsize=(10, 6))
    metrics = {
        'Accuracy': test_results['accuracy'],
        'Precision': test_results['precision'],
        'Recall': test_results['recall'],
        'F1 Score': test_results['f1'],
        'ROC AUC': test_results['roc_auc']
    }
    bars = plt.bar(metrics.keys(), metrics.values(), color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    plt.ylim([0, 1.0])
    plt.ylabel('Score', fontsize=12)
    plt.title('Model Performance Metrics Summary', fontsize=14, fontweight='bold')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/metrics_summary.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/metrics_summary.png")
    
    # Performance Comparison Table (as image)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = [
        ['Metric', 'Score', 'Interpretation'],
        ['Accuracy', f"{test_results['accuracy']:.4f}", 'Overall correctness'],
        ['Precision', f"{test_results['precision']:.4f}", 'Positive predictive value'],
        ['Recall', f"{test_results['recall']:.4f}", 'True positive rate'],
        ['F1 Score', f"{test_results['f1']:.4f}", 'Harmonic mean of P&R'],
        ['ROC AUC', f"{test_results['roc_auc']:.4f}", 'Area under ROC curve'],
    ]
    
    table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                     colWidths=[0.3, 0.2, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    for i in range(3):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Model Evaluation Metrics - Detailed Summary', 
              fontsize=14, fontweight='bold', pad=20)
    plt.savefig(f"{output_dir}/metrics_table.png", dpi=300, bbox_inches='tight')
    plt.close()

def main(data_file_path):
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available!")
    
    if TARGET_GPU >= torch.cuda.device_count():
        raise RuntimeError(f"GPU {TARGET_GPU} not available. Found {torch.cuda.device_count()} GPUs.")
    
    print(f"\nUsing GPU {TARGET_GPU}: {torch.cuda.get_device_name(TARGET_GPU)}")
    
    temp_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    codes, labels = load_and_filter_jsonl(data_file_path, temp_tokenizer, MAX_TOKENS)
    codes, labels = balance_classes_with_augmentation(codes, labels, augment_minority=True)
    
    train_codes, val_codes, test_codes, train_labels, val_labels, test_labels = stratified_split(
        codes, labels, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
    )
    
    tokenizer, model = setup_model_and_tokenizer(MODEL_NAME, TARGET_GPU)
    train_dataset = create_dataset(train_codes, train_labels, tokenizer, MAX_TOKENS)
    val_dataset = create_dataset(val_codes, val_labels, tokenizer, MAX_TOKENS)
    test_dataset = create_dataset(test_codes, test_labels, tokenizer, MAX_TOKENS)
    trainer = train_model(model, tokenizer, train_dataset, val_dataset, TARGET_GPU)
    test_results = evaluate_model(trainer, test_dataset, test_labels)
    
    create_thesis_visualizations(test_results)
    
    trainer.save_model("./qwen_vulnerability_classifier/final_model")
    tokenizer.save_pretrained("./qwen_vulnerability_classifier/final_model")
    
    import pickle
    with open("./thesis_visualizations/test_results.pkl", 'wb') as f:
        pickle.dump(test_results, f)
    
    print("Training completed successfully!")
    print(f"F1 Score: {test_results['f1']:.4f}")
    print(f"Visualizations: ./thesis_visualizations/")
    
    return trainer, test_results


if __name__ == "__main__":
    DATA_FILE = "./ds.jsonl"
    trainer, results = main(DATA_FILE)
