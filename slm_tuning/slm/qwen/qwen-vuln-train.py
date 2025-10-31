
import os
import json
import random
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Hard code
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

CUSTOM_CACHE_DIR = "/local/s3905020/temp/"

# Hugging Face Transformers cache
os.environ["TRANSFORMERS_CACHE"] = CUSTOM_CACHE_DIR
# Hugging Face Datasets cache
os.environ["HF_DATASETS_CACHE"] = CUSTOM_CACHE_DIR
# Tokenizer cache
os.environ["HF_HOME"] = CUSTOM_CACHE_DIR

MODEL_NAME = "Qwen/Qwen2.5-Coder-0.5B"
DEVICE = 'cuda:1'  # Hard-coded device
MAX_TOKENS = 512
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 3e-5
NUM_EPOCHS = 3
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

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


def balance_classes(codes, labels):
    codes = np.array(codes)
    labels = np.array(labels)
    
    idx_vul = np.where(labels == 1)[0]
    idx_nonvul = np.where(labels == 0)[0]
    print(f"Before balancing - Vulnerable: {len(idx_vul)}, Non-vulnerable: {len(idx_nonvul)}")
    
    min_class_size = min(len(idx_vul), len(idx_nonvul))
    selected_vul = np.random.choice(idx_vul, min_class_size, replace=False)
    selected_nonvul = np.random.choice(idx_nonvul, min_class_size, replace=False)

    selected_indices = np.concatenate([selected_vul, selected_nonvul])
    np.random.shuffle(selected_indices)
    
    balanced_codes = codes[selected_indices].tolist()
    balanced_labels = labels[selected_indices].tolist()
    
    print(f"After balancing - Total samples: {len(balanced_codes)}, "
          f"Vulnerable: {sum(balanced_labels)}, Non-vulnerable: {len(balanced_labels) - sum(balanced_labels)}")
    
    return balanced_codes, balanced_labels

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

def setup_model_and_tokenizer(model_name, device):
    print(f"\nLoading tokenizer from {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # print(f"Loading model with 4-bit quantization")
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
        device_map={'': 0},  # Hard-code to cuda:0
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    
    model.config.pad_token_id = tokenizer.pad_token_id
    
    print("Preparing model for k-bit training with gradient checkpointing")
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True
    )
    
    print("Configuring LoRA")
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Attention layers
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable parameters: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.2f}%)")
    # model.to(device)
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

def train_model(model, tokenizer, train_dataset, val_dataset, device):

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
        fp16=False,  # Use bf16 instead
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        report_to="none",
        seed=SEED,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    
    return trainer

def evaluate_model(trainer, test_dataset):
    print("Evaluating on test set") 
    test_results = trainer.evaluate(test_dataset)
    
    for key, value in test_results.items():
        print(f"  {key}: {value:.4f}")
    
    return test_results

def main(data_file_path):
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available!")
    
    print(f"\nUsing device: {DEVICE}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    temp_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    codes, labels = load_and_filter_jsonl(data_file_path, temp_tokenizer, MAX_TOKENS)
    codes, labels = balance_classes(codes, labels)
    
    train_codes, val_codes, test_codes, train_labels, val_labels, test_labels = stratified_split(
        codes, labels, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
    )
    
    tokenizer, model = setup_model_and_tokenizer(MODEL_NAME, DEVICE)
    
    print("\nCreating datasets")
    train_dataset = create_dataset(train_codes, train_labels, tokenizer, MAX_TOKENS)
    val_dataset = create_dataset(val_codes, val_labels, tokenizer, MAX_TOKENS)
    test_dataset = create_dataset(test_codes, test_labels, tokenizer, MAX_TOKENS)
    
    trainer = train_model(model, tokenizer, train_dataset, val_dataset, DEVICE)
    
    test_results = evaluate_model(trainer, test_dataset)
    
    # print("\nSaving final model")
    trainer.save_model("./qwen_vulnerability_classifier/final_model")
    tokenizer.save_pretrained("./qwen_vulnerability_classifier/final_model")
    
    return trainer, test_results


if __name__ == "__main__":
    DATA_FILE = "./ds.jsonl"
    trainer, results = main(DATA_FILE)
