import os
import torch
import json
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
)
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from data_model_2 import ResponseFormat, ResponseFormatMulti

# Configuration
MODEL_NAME = "Salesforce/codet5p-220m"
TRAIN_PATH = "/local/s3905020/code/dataset-creation/train.jsonl"
TEST_PATH = "/local/s3905020/code/dataset-creation/test.jsonl"
OUTPUT_DIR = "/local/s3905020/output/codet5p-go-cwe"
BATCH_SIZE = 4
NUM_EPOCHS = 4
LR = 3e-5
GROUP_SIZE = 4  # number of samples per prompt for GRPO
EPSILON = 0.2  # PPO clipping parameter
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare policy model: quantized + LoRA
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
policy_model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_cfg,
    device_map="auto",
)
policy_model.gradient_checkpointing_enable()
policy_model.config.use_cache = False
policy_model = prepare_model_for_kbit_training(policy_model)

lora_cfg = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_2_SEQ_LM",
)
policy_model = get_peft_model(policy_model, lora_cfg)
policy_model.to(DEVICE)
policy_model.eval()

# Tokenizer for policy model
policy_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Data
raw_dataset = load_dataset("json", data_files={"train": TRAIN_PATH, "test": TEST_PATH})
train_dataset = raw_dataset["train"]
test_dataset = raw_dataset["test"]

# Define a function to preprocess the data for the trainer
def preprocess_function(examples):
    # Ensure prompt is truncated to model max length
    return policy_tokenizer(
        examples["prompt"],
        truncation=True,
        padding="max_length",
        max_length=512,
    )

# Add 'prompt' field for GRPOTrainer compatibility BEFORE tokenization
def add_prompt_field(example):
    example["prompt"] = example["input"]
    return example

train_dataset = train_dataset.map(add_prompt_field)
test_dataset = test_dataset.map(add_prompt_field)

train_dataset = train_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)


# GRPO config
grpo_config = GRPOConfig(
    # Essential parameters
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    num_generations=GROUP_SIZE,  # Number of completions to generate for each prompt
    per_device_train_batch_size=BATCH_SIZE,  # We want to get all generations in one device batch
    # Optional but useful
    gradient_accumulation_steps=2,
    learning_rate=LR,
    logging_steps=10,
    # GRPO specific (optional)
    use_vllm=False,  # Speed up generation
)

# Reward function
def extract_structured_response(completion, multi_label=True):
    try:
        data = json.loads(completion)
        if multi_label:
            parsed = ResponseFormatMulti(**data)
        else:
            parsed = ResponseFormat(**data)
        return parsed
    except Exception:
        return None

def custom_reward(completions, answers, multi_label=True, **kwargs):
    """
    completions: list of model outputs (str)
    answers: list of ground truth dicts (parsed from dataset)
    multi_label: whether to use the multi-label schema
    """
    rewards = []
    for completion, correct in zip(completions, answers):
        parsed = extract_structured_response(completion, multi_label=multi_label)
        # Reward for valid structure
        structure_reward = 0.5 if parsed is not None else 0.0
        # Reward for correct values
        value_reward = 0.0
        if parsed is not None:
            try:
                # Compare vulnerability
                vuln_match = int(parsed.vulnerability == correct["vulnerability"])
                # Compare vulnerability_type (single or multi)
                if multi_label:
                    gt_types = set(correct.get("vulnerability_type", []) or [])
                    pred_types = set(parsed.vulnerability_type or [])
                    type_match = int(gt_types == pred_types)
                else:
                    type_match = int(parsed.vulnerability_type == correct.get("vulnerability_type"))
                value_reward = 0.5 * (vuln_match + type_match)
            except Exception:
                value_reward = 0.0
        rewards.append(structure_reward + value_reward)
    return rewards



# Prepare ground truth answers from the dataset
# Assumes 'label' column contains the ground truth dict for each example
print("Preparing ground truth answers...")
print(f"{raw_dataset['train']}")
train_answers = [ex["output"] for ex in raw_dataset["train"]]
test_answers = [ex["output"] for ex in raw_dataset["test"]]


# Trainer
trainer = GRPOTrainer(
    model=policy_model,
    reward_funcs=custom_reward,
    args=grpo_config,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    processing_class=policy_tokenizer,
)

# Train
trainer.train()

# Save adapters & tokenizer
os.makedirs(OUTPUT_DIR, exist_ok=True)
policy_model.save_pretrained(OUTPUT_DIR)
policy_tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ Training complete. Model saved at {OUTPUT_DIR}")
