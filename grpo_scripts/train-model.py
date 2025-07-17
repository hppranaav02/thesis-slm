# import os
# import json
# from datasets import load_dataset
# from transformers import (
#     AutoTokenizer,
#     AutoModelForSeq2SeqLM,
#     Seq2SeqTrainingArguments,
#     Seq2SeqTrainer,
# )
# from data_model_2 import ResponseFormatMulti, instruction_multi as instruction

# # # ------------------------------ Configuration ------------------------------
# MODEL_NAME = "Salesforce/codet5p-220m"
# # # DATA_PATH = "./check_for_2015.jsonl"     # each record: {"input":<str>,"output":[...]}
# OUTPUT_DIR = "/local/s3905020/output/codet5p-go-cwe"
# # TRAIN_PATH = "/local/s3905020/dataset-creation/train.jsonl"
# # TEST_PATH  = "/local/s3905020/dataset-creation/test.jsonl"

# # # --------------------------- Load and split data ---------------------------
# # dataset = load_dataset("json", data_files=DATA_PATH, split="train")
# # dataset = dataset.train_test_split(test_size=0.2)
# dataset = load_dataset("json", data_files={"train": TRAIN_PATH,
#                                            "test":  TEST_PATH})
# print(
#     f"Dataset loaded with {len(dataset['train'])} training examples and "
#     f"{len(dataset['test'])} test examples."
# )

# # # ----------------------------- Tokenizer -----------------------------------
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# print("Tokenizer loaded successfully.")

# # ------------------------- Pre-processing fn -------------------------------
# def preprocess(examples):
#     # Build the conditioning prompt
#     inputs = [
#         f"instruction:\n{instruction}\ninput:\n{code}"
#         for code in examples["input"]
#     ]
#     model_inputs = tokenizer(
#         inputs, truncation=True, max_length=512, padding="max_length"
#     )

#     # Build target strings in ResponseFormatMulti JSON
#     with tokenizer.as_target_tokenizer():
#         targets = []
#         for cwe_list in examples["output"]:
#             # Ensure cwe_list is a list; secure examples should be []
#             if cwe_list is None:
#                 cwe_list = []
#             resp = ResponseFormatMulti(
#                 type="json",
#                 vulnerability=bool(cwe_list),
#                 vulnerability_type=cwe_list,
#                 reasoning=None,
#                 source=None,
#             )
#             targets.append(f"```json\n{resp.json()}\n```")

#         label_tokens = tokenizer(
#             targets, truncation=True, max_length=128, padding="max_length"
#         )

#     model_inputs["labels"] = label_tokens["input_ids"]
#     return model_inputs


# tokenized = dataset.map(preprocess, batched=True, remove_columns=dataset["train"].column_names)
# print("Dataset tokenized successfully.")

# # ------------------------------ Model --------------------------------------
# model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
# print("Model loaded successfully.")

# # ----------------------- Training arguments --------------------------------
# training_args = Seq2SeqTrainingArguments(
#     output_dir=OUTPUT_DIR,
#     eval_strategy="epoch",
#     save_strategy="epoch",
#     learning_rate=5e-5,
#     per_device_train_batch_size=2,
#     per_device_eval_batch_size=2,
#     num_train_epochs=4,
#     weight_decay=0.01,
#     predict_with_generate=True,
#     fp16=True,  # works fine on consumer GPUs with 16-bit support
#     logging_dir=os.path.join(OUTPUT_DIR, "logs"),
#     logging_steps=50,
#     logging_strategy="steps",
#     save_total_limit=2,
#     report_to=["wandb"],
# )
# print("Training arguments defined successfully.")

# # ------------------------------ Trainer ------------------------------------
# trainer = Seq2SeqTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized["train"],
#     eval_dataset=tokenized["test"],
#     tokenizer=tokenizer,
# )
# print("Trainer initialized successfully.")

# # ------------------------------- Train! ------------------------------------
# trainer.train()
# print("Training complete.")

# tokenizer.save_pretrained(OUTPUT_DIR)
# print("Tokenizer saved to output directory.")

# # Ensure tokenizer.model is present in the output directory
# from copy_tokenizer_model import copy_tokenizer_model
# copy_tokenizer_model(tokenizer, OUTPUT_DIR, MODEL_NAME)
# print("Checked and copied tokenizer.model if needed.")


import os
import json
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,    
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from data_model_2 import ResponseFormatMulti, instruction_multi as INSTRUCTION

def preprocess_fn(examples, tokenizer, max_input_len=512, max_target_len=64):
    inputs = [f"instruction:\n{INSTRUCTION}\ninput:\n{code}"
              for code in examples["input"]]
    model_inputs = tokenizer(
        inputs, truncation=True, padding="max_length", max_length=max_input_len
    )

    with tokenizer.as_target_tokenizer():
        targets = []
        for cwe_list in examples["output"]:
            # secure => empty list
            cwes = cwe_list or []
            resp = ResponseFormatMulti(
                type="json",
                vulnerability=bool(cwes),
                vulnerability_type=cwes,
                reasoning=None,
                source=None,
            )
            targets.append(f"```json\n{resp.json()}\n```")

        labels = tokenizer(
            targets, truncation=True, padding="max_length", max_length=max_target_len
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def train_model(
    model_name: str,
    train_path: str,
    test_path: str,
    output_dir: str,
    batch_size: int = 4,
    num_epochs: int = 4,
    lr: float = 3e-4,
):
    # 1) Load dataset
    ds = load_dataset("json", data_files={"train": train_path, "test": test_path})
    # 2) Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # 3) Pre-tokenize
    tokenized = ds.map(
        lambda ex: preprocess_fn(ex, tokenizer),
        batched=True,
        remove_columns=["input", "output"],
    )

    # 4) Quantize & load base model in 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )

    model.gradient_checkpointing_enable()            # enable gradient checkpointing
    model.config.use_cache = False                   # disable cache for training
    model = prepare_model_for_kbit_training(model)  # prepare model for k-bit training

    # 5) Attach LoRA adapters
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_2_SEQ_LM"
    )
    model = get_peft_model(model, lora_cfg)

    # 6) Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        fp16=True,
        gradient_checkpointing=True,
        logging_steps=20,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=2,
    )

    # 7) Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
    )

    # 8) Train!
    trainer.train()
    # 9) Save LoRA adapters + tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Training complete. Artifacts saved to {output_dir}")

if __name__ == "__main__":
    train_model(
        model_name="Salesforce/codegen-350M-mono",
        train_path="/local/s3905020/dataset-creation/train.jsonl",
        test_path="/local/s3905020/dataset-creation/test.jsonl",
        output_dir="/local/s3905020/output/codet5p-go-cwe",
    )
