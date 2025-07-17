import os
import json
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments
)
from sklearn.metrics import classification_report, accuracy_score
from data_model import ResponseFormat, instruction

# --- Config ---
MODEL_DIR = "../models/codet5p-go-cwe/checkpoint-4832"
DATA_FILE = "../data/check_for_2015.jsonl"
OUTPUT_REPORT = "../outputs/eval_report.json"

# --- Load model + tokenizer ---
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

# --- Load and split dataset ---
dataset = load_dataset("json", data_files=DATA_FILE, split="train")
dataset = dataset.train_test_split(test_size=0.2)
test_data = dataset["test"]

# --- Preprocess function ---
def preprocess(examples):
    inputs = [f"instruction:\n{i}\ninput:{c}" for i, c in zip([instruction]*len(examples["input"]), examples["input"])]
    model_inputs = tokenizer(inputs, truncation=True, max_length=512, padding="max_length")
    print("Inputs tokenized successfully.")

    # Pad the tokens to maximum length

    #ask what this does
    with tokenizer.as_target_tokenizer():
        outputs= []
        for output in zip(examples["output"]):
            print(f"Processing output: {output[0]}")
            vul, vul_type = output[0].split("=") if output[0].lower() != 'secure' else (output[0], None)
            vul = vul.strip()
            vul_type = vul_type.strip() if vul_type else None
            response = ResponseFormat(
                type="json",
                vulnerability=(vul.lower() == "vulnerable"),
                vulnerability_type=vul_type,
            )
            # examples["output"] = f"```json\n{response.json()}\n```"
            outputs.append((f"```json\n{response.json()}\n```"))
        examples["output"] = outputs

        labels = tokenizer(examples["output"], truncation=True, max_length=128, padding="max_length") # try 128,

    model_inputs["labels"] = labels["input_ids"]
    print("Labels tokenized successfully.")
    return model_inputs

tokenized_test = test_data.map(preprocess, batched=True)
collator = DataCollatorForSeq2Seq(tokenizer, model=model)
training_args = Seq2SeqTrainingArguments(
    output_dir="../outputs/test-run",
    per_device_eval_batch_size=4,
    predict_with_generate=True,
    do_train=False,
    do_eval=False,
    logging_dir="../outputs/logs"
)

# --- Inference ---
trainer = Seq2SeqTrainer(model=model, tokenizer=tokenizer,args=training_args, data_collator=collator)
raw_preds = trainer.predict(tokenized_test)
# --- Decode predictions ---
pred_tokens = tokenizer.batch_decode(raw_preds.predictions, skip_special_tokens=True)
true_labels = [ResponseFormat.parse_raw(x["output"]).vulnerability for x in test_data]
pred_labels = [
    ResponseFormat.parse_raw(x).vulnerability if x.startswith("```json") else None
    for x in pred_tokens
]

# --- Metrics ---
accuracy = accuracy_score(true_labels, pred_labels)
report = classification_report(true_labels, pred_labels, output_dict=True, zero_division=0)

# --- Save full report to JSON ---
os.makedirs("outputs", exist_ok=True)
with open(OUTPUT_REPORT, "w") as f:
    json.dump({
        "accuracy": accuracy,
        "classification_report": report
    }, f, indent=2)

# --- Print key metrics ---
print(f"Accuracy: {accuracy:.4f}")
print("Top-level F1 scores:")
for label, scores in report.items():
    if isinstance(scores, dict) and "f1-score" in scores:
        print(f"{label:>30}: F1 = {scores['f1-score']:.4f}")
