import os
import json
import argparse
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
from sklearn.metrics import classification_report, accuracy_score
from datetime import datetime

def preprocess_dataset(dataset, tokenizer):
    def preprocess(examples):
        inputs = [f"{i} {c}" for i, c in zip(examples["instruction"], examples["input"])]
        model_inputs = tokenizer(inputs, truncation=True, padding="max_length", max_length=512)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["output"], truncation=True, padding="max_length", max_length=64)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    return dataset.map(preprocess, batched=True)

def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset and optionally subsample
    dataset = load_dataset("json", data_files=args.data_file, split="train")
    if args.data_fraction < 1.0:
        dataset = dataset.train_test_split(test_size=(1 - args.data_fraction), seed=42)["train"]
    dataset = dataset.train_test_split(test_size=0.2)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    tokenized = preprocess_dataset(dataset, tokenizer)

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(args.output_dir, "model"),
        evaluation_strategy="epoch",
        save_strategy="no",
        learning_rate=5e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        predict_with_generate=True,
        fp16=args.fp16,
        logging_dir=os.path.join(args.output_dir, "logs"),
        report_to="none"
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)
    )

    trainer.train()

    # Evaluate
    raw_preds = trainer.predict(tokenized["test"])
    preds = tokenizer.batch_decode(raw_preds.predictions, skip_special_tokens=True)
    labels = [x["output"].strip() for x in dataset["test"]]
    preds = [p.strip() for p in preds]

    # Save results
    accuracy = accuracy_score(labels, preds)
    report = classification_report(labels, preds, output_dict=True, zero_division=0)

    save_json(vars(args), os.path.join(args.output_dir, "config.json"))
    save_json({
        "timestamp": str(datetime.utcnow()),
        "accuracy": accuracy,
        "classification_report": report
    }, os.path.join(args.output_dir, "metrics.json"))

    print(f"✅ Done. Accuracy: {accuracy:.4f}. Results saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--data_file", type=str, default="data/check_for_2015.jsonl")
    parser.add_argument("--data_fraction", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--output_dir", type=str, required=True)
    # reasoning
    parser.add_argument("--reasoning", action="store_true", help="Include reasoning in the output format")
    # token length
    parser.add_argument("--max_length", type=int, default=512, help="Maximum length for input tokens")
    args = parser.parse_args()
    if not os.path.isfile(args.data_file):
        print("invalid data file path")
    elif not os.path.exists(args.output_dir):
        print("Invalid output directory")
    else:
        main(args)
