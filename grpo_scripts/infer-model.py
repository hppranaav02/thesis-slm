#!/usr/bin/env python3
import argparse, json, sys, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


CHECKPOINT = "/local/s3905020/output/codet5p-go-cwe/checkpoint-4656"   # where Trainer saved model
DEVICE     = "cuda:0" if torch.cuda.is_available() else "cpu"
MAX_INPUT  = 512
MAX_NEWTOK = 128

# The same instruction string you used in training
from data_model_2 import instruction_multi as instruction

def load_text(path: str | None) -> str:
    if path:
        return Path(path).read_text(encoding="utf-8")
    return sys.stdin.read()

def build_prompt(code: str) -> str:
    return f"instruction:\n{instruction}\ninput:\n{code}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", nargs="?", help="Go source/diff file; stdin if omitted")
    args = parser.parse_args()

    code  = load_text(args.file)
    prompt = build_prompt(code)

    tok   = AutoTokenizer.from_pretrained(CHECKPOINT)
    model = AutoModelForSeq2SeqLM.from_pretrained(CHECKPOINT).to(DEVICE)

    tokens = tok(prompt, return_tensors="pt", truncation=True,
                 max_length=MAX_INPUT).to(DEVICE)

    with torch.no_grad():
        out_ids = model.generate(
            **tokens,
            max_new_tokens=MAX_NEWTOK,
            do_sample=False,        # change to True + temperature for diversity
        )

    result_str = tok.decode(out_ids[0], skip_special_tokens=True)
    result_str = result_str.strip()
    if result_str.startswith("```json"):
        result_str = result_str.split("```json", 1)[1].rsplit("```", 1)[0]

    try:
        data = json.loads(result_str)
    except json.JSONDecodeError:
        print("Model output is not valid JSON:\n", result_str)
        return

    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
