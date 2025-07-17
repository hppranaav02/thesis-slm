#!/usr/bin/env python3
"""
pg2train.py
===========

Pull rows from PostgreSQL, group by file_change_id, and emit TWO examples
per change (vulnerable + secure) for your CWE-training pipeline.

CWE filtering
-------------
  • --min-count K   keep CWEs with frequency ≥ K   (default: 2)
  • --top N         keep the N most common CWEs    (overrides --min-count)

Required columns in query/table
-------------------------------
  file_change_id, cwe_id, code_before, code_after, diff (if using --use diff/hybrid)
"""

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, DefaultDict, Union

import psycopg2
from psycopg2.extras import DictCursor
from tqdm import tqdm

from stratified_split import stratified_multilabel_split, save_jsonl


# ──────────────────────────── DB Helpers ──────────────────────────────── #
def connect_pg(cfg) -> psycopg2.extensions.connection:
    return psycopg2.connect(
        host=cfg.host,
        port=cfg.port,
        dbname=cfg.dbname,
        user=cfg.user,
        password=cfg.password or os.getenv("PGPASSWORD"),
    )


def fetch_rows(cur, table: str, query_file: Path | None) -> List[Dict[str, Any]]:
    if query_file:
        sql_text = query_file.read_text(encoding="utf-8")
        cur.execute(sql_text)
    else:
        cur.execute(f"SELECT * FROM {table}")
    return cur.fetchall()  # DictCursor → list[dict]


# ─────────────────────────── File Helpers ─────────────────────────────── #
def write_jsonl(items: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def choose_code(row: Dict[str, Any], strategy: str, before: bool) -> str:
    if strategy == "code_before":
        return row["code_before"] if before else row["code_after"]
    if strategy == "diff":
        return row["diff"] if before else row["code_after"]
    if strategy == "hybrid":
        if before:
            return f"{row['code_before']}\n\n/* DIFF */\n{row['diff']}"
        return row["code_after"]
    raise ValueError(strategy)


# ───────────────────────────── CLI ─────────────────────────────────────── #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PostgreSQL → model-ready JSONL")

    # DB connection
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", default=5432, type=int)
    p.add_argument("--dbname", required=True)
    p.add_argument("--user", required=True)
    p.add_argument("--password", help="Optional (falls back to $PGPASSWORD)")

    # Data selection
    p.add_argument("--table", default="changes",
                   help="Table name for SELECT * (ignored if --query given)")
    p.add_argument("--query", type=Path,
                   help="Custom SQL SELECT file (overrides --table)")

    # Output
    p.add_argument("--train", required=True, type=Path,
                   help="Destination JSONL with training examples")
    p.add_argument("--raw", type=Path,
                   help="Optional raw dump of fetched rows")

    # Input snippet choice
    p.add_argument("--use", choices=["code_before", "diff", "hybrid"],
                   default="code_before",
                   help="Snippet to place in vulnerable 'input' field")
    p.add_argument("--train-out", default="train_fullfile.jsonl", type=Path,
               help="Where to write stratified TRAIN set (default: train_fullfile.jsonl)")
    p.add_argument("--test-out",  default="test_fullfile.jsonl",  type=Path,
                help="Where to write stratified TEST  set (default: test_fullfile.jsonl)")
    p.add_argument("--split",     default=0.8, type=float,
                help="Proportion of rows to place in TRAIN (default: 0.8)")

    # CWE filtering
    group = p.add_mutually_exclusive_group()
    group.add_argument("--min-count", type=int, default=2,
                       help="Keep CWEs with frequency ≥ this value (default: 2)")
    group.add_argument("--top", type=int,
                       help="Keep only the N most frequent CWEs (overrides --min-count)")

    return p.parse_args()


# ───────────────────────────── main ────────────────────────────────────── #
def main() -> None:
    cfg = parse_args()

    # 1. Fetch rows
    with connect_pg(cfg) as conn, conn.cursor(cursor_factory=DictCursor) as cur:
        rows = fetch_rows(cur, cfg.table, cfg.query)
    print(f"Fetched {len(rows)} rows.")

    # optional raw dump
    if cfg.raw:
        write_jsonl(rows, cfg.raw)
        print(f"Raw rows → {cfg.raw}")

    # 2. Build frequency map
    freq = Counter(r["cwe_id"] for r in rows if r.get("cwe_id"))

    # Select allowed CWEs
    if cfg.top is not None:
        most_common = {cwe for cwe, _ in freq.most_common(cfg.top)}
        allowed = most_common
        print(f"Keeping top {cfg.top} CWEs ({len(allowed)} unique).")
    else:
        allowed = {cwe for cwe, cnt in freq.items() if cnt >= cfg.min_count}
        print(f"Keeping CWEs with count ≥ {cfg.min_count} ({len(allowed)} unique).")

    # 3. Group rows by change
    grouped: DefaultDict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[r["file_change_id"]].append(r)

    # 4. Build training examples
    examples: List[Dict[str, Any]] = []
    dropped = 0
    for rows_same_change in tqdm(grouped.values(), desc="Building examples"):
        ref_row = rows_same_change[0]

        cwes = sorted({
            r["cwe_id"] for r in rows_same_change
            if r.get("cwe_id") and r["cwe_id"] in allowed
        })

        if not cwes:      # no allowed CWE left → skip whole pair
            dropped += 1
            continue

        # vulnerable
        examples.append({
            "input":  choose_code(ref_row, cfg.use, before=True),
            "output": cwes,
            "source": []
        })
        # secure
        examples.append({
            "input":  choose_code(ref_row, cfg.use, before=False),
            "output": [],
            "source": []
        })

    # 5. Write
    # write_jsonl(examples, cfg.train)
    # Pair vulnerable+secure rows (order is guaranteed)
    pairs = [(examples[i], examples[i + 1]) for i in range(0, len(examples), 2)]

    train_pairs, test_pairs = stratified_multilabel_split(
        pairs, train_ratio=cfg.split, seed=2025
    )

    train_rows = [r for pair in train_pairs for r in pair]
    test_rows  = [r for pair in test_pairs  for r in pair]

    save_jsonl(train_rows, cfg.train_out)
    save_jsonl(test_rows,  cfg.test_out)

    print(f"TRAIN rows : {len(train_rows)}   → {cfg.train_out}")
    print(f"TEST  rows : {len(test_rows)}   → {cfg.test_out}")
    print(f"Wrote {len(examples)} examples "
          f"(skipped {dropped} change(s) with no remaining CWE).")


if __name__ == "__main__":
    main()
