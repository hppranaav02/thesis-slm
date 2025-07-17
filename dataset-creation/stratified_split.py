# stratified_split.py ---------------------------------------------------------
import json, random
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any, Tuple, Set

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]

def save_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))

def stratified_multilabel_split(
    pairs: List[Tuple[Dict, Dict]],
    train_ratio: float = 0.7,
    seed: int = 42,
) -> Tuple[List[Tuple[Dict, Dict]], List[Tuple[Dict, Dict]]]:
    """
    Greedy algorithm that keeps the vulnerable+secure pair together and
    tries to match each CWE's global train_ratio.
    """
    rnd = random.Random(seed)
    rnd.shuffle(pairs)

    total = Counter()
    for vuln, _ in pairs:
        total.update(vuln["output"])

    target_train = {cwe: int(round(cnt * train_ratio)) for cwe, cnt in total.items()}
    tally_train = Counter()

    train, test = [], []
    for vuln, secure in pairs:
        cwes: Set[str] = set(vuln["output"])

        # How many of these CWEs are still missing from train?
        deficit = sum(max(0, target_train[c] - tally_train[c]) for c in cwes)
        surplus = sum(max(0, tally_train[c] + 1 - target_train[c]) for c in cwes)

        choose_train = deficit >= surplus
        if choose_train:
            train.append((vuln, secure))
            tally_train.update(cwes)
        else:
            test.append((vuln, secure))

    return train, test
