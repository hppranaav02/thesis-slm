
"""
Line Localization Metrics & Visualization Helpers
"""

from typing import List, Dict, Any, Iterable, Tuple
import ast
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, Counter


def _ensure_list(x):
    if isinstance(x, list):
        return [int(v) for v in x]
    if isinstance(x, (np.integer, int, float)) and not math.isnan(x):
        return [int(x)]
    if isinstance(x, str):
        s = x.strip()
        try:
            val = ast.literal_eval(s)
            if isinstance(val, (list, tuple)):
                return [int(v) for v in val]
            return [int(val)]
        except Exception:
            # fallback: split by comma
            parts = [p for p in s.replace('[','').replace(']','').split(',') if p.strip()!='']
            return [int(p) for p in parts] if parts else []
    return []


def _detect_indexing(true_lines: List[int], pred_line: int) -> int:
    if 0 in true_lines or pred_line == 0:
        return 0
    if len(true_lines) > 0 and min(true_lines) >= 1 and pred_line >= 1:
        return 1
    return 0


def nearest_distance(pred: int, truths: Iterable[int]) -> int:
    truths = list(truths)
    if len(truths) == 0:
        return np.nan
    return int(min(abs(pred - t) for t in truths))


def hit_within_window(pred: int, truths: Iterable[int], w: int) -> bool:
    """Is the prediction within ±w lines of any ground-truth line?"""
    return any(abs(pred - t) <= w for t in truths)


def success_at_k(ranked: List[int], truths: Iterable[int], k: int) -> bool:
    """Hit@k for ranked predictions."""
    truths = set(int(t) for t in truths)
    return any(int(l) in truths for l in ranked[:k])


def reciprocal_rank(ranked: List[int], truths: Iterable[int]) -> float:
    """Reciprocal rank for ranked predictions."""
    truths = set(int(t) for t in truths)
    for i, l in enumerate(ranked, start=1):
        if int(l) in truths:
            return 1.0 / i
    return 0.0


def compute_localization_metrics(
    df: pd.DataFrame,
    pred_col: str = "pred_line",
    true_col: str = "true_lines",
    rank_col: str = "pred_ranked_lines",
    cwe_col: str = "cwe",
    window_set = (0, 1, 2, 3, 5, 10),
) -> Dict[str, Any]:

    if true_col not in df.columns or pred_col not in df.columns:
        raise ValueError(f"DataFrame must have columns '{true_col}' and '{pred_col}'. Found: {list(df.columns)}")


    dd = df.copy()
    dd[true_col] = dd[true_col].apply(_ensure_list)


    sample = dd.dropna(subset=[pred_col, true_col]).iloc[0] if len(dd) else None
    offset = 0
    if sample is not None:
        offset = _detect_indexing(sample[true_col], int(sample[pred_col]))
    dd[true_col] = dd[true_col].apply(lambda L: [int(v) - offset for v in L])
    dd[pred_col] = dd[pred_col].apply(lambda v: int(v) - offset if pd.notnull(v) else v)

    if rank_col in dd.columns:
        dd[rank_col] = dd[rank_col].apply(_ensure_list)
        dd[rank_col] = dd[rank_col].apply(lambda L: [int(v) - offset for v in L])

    dd["distance"] = dd.apply(lambda r: nearest_distance(int(r[pred_col]), r[true_col]) if pd.notnull(r[pred_col]) else np.nan, axis=1)
    for w in window_set:
        dd[f"acc_within_±{w}"] = dd.apply(lambda r: hit_within_window(int(r[pred_col]), r[true_col], w) if pd.notnull(r[pred_col]) else False, axis=1)

    hit_at_k = None
    mrr = None
    if rank_col in dd.columns:
        ks = [1, 3, 5, 10]
        data = {}
        for k in ks:
            data[f"Hit@{k}"] = dd.apply(lambda r: success_at_k(r[rank_col], r[true_col], k) if isinstance(r[rank_col], list) else False, axis=1).mean()
        hit_at_k = pd.DataFrame([data])
        mrr = dd.apply(lambda r: reciprocal_rank(r[rank_col], r[true_col]) if isinstance(r[rank_col], list) else 0.0, axis=1).mean()

    acc_cols = [c for c in dd.columns if c.startswith("acc_within_")]
    summary = {
        "n_samples": len(dd),
        "exact_match_acc": dd["acc_within_±0"].mean(),
        "mean_abs_distance": dd["distance"].dropna().mean(),
        "median_abs_distance": dd["distance"].dropna().median(),
    }
    summary.update({c: dd[c].mean() for c in acc_cols})
    summary_df = pd.DataFrame([summary])
    if cwe_col in dd.columns:
        by_cwe = dd.groupby(cwe_col).agg(
            n=("distance", "count"),
            mean_abs_distance=("distance", "mean"),
            median_abs_distance=("distance", "median"),
            exact_match_acc=("acc_within_±0", "mean"),
            acc_within_1=("acc_within_±1", "mean"),
            acc_within_3=("acc_within_3", "mean"),
            acc_within_5=("acc_within_5", "mean"),
            acc_within_10=("acc_within_10", "mean"),
        ).reset_index()
    else:
        by_cwe = None

    return {
        "per_sample": dd,
        "summary": summary_df,
        "by_cwe": by_cwe,
        "hit_at_k": hit_at_k,
        "mrr": mrr,
    }


def save_tables(results: Dict[str, Any], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    per_sample = results["per_sample"]
    summary = results["summary"]
    by_cwe = results["by_cwe"]
    hit_at_k = results["hit_at_k"]
    mrr = results["mrr"]

    per_sample.to_csv(os.path.join(out_dir, "line_loc_per_sample.csv"), index=False)
    summary.to_csv(os.path.join(out_dir, "line_loc_summary.csv"), index=False)
    if by_cwe is not None:
        by_cwe.to_csv(os.path.join(out_dir, "line_loc_by_cwe.csv"), index=False)
    if hit_at_k is not None:
        hit_at_k.to_csv(os.path.join(out_dir, "line_loc_hit_at_k.csv"), index=False)
    with open(os.path.join(out_dir, "line_loc_summary.tex"), "w") as f:
        f.write(summary.to_latex(index=False, float_format="%.4f"))
    if by_cwe is not None:
        with open(os.path.join(out_dir, "line_loc_by_cwe.tex"), "w") as f:
            f.write(by_cwe.to_latex(index=False, float_format="%.4f"))
    if hit_at_k is not None:
        with open(os.path.join(out_dir, "line_loc_hit_at_k.tex"), "w") as f:
            f.write(hit_at_k.to_latex(index=False, float_format="%.4f"))
    if mrr is not None:
        with open(os.path.join(out_dir, "line_loc_mrr.txt"), "w") as f:
            f.write(f"MRR: {mrr:.4f}\n")


def plot_distance_hist(per_sample: pd.DataFrame, path: str):
    import matplotlib.pyplot as plt
    vals = per_sample["distance"].dropna().values
    plt.figure()
    plt.hist(vals, bins=30)
    plt.title("Line Localization: Absolute Distance Histogram")
    plt.xlabel("Absolute distance (lines)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_distance_cdf(per_sample: pd.DataFrame, path: str):
    import matplotlib.pyplot as plt
    vals = np.sort(per_sample["distance"].dropna().values)
    y = np.arange(1, len(vals)+1) / len(vals)
    plt.figure()
    plt.plot(vals, y, drawstyle="steps-post")
    plt.title("Line Localization: CDF of Absolute Distance")
    plt.xlabel("Absolute distance (lines)")
    plt.ylabel("Cumulative fraction")
    plt.grid(True, which="both", axis="both", linewidth=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_window_accuracy(per_sample: pd.DataFrame, path: str):
    import matplotlib.pyplot as plt
    acc_cols = [c for c in per_sample.columns if c.startswith("acc_within_")]
    ks = []
    vals = []
    for c in sorted(acc_cols, key=lambda x: int(x.split("±")[1])):
        ks.append(c.replace("acc_within_", ""))
        vals.append(per_sample[c].mean())
    plt.figure()
    plt.bar(ks, vals)
    plt.title("Accuracy within ±k Lines")
    plt.xlabel("Window k")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_by_cwe_bar(by_cwe: pd.DataFrame, metric: str, path: str, top_n: int = 20, sort_ascending: bool = True):
    import matplotlib.pyplot as plt
    df = by_cwe.sort_values(metric, ascending=sort_ascending).head(top_n)
    plt.figure()
    plt.bar(df.iloc[:,0].astype(str), df[metric].values)
    plt.title(f"CWE vs {metric}")
    plt.xlabel("CWE")
    plt.ylabel(metric)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_filelen_vs_error(per_sample: pd.DataFrame, path: str):
    import matplotlib.pyplot as plt
    if "n_lines" not in per_sample.columns:
        return
    xs = per_sample["n_lines"].values
    ys = per_sample["distance"].values
    plt.figure()
    plt.scatter(xs, ys, s=8)
    plt.title("File Length vs Localization Error")
    plt.xlabel("File length (lines)")
    plt.ylabel("Abs distance (lines)")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_hit_at_k(hit_at_k: "pd.DataFrame|None", mrr: "float|None", path: str):
    if hit_at_k is None:
        return
    import matplotlib.pyplot as plt
    row = hit_at_k.iloc[0].to_dict()
    labels = list(row.keys())
    vals = [row[k] for k in labels]
    plt.figure()
    plt.bar(labels, vals)
    plt.title("Ranked Line Localization (Hit@k)")
    plt.xlabel("k")
    plt.ylabel("Hit rate")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
