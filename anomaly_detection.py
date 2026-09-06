"""
anomaly_detection.py

Day 7 build — trains Isolation Forest on the Day 6 wallet features, purely
unsupervised (it never sees involved_in_illicit_pattern during training),
then evaluates the resulting anomaly scores against that ground truth
AFTERWARD to get real precision/recall numbers.

Key design choices, worth understanding:

1. LOG-TRANSFORM SKEWED FEATURES FIRST (the fix flagged on Day 6).
   tx_frequency_per_day and the geo-velocity features have extreme outlier
   ranges (max ~1,000,000 vs mean ~1,500) purely from division-by-small-
   number artifacts, not real signal magnitude. Left untransformed, Isolation
   Forest's random split points would be dominated by that raw scale rather
   than genuine distributional shape. We use a SIGNED log1p transform
   (sign(x) * log1p(abs(x))) so it works on features that can be negative
   (the z-score features) too.

2. NO contamination-rate leakage into evaluation.
   We deliberately do NOT tune a single contamination threshold against the
   known ~20% illicit-touched rate and call that "the" result -- knowing the
   true contamination rate in advance is a luxury a real deployment won't
   have. Instead we use continuous anomaly scores and report a full
   precision-recall curve (+ AUC-PR), plus precision/recall at a few
   concrete "review the top-K%" operating points, which is how an analyst
   would actually use this (a ranked queue, not a hard yes/no cutoff).

3. Per-pattern-type breakdown.
   Reporting one blended recall number hides which patterns this feature
   set can and can't see. CoinJoin transactions especially are NOT expected
   to score highly here -- their signature (many similar-value inputs/
   outputs in one tx) isn't captured by any of these per-wallet features.
   That's an honest, useful finding for the write-up, not a failure to hide.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

FEATURE_COLS = [
    "avg_amount_zscore", "max_abs_amount_zscore", "tx_count",
    "tx_frequency_per_day", "fan_in_degree", "fan_out_degree",
    "max_geo_velocity_kmh", "avg_geo_velocity_kmh", "time_of_day_entropy",
]
SKEWED_COLS = [
    "avg_amount_zscore", "max_abs_amount_zscore",
    "tx_frequency_per_day", "max_geo_velocity_kmh", "avg_geo_velocity_kmh",
]


def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


def load_and_transform(path: str = "wallet_features.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in SKEWED_COLS:
        df[col + "_log"] = signed_log1p(df[col])
    return df


def get_model_feature_cols():
    return [c + "_log" if c in SKEWED_COLS else c for c in FEATURE_COLS]


def train_isolation_forest(df: pd.DataFrame, contamination="auto", random_state=42):
    model_cols = get_model_feature_cols()
    X = df[model_cols].values
    model = IsolationForest(
        n_estimators=200,
        max_samples="auto",
        contamination=contamination,  # only affects .predict()'s hard cutoff, NOT the scores we evaluate
        random_state=random_state,
    )
    model.fit(X)
    # Higher = more anomalous (flip sklearn's convention, which is lower = more abnormal)
    anomaly_score = -model.score_samples(X)
    return model, anomaly_score


def precision_recall_at_k(y_true, scores, k_fracs=(0.10, 0.20, 0.30)):
    n = len(scores)
    order = np.argsort(-scores)  # most anomalous first
    results = {}
    for k in k_fracs:
        top_n = max(1, int(round(n * k)))
        flagged = set(order[:top_n])
        tp = sum(1 for i in flagged if y_true[i] == 1)
        precision = tp / top_n
        recall = tp / max(y_true.sum(), 1)
        results[k] = {"precision": precision, "recall": recall, "n_flagged": top_n, "n_true_positive": tp}
    return results


def per_pattern_recall(df: pd.DataFrame, scores, k_fracs=(0.10, 0.20, 0.30)):
    n = len(scores)
    order = np.argsort(-scores)
    pattern_lists = df["pattern_types_involved"].fillna("").tolist()
    all_patterns = set()
    for p in pattern_lists:
        all_patterns.update(p.split(",")) if p else None
    all_patterns.discard("")

    report = {}
    for pattern in sorted(all_patterns):
        pattern_wallet_idx = {i for i, p in enumerate(pattern_lists) if pattern in p.split(",")}
        if not pattern_wallet_idx:
            continue
        row = {}
        for k in k_fracs:
            top_n = max(1, int(round(n * k)))
            flagged = set(order[:top_n])
            hits = len(pattern_wallet_idx & flagged)
            row[k] = hits / len(pattern_wallet_idx)
        report[pattern] = {"n_wallets": len(pattern_wallet_idx), "recall_at_k": row}
    return report


if __name__ == "__main__":
    df = load_and_transform("wallet_features.csv")
    model, anomaly_score = train_isolation_forest(df)
    df["anomaly_score"] = anomaly_score

    y_true = df["involved_in_illicit_pattern"].astype(int).values

    ap = average_precision_score(y_true, anomaly_score)
    auc = roc_auc_score(y_true, anomaly_score)
    print("=== Overall ranking quality (threshold-independent) ===")
    print(f"AUC-PR (average precision): {ap:.3f}")
    print(f"AUC-ROC:                    {auc:.3f}")
    print(f"Baseline (random) AUC-PR would be ~{y_true.mean():.3f} (the base illicit rate)")
    print()

    print("=== Precision/Recall at 'review top-K% most anomalous' operating points ===")
    pr_at_k = precision_recall_at_k(y_true, anomaly_score)
    for k, r in pr_at_k.items():
        print(f"Top {int(k*100)}%  ({r['n_flagged']} wallets flagged): "
              f"precision={r['precision']:.3f}  recall={r['recall']:.3f}  "
              f"({r['n_true_positive']} true positives caught)")
    print()

    print("=== Recall by pattern type (does this feature set see peeling/fanout as well as CoinJoin?) ===")
    pattern_report = per_pattern_recall(df, anomaly_score)
    for pattern, info in pattern_report.items():
        recalls = ", ".join(f"top{int(k*100)}%={v:.2f}" for k, v in info["recall_at_k"].items())
        print(f"{pattern:10s} (n={info['n_wallets']:3d}): {recalls}")

    df.to_csv("wallet_anomaly_scores.csv", index=False)
    print("\nSaved wallet_anomaly_scores.csv")
