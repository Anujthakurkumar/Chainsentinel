"""
peeling_chain_classifier.py

Day 9 build — extracts chain-level features from every candidate chain Day 8's
graph traversal found (both true peeling chains AND coincidental normal-traffic
look-alikes), then trains a Random Forest to score "peeling likelihood" instead
of relying on a single hard rule (e.g. "hop_count >= 5").

HONEST CAVEAT, worth stating in your write-up rather than glossing over:
this dataset only has 4 true peeling chains vs 6 false-positive candidates.
That's too few for a real train/test split, so we use Leave-One-Out cross-
validation (train on 9, test on the 1 left out, repeat for all 10) to get an
honest generalization estimate instead of a number that's really just
"memorized the training set." It's also worth noting plainly that in THIS
dataset, hop_count and timing-gap tightness alone already separate the
classes almost perfectly (true chains: 8-9 hops, ~50-300s gaps; false
positives: 2 hops, hours-long gaps) -- a simple rule could nearly solve this
toy case. The value of training a real classifier here is (a) it generalizes
better to messier real-world cases where the separation won't be this clean,
and (b) it gives you feature-importance-based explainability instead of an
arbitrary hardcoded threshold.
"""

import pickle
from datetime import datetime

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from peeling_chain_detection import find_qualifying_2out_txs, detect_peeling_chains, load_graph

FEATURE_NAMES = [
    "hop_count", "mean_time_gap_sec", "std_time_gap_sec",
    "min_time_gap_sec", "max_time_gap_sec",
    "mean_ratio", "std_ratio", "decline_fraction",
    "log_start_amount", "mean_fee_fraction",
]


def extract_chain_features(W, qualifying: dict, chain: list) -> dict:
    timestamps = [datetime.fromisoformat(qualifying[tx]["timestamp"]) for tx in chain]
    gaps = [(timestamps[j + 1] - timestamps[j]).total_seconds() for j in range(len(timestamps) - 1)]
    ratios = [qualifying[tx]["ratio"] for tx in chain]

    start_amount = qualifying[chain[0]]["small"][1] + qualifying[chain[0]]["large"][1]
    end_amount = qualifying[chain[-1]]["large"][1]
    decline_fraction = (start_amount - end_amount) / start_amount if start_amount > 0 else 0.0

    fee_fractions = []
    for tx in chain:
        total_out = qualifying[tx]["small"][1] + qualifying[tx]["large"][1]
        fee = W.nodes[tx].get("fee", 0.0)
        total_in = total_out + fee
        if total_in > 0:
            fee_fractions.append(fee / total_in)

    return {
        "hop_count": len(chain),
        "mean_time_gap_sec": float(np.mean(gaps)) if gaps else 0.0,
        "std_time_gap_sec": float(np.std(gaps)) if gaps else 0.0,
        "min_time_gap_sec": float(np.min(gaps)) if gaps else 0.0,
        "max_time_gap_sec": float(np.max(gaps)) if gaps else 0.0,
        "mean_ratio": float(np.mean(ratios)),
        "std_ratio": float(np.std(ratios)),
        "decline_fraction": decline_fraction,
        "log_start_amount": float(np.log1p(start_amount)),
        "mean_fee_fraction": float(np.mean(fee_fractions)) if fee_fractions else 0.0,
    }


def build_dataset(W, chains: list):
    qualifying = find_qualifying_2out_txs(W)
    X, y = [], []
    for chain in chains:
        feats = extract_chain_features(W, qualifying, chain)
        X.append([feats[name] for name in FEATURE_NAMES])
        chain_ids = {W.nodes[tx].get("chain_id") for tx in chain if W.nodes[tx].get("chain_id")}
        y.append(1 if chain_ids else 0)
    return np.array(X), np.array(y)


def evaluate_with_loo(X, y):
    """Leave-One-Out cross-validation -- the only honest option with this few samples."""
    loo = LeaveOneOut()
    preds = np.zeros_like(y)
    for train_idx, test_idx in loo.split(X):
        clf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
        clf.fit(X[train_idx], y[train_idx])
        preds[test_idx] = clf.predict(X[test_idx])
    return preds


def evaluate_with_holdout(X, y, test_size=0.3, random_state=7):
    """A genuine held-out train/test split -- the model NEVER sees the test
    examples during training. This is stronger evidence of real generalization
    than LOO alone, and is only meaningful now that we have enough samples
    (including real hard negatives) to make a stratified split sensible."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    clf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    return y_test, preds, (len(X_train), len(X_test))


if __name__ == "__main__":
    W = load_graph("btc_graph_wallet_only.gpickle")
    chains = detect_peeling_chains(W, min_hops=2)
    X, y = build_dataset(W, chains)

    print(f"Candidate chains: {len(chains)}  (true peeling: {y.sum()}, false-positive look-alikes: {len(y)-y.sum()})")
    print()
    print("HONEST CAVEAT: only", int(y.sum()), "true positive examples exist in this dataset.")
    print("Using Leave-One-Out CV below because a real train/test split isn't meaningful at this sample size.")
    print()

    loo_preds = evaluate_with_loo(X, y)
    precision = precision_score(y, loo_preds, zero_division=0)
    recall = recall_score(y, loo_preds, zero_division=0)
    f1 = f1_score(y, loo_preds, zero_division=0)
    cm = confusion_matrix(y, loo_preds)

    print("=== Leave-One-Out cross-validated performance ===")
    print(f"Precision: {precision:.3f}  Recall: {recall:.3f}  F1: {f1:.3f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(cm)
    print()

    if len(y) >= 15 and y.sum() >= 5 and (len(y) - y.sum()) >= 5:
        y_test, holdout_preds, (n_train, n_test) = evaluate_with_holdout(X, y)
        h_precision = precision_score(y_test, holdout_preds, zero_division=0)
        h_recall = recall_score(y_test, holdout_preds, zero_division=0)
        h_f1 = f1_score(y_test, holdout_preds, zero_division=0)
        h_cm = confusion_matrix(y_test, holdout_preds)
        print(f"=== Held-out train/test split ({n_train} train / {n_test} test, model never saw test set) ===")
        print(f"Precision: {h_precision:.3f}  Recall: {h_recall:.3f}  F1: {h_f1:.3f}")
        print("Confusion matrix [[TN, FP], [FN, TP]]:")
        print(h_cm)
        print("This is stronger evidence of real generalization than LOO alone.")
        print()
    else:
        print("Skipping held-out split -- still too few samples per class for a stratified split to be meaningful.")
        print()

    # Final model, trained on everything, for feature importance / scoring new chains
    final_model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
    final_model.fit(X, y)

    print("=== Feature importances (explainability) ===")
    importances = sorted(zip(FEATURE_NAMES, final_model.feature_importances_), key=lambda x: -x[1])
    for name, imp in importances:
        print(f"  {name:20s}: {imp:.3f}")

    with open("peeling_classifier.pkl", "wb") as f:
        pickle.dump(final_model, f)
    print("\nSaved peeling_classifier.pkl")
