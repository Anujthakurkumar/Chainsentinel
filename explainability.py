"""
explainability.py

Day 11 build — adds real SHAP-based explainability to BOTH models built so
far, replacing generic "anomaly score was high" text with actual per-feature
attributions: which specific features pushed a wallet's/chain's score up,
and by how much.

1. Isolation Forest (Day 7) -- shap.TreeExplainer works directly on it (it's
   tree-based under the hood). For each wallet, we get a SHAP value per
   feature showing how much that feature pushed the anomaly score away from
   the model's baseline expectation.

2. Peeling-chain Random Forest (Day 9) -- standard TreeExplainer usage,
   explaining the model's P(real peeling chain) prediction per candidate
   chain.

Both produce a human-readable reason string built from the TOP contributing
features, not just the raw numeric score — this is what turns "risk = 0.87"
into something an investigator can actually act on and defend.
"""

import pickle

import numpy as np
import pandas as pd
import shap

from anomaly_detection import load_and_transform, train_isolation_forest, get_model_feature_cols
from peeling_chain_classifier import build_dataset, FEATURE_NAMES
from peeling_chain_detection import load_graph, detect_peeling_chains

# Human-readable phrasing for each raw feature name, used when building reason strings.
FEATURE_PHRASES = {
    "avg_amount_zscore_log": "unusually large average transaction amount",
    "max_abs_amount_zscore_log": "an unusually extreme single transaction amount",
    "tx_count": "an atypical number of transactions",
    "tx_frequency_per_day_log": "an unusually high transaction frequency",
    "fan_in_degree": "an unusually high number of incoming payments",
    "fan_out_degree": "an unusually high number of outgoing payments",
    "max_geo_velocity_kmh_log": "physically implausible peer geo-velocity",
    "avg_geo_velocity_kmh_log": "consistently high peer geo-velocity",
    "time_of_day_entropy": "an atypical time-of-day transaction pattern",
    "hop_count": "chain length",
    "mean_time_gap_sec": "the average time between hops",
    "std_time_gap_sec": "variability in time between hops",
    "min_time_gap_sec": "how tightly-timed the fastest hop was",
    "max_time_gap_sec": "how tightly-timed the slowest hop was",
    "mean_ratio": "the consistency of the peel-to-remainder ratio",
    "std_ratio": "variability in the peel-to-remainder ratio",
    "decline_fraction": "how much the balance declined end-to-end",
    "log_start_amount": "the size of the starting balance",
    "mean_fee_fraction": "the fee pattern across hops",
}


def build_reason_string(feature_names, shap_row, top_n=3, min_abs_shap=0.001):
    """Turns a row of SHAP values into a human-readable reason string,
    keeping only features that pushed the score UP (positive contribution)
    and are large enough to matter."""
    contributions = [(f, v) for f, v in zip(feature_names, shap_row) if v > min_abs_shap]
    contributions.sort(key=lambda x: -x[1])
    top = contributions[:top_n]
    if not top:
        return "No individual feature stood out; flagged only by combined weak signals."
    phrases = [f"{FEATURE_PHRASES.get(f, f)} (contributed +{v:.3f})" for f, v in top]
    return "Driven by: " + "; ".join(phrases) + "."


def explain_anomaly_model(features_path: str = "wallet_features.csv"):
    df = load_and_transform(features_path)
    model, anomaly_score = train_isolation_forest(df)
    model_cols = get_model_feature_cols()
    X = df[model_cols].values

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)  # (n_wallets, n_features)

    reasons = [build_reason_string(model_cols, shap_values[i]) for i in range(len(df))]

    out = df[["wallet", "involved_in_illicit_pattern", "pattern_types_involved"]].copy()
    out["anomaly_score"] = anomaly_score
    out["shap_reason"] = reasons
    for j, col in enumerate(model_cols):
        out[f"shap_{col}"] = shap_values[:, j]
    return out.sort_values("anomaly_score", ascending=False).reset_index(drop=True)


def explain_peeling_model(graph_path: str = "btc_graph_wallet_only.gpickle",
                           classifier_path: str = "peeling_classifier.pkl"):
    W = load_graph(graph_path)
    chains = detect_peeling_chains(W, min_hops=2)
    X, y = build_dataset(W, chains)

    with open(classifier_path, "rb") as f:
        clf = pickle.load(f)

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X)  # (n_chains, n_features, n_classes)
    shap_class1 = shap_values[:, :, 1]  # attributions toward "real peeling chain"

    probas = clf.predict_proba(X)[:, 1]
    reasons = [build_reason_string(FEATURE_NAMES, shap_class1[i]) for i in range(len(chains))]

    rows = []
    for i, chain in enumerate(chains):
        rows.append({
            "chain_index": i,
            "hop_count": len(chain),
            "true_label": int(y[i]),
            "predicted_probability": probas[i],
            "shap_reason": reasons[i],
        })
    return pd.DataFrame(rows).sort_values("predicted_probability", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    print("=== Isolation Forest SHAP explanations (top 5 most anomalous wallets) ===")
    anomaly_explanations = explain_anomaly_model()
    for _, row in anomaly_explanations.head(5).iterrows():
        print(f"{row['wallet'][:20]}...  score={row['anomaly_score']:.3f}  "
              f"illicit_gt={row['involved_in_illicit_pattern']}")
        print(f"    {row['shap_reason']}")
    anomaly_explanations.to_csv("wallet_shap_explanations.csv", index=False)
    print("\nSaved wallet_shap_explanations.csv\n")

    print("=== Peeling-chain classifier SHAP explanations (all candidate chains) ===")
    chain_explanations = explain_peeling_model()
    for _, row in chain_explanations.iterrows():
        print(f"Chain {row['chain_index']}: P(peeling)={row['predicted_probability']:.3f}  "
              f"true_label={row['true_label']}")
        print(f"    {row['shap_reason']}")
    chain_explanations.to_csv("chain_shap_explanations.csv", index=False)
    print("\nSaved chain_shap_explanations.csv")
