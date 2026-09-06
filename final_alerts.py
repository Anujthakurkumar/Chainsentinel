"""
final_alerts.py

Day 11 (integration) — merges Day 11's SHAP explanations into Day 10's final
ranked risk list, so the alerts an investigator actually sees carry SPECIFIC
feature-level reasons ("driven by unusually high transaction frequency and
tight hop timing") instead of generic score restatements ("anomaly score was
high"). This is the actual "explainable, evidence-backed alert list with
confidence score" the problem statement asks for, fully assembled.

Propagated risk (Day 10) doesn't get a SHAP explanation -- PageRank isn't a
supervised model, so there's nothing for SHAP to attribute. Its explanation
stays as "network proximity to known cash-out wallets", which is already
fully transparent by construction (you can literally trace the graph path).
"""

import pickle

import pandas as pd

from explainability import explain_anomaly_model, explain_peeling_model
from peeling_chain_detection import load_graph, detect_peeling_chains, find_qualifying_2out_txs


def map_chain_reasons_to_wallets(W, chains, chain_explanations: pd.DataFrame) -> dict:
    """For each wallet, keep the SHAP reason from whichever chain gave it the
    HIGHEST peeling probability (same max-based mapping Day 10 used for the score
    itself, kept consistent here)."""
    qualifying = find_qualifying_2out_txs(W)
    wallet_reason = {}
    wallet_best_proba = {}

    for i, chain in enumerate(chains):
        row = chain_explanations[chain_explanations["chain_index"] == i].iloc[0]
        proba, reason = row["predicted_probability"], row["shap_reason"]

        chain_wallets = set()
        for tx in chain:
            for addr, _ in [qualifying[tx]["small"], qualifying[tx]["large"]]:
                chain_wallets.add(addr)
            for u, _, d in W.in_edges(tx, data=True):
                if d.get("relation") == "spends_in":
                    chain_wallets.add(u)

        for w in chain_wallets:
            if proba > wallet_best_proba.get(w, -1):
                wallet_best_proba[w] = proba
                wallet_reason[w] = reason
    return wallet_reason


def build_final_explanation(row) -> str:
    parts = []
    if row["risk_propagated"] > 0.3:
        parts.append(f"[Network] Proximity to known cash-out wallets (propagated risk {row['risk_propagated']:.2f}).")
    if row["risk_anomaly"] > 0.5 and pd.notna(row.get("anomaly_shap_reason")):
        parts.append(f"[Anomaly] {row['anomaly_shap_reason']}")
    if row["risk_peeling"] > 0.5 and pd.notna(row.get("peeling_shap_reason")):
        parts.append(f"[Peeling-chain, P={row['risk_peeling']:.2f}] {row['peeling_shap_reason']}")
    if not parts:
        return "No single strong signal; flagged only due to combined weak evidence."
    return " ".join(parts)


if __name__ == "__main__":
    risk_df = pd.read_csv("wallet_final_risk_scores.csv")

    anomaly_explanations = explain_anomaly_model("wallet_features.csv")
    anomaly_lookup = dict(zip(anomaly_explanations["wallet"], anomaly_explanations["shap_reason"]))

    W = load_graph("btc_graph_wallet_only.gpickle")
    chains = detect_peeling_chains(W, min_hops=2)
    chain_explanations = explain_peeling_model("btc_graph_wallet_only.gpickle", "peeling_classifier.pkl")
    peeling_lookup = map_chain_reasons_to_wallets(W, chains, chain_explanations)

    risk_df["anomaly_shap_reason"] = risk_df["wallet"].map(anomaly_lookup)
    risk_df["peeling_shap_reason"] = risk_df["wallet"].map(peeling_lookup)
    risk_df["final_explanation"] = risk_df.apply(build_final_explanation, axis=1)

    risk_df = risk_df.sort_values(["final_risk", "tie_break_score"], ascending=[False, False]).reset_index(drop=True)

    print("=== Final ranked, SHAP-explained alerts (top 10) ===\n")
    for _, row in risk_df.head(10).iterrows():
        seed_tag = " [SEED]" if row.get("is_seed") else ""
        print(f"{row['wallet'][:24]}...  final_risk={row['final_risk']:.3f}{seed_tag}")
        print(f"   {row['final_explanation']}\n")

    risk_df.to_csv("final_alerts_explained.csv", index=False)
    print("Saved final_alerts_explained.csv")
