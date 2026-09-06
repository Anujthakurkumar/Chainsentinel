"""
risk_propagation.py

Day 10 build — the final synthesis step. Combines three independently-built
signals into ONE ranked, explainable risk score per wallet:

  1. PROPAGATED RISK (built here) -- Personalized PageRank seeded from a SMALL,
     realistic set of "known illicit" wallets (simulating external threat intel
     on a handful of confirmed cash-out addresses, NOT full ground truth --
     using the full ground truth as seeds would be circular/cheating).
     Risk flows outward through the transaction graph, decaying with hop
     distance and weighted by transaction amount (more money moved = stronger
     link), matching how real chain-analysis "poison"/haircut risk-scoring
     methods work conceptually.

  2. ANOMALY SCORE (Day 7) -- Isolation Forest's per-wallet anomaly score.

  3. PEELING-CHAIN SCORE (Day 9) -- the Random Forest's predicted probability
     that a wallet's chain is a genuine peeling chain, mapped from chains to
     the individual wallets that participate in them.

Combination method: NOISY-OR, not a plain average.
  final_risk = 1 - (1-r_propagated)*(1-r_anomaly)*(1-r_peeling)
This reflects "ANY strong signal alone should raise suspicion" -- a wallet
that scores very high on just ONE signal (e.g. directly touches a known-bad
address) shouldn't have its risk diluted by two unrelated near-zero signals,
which is what a plain average would do.
"""

import pickle

import networkx as nx
import numpy as np
import pandas as pd

from graph_builder import wallet_tx_subgraph
from peeling_chain_classifier import find_qualifying_2out_txs, extract_chain_features, FEATURE_NAMES

# Known from generate_dataset.py's config (NUM_NORMAL_ENTITIES=60, NUM_CASHOUT_ENTITIES=5) --
# these entity ids are reserved as mule/cash-out sinks in the generator.
CASHOUT_ENTITY_IDS = set(range(60, 65))
N_SEED_WALLETS = 6          # a SMALL, realistic amount of external threat intel
PAGERANK_ALPHA = 0.55       # lower than the default 0.85 -- we want risk to decay
                            # fast with hop distance, not diffuse broadly like
                            # general web PageRank does


def load_full_graph(path: str = "btc_graph.gpickle"):
    with open(path, "rb") as f:
        return pickle.load(f)


def select_seed_wallets(W, n_seeds: int = N_SEED_WALLETS, random_state: int = 3) -> list:
    """Simulates a small amount of external threat intelligence: a handful of
    wallets known (from outside this system) to belong to cash-out/mule
    entities. NOT the full set -- that would make propagation circular."""
    cashout_wallets = [n for n, d in W.nodes(data=True)
                        if d.get("node_type") == "wallet" and d.get("entity_ground_truth") in CASHOUT_ENTITY_IDS]
    rng = np.random.RandomState(random_state)
    n_seeds = min(n_seeds, len(cashout_wallets))
    return list(rng.choice(cashout_wallets, size=n_seeds, replace=False))


def propagate_risk(W, seed_wallets: list) -> dict:
    """Personalized PageRank over the wallet+tx graph (no IP layer -- same
    reasoning as Day 5's wallet_tx_subgraph: the IP layer would let risk leak
    between unrelated wallets just because they share a relay peer)."""
    personalization = {n: (1.0 if n in seed_wallets else 0.0) for n in W.nodes()}
    total = sum(personalization.values())
    personalization = {n: v / total for n, v in personalization.items()}

    scores = nx.pagerank(W, alpha=PAGERANK_ALPHA, personalization=personalization, weight="amount")
    wallet_scores = {n: s for n, s in scores.items() if W.nodes[n].get("node_type") == "wallet"}
    max_score = max(wallet_scores.values()) if wallet_scores else 1.0
    return {w: s / max_score for w, s in wallet_scores.items()}  # normalize to [0,1]


def load_anomaly_scores(path: str = "wallet_anomaly_scores.csv") -> dict:
    df = pd.read_csv(path)
    raw = dict(zip(df["wallet"], df["anomaly_score"]))
    vals = np.array(list(raw.values()))
    lo, hi = vals.min(), vals.max()
    return {w: (s - lo) / (hi - lo) if hi > lo else 0.0 for w, s in raw.items()}


def load_peeling_scores(W, chains_path: str = "detected_peeling_chains.pkl",
                         classifier_path: str = "peeling_classifier.pkl") -> dict:
    """Maps Day 9's chain-level peeling probability down to individual wallets:
    a wallet's peeling_score is the MAX probability among all chains it
    participates in (a wallet touched by even one high-confidence peeling
    chain should be flagged, not averaged down by other chains it's not in)."""
    with open(chains_path, "rb") as f:
        chains = pickle.load(f)
    with open(classifier_path, "rb") as f:
        clf = pickle.load(f)

    qualifying = find_qualifying_2out_txs(W)
    wallet_peeling_score = {}
    for chain in chains:
        feats = extract_chain_features(W, qualifying, chain)
        X = np.array([[feats[name] for name in FEATURE_NAMES]])
        proba = clf.predict_proba(X)[0][1]  # P(class=1, i.e. real peeling chain)

        chain_wallets = set()
        for tx in chain:
            for addr, _ in [qualifying[tx]["small"], qualifying[tx]["large"]]:
                chain_wallets.add(addr)
            for u, _, d in W.in_edges(tx, data=True):
                if d.get("relation") == "spends_in":
                    chain_wallets.add(u)

        for w in chain_wallets:
            wallet_peeling_score[w] = max(wallet_peeling_score.get(w, 0.0), proba)
    return wallet_peeling_score


def combine_noisy_or(r_propagated, r_anomaly, r_peeling) -> float:
    return 1 - (1 - r_propagated) * (1 - r_anomaly) * (1 - r_peeling)


def explain(row) -> str:
    reasons = []
    if row["risk_propagated"] > 0.3:
        reasons.append(f"network proximity to known cash-out wallets (propagated risk {row['risk_propagated']:.2f})")
    if row["risk_anomaly"] > 0.5:
        reasons.append(f"statistically anomalous behavior (anomaly score {row['risk_anomaly']:.2f})")
    if row["risk_peeling"] > 0.5:
        reasons.append(f"participation in a high-confidence peeling chain (probability {row['risk_peeling']:.2f})")
    if not reasons:
        return "No single strong signal; flagged only due to combined weak evidence."
    return "Flagged for: " + "; ".join(reasons) + "."


if __name__ == "__main__":
    W_full = load_full_graph("btc_graph.gpickle")
    W = wallet_tx_subgraph(W_full)  # propagate on wallet+tx only, no IP layer

    seeds = select_seed_wallets(W)
    print(f"Seed wallets (simulated external threat intel, n={len(seeds)}):")
    for s in seeds:
        print(f"  {s}  (entity {W.nodes[s].get('entity_ground_truth')})")
    print()

    risk_propagated = propagate_risk(W, seeds)
    risk_anomaly = load_anomaly_scores("wallet_anomaly_scores.csv")
    risk_peeling = load_peeling_scores(W, "detected_peeling_chains.pkl", "peeling_classifier.pkl")

    all_wallets = [n for n, d in W.nodes(data=True) if d.get("node_type") == "wallet"]
    rows = []
    for w in all_wallets:
        r_prop = risk_propagated.get(w, 0.0)
        r_anom = risk_anomaly.get(w, 0.0)
        r_peel = risk_peeling.get(w, 0.0)
        final_risk = combine_noisy_or(r_prop, r_anom, r_peel)
        rows.append({
            "wallet": w,
            "risk_propagated": r_prop,
            "risk_anomaly": r_anom,
            "risk_peeling": r_peel,
            "final_risk": final_risk,
            "involved_in_illicit_pattern": W.nodes[w].get("entity_ground_truth") is not None,  # placeholder, overwritten below
        })

    df = pd.DataFrame(rows)

    # pull the real ground truth from Day 6/7's feature file rather than re-deriving it
    feat_df = pd.read_csv("wallet_features.csv")[["wallet", "involved_in_illicit_pattern", "pattern_types_involved"]]
    df = df.drop(columns=["involved_in_illicit_pattern"]).merge(feat_df, on="wallet", how="left")
    df["involved_in_illicit_pattern"] = df["involved_in_illicit_pattern"].fillna(False)
    df["is_seed"] = df["wallet"].isin(seeds)

    df["explanation"] = df.apply(explain, axis=1)
    # Secondary sort key: noisy-OR saturates toward 1.0 once ANY signal is strong, so many
    # top wallets can tie at the same final_risk. Break ties by total supporting evidence
    # (sum of the three raw signals) -- a wallet with 3 moderately-elevated signals should
    # rank above one with only 1 saturated signal and 2 near-zero ones, even if their
    # final_risk rounds to the same value.
    df["tie_break_score"] = df["risk_propagated"] + df["risk_anomaly"] + df["risk_peeling"]
    df = df.sort_values(["final_risk", "tie_break_score"], ascending=[False, False]).reset_index(drop=True)

    print("=== Top 10 ranked alerts ===")
    for _, row in df.head(10).iterrows():
        seed_tag = " [SEED]" if row["is_seed"] else ""
        print(f"{row['wallet'][:20]}...  risk={row['final_risk']:.3f} "
              f"(evidence_sum={row['tie_break_score']:.2f}){seed_tag}")
        print(f"    {row['explanation']}")
    print()

    # ---- Evaluate: does combining actually beat any single signal alone? ----
    from sklearn.metrics import average_precision_score
    y = df["involved_in_illicit_pattern"].astype(int).values
    print("=== AUC-PR comparison: combined vs. each signal alone ===")
    for col, label in [("risk_propagated", "Propagated risk only"),
                        ("risk_anomaly", "Anomaly score only"),
                        ("risk_peeling", "Peeling score only"),
                        ("final_risk", "COMBINED (noisy-OR)")]:
        ap = average_precision_score(y, df[col].values)
        print(f"{label:25s}: AUC-PR = {ap:.3f}")

    df.to_csv("wallet_final_risk_scores.csv", index=False)
    print("\nSaved wallet_final_risk_scores.csv")
