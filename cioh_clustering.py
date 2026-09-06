"""
cioh_clustering.py

Day 5 build — Common-Input-Ownership Heuristic (CIOH) clustering using
Union-Find, run on btc_graph_wallet_only.gpickle, then evaluated against
the entity_ground_truth already sitting on each wallet node.

The heuristic: if two+ wallet addresses appear as INPUTS to the same
transaction, assume they're controlled by the same real-world entity, and
union() them. Do this for every transaction, then read out the resulting
clusters as predicted_entity_id.

This will NOT be perfect by design -- CoinJoin-like transactions (Day 3)
were specifically injected to violate this assumption (multiple independent
entities co-signing one transaction), so a chunk of your false-merge errors
should trace directly back to those. Measuring that is the point: it proves
your dataset actually stresses the heuristic instead of only containing
cases where it trivially works.
"""

import pickle
from collections import defaultdict
from itertools import combinations

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from union_find import UnionFind


def run_cioh_clustering(graph_path: str = "btc_graph_wallet_only.gpickle"):
    with open(graph_path, "rb") as f:
        W = pickle.load(f)

    uf = UnionFind()
    coinjoin_false_merge_pairs = 0  # pairs unioned by CIOH that are NOT the same true entity, from coinjoin txs
    coinjoin_total_merge_pairs = 0

    tx_nodes = [n for n, d in W.nodes(data=True) if d.get("node_type") == "tx"]
    for tx in tx_nodes:
        input_wallets = [u for u, v, d in W.in_edges(tx, data=True) if d.get("relation") == "spends_in"]
        if len(input_wallets) < 2:
            continue  # CIOH only applies when there's something to co-own

        is_coinjoin = W.nodes[tx].get("pattern_type") == "coinjoin"

        for w1, w2 in combinations(input_wallets, 2):
            if is_coinjoin:
                e1 = W.nodes[w1].get("entity_ground_truth")
                e2 = W.nodes[w2].get("entity_ground_truth")
                coinjoin_total_merge_pairs += 1
                if e1 is not None and e2 is not None and e1 != e2:
                    coinjoin_false_merge_pairs += 1
            uf.union(w1, w2)

    wallet_nodes = [n for n, d in W.nodes(data=True) if d.get("node_type") == "wallet"]
    predicted_entity_id = {w: uf.find(w) for w in wallet_nodes}

    return predicted_entity_id, W, {
        "coinjoin_total_merge_pairs": coinjoin_total_merge_pairs,
        "coinjoin_false_merge_pairs": coinjoin_false_merge_pairs,
    }


def pair_counting_metrics(true_labels: dict, pred_labels: dict) -> dict:
    """
    Standard pair-counting precision/recall for clustering evaluation:
      precision = of all wallet pairs CIOH put in the same cluster,
                  what fraction are ACTUALLY the same true entity?
      recall    = of all wallet pairs that ARE the same true entity,
                  what fraction did CIOH correctly put together?
    This is the concrete "detects X% with Y% false positive rate" style
    number worth putting in your write-up.
    """
    wallets = [w for w in true_labels if true_labels[w] is not None]

    true_groups, pred_groups = defaultdict(set), defaultdict(set)
    for w in wallets:
        true_groups[true_labels[w]].add(w)
        pred_groups[pred_labels[w]].add(w)

    def pair_count(groups):
        return sum(len(g) * (len(g) - 1) // 2 for g in groups.values())

    total_true_pairs = pair_count(true_groups)
    total_pred_pairs = pair_count(pred_groups)

    intersection = 0
    for members in pred_groups.values():
        sub = defaultdict(int)
        for w in members:
            sub[true_labels[w]] += 1
        intersection += sum(c * (c - 1) // 2 for c in sub.values())

    precision = intersection / total_pred_pairs if total_pred_pairs else 0.0
    recall = intersection / total_true_pairs if total_true_pairs else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "precision": precision, "recall": recall, "f1": f1,
        "true_pairs": total_true_pairs, "pred_pairs": total_pred_pairs,
        "correctly_merged_pairs": intersection,
    }


if __name__ == "__main__":
    predicted_entity_id, W, coinjoin_stats = run_cioh_clustering()

    true_labels = {w: W.nodes[w].get("entity_ground_truth") for w in predicted_entity_id}
    metrics = pair_counting_metrics(true_labels, predicted_entity_id)

    # standard sklearn metrics too, for external credibility in the write-up
    wallets = [w for w in true_labels if true_labels[w] is not None]
    true_arr = [true_labels[w] for w in wallets]
    pred_arr = [predicted_entity_id[w] for w in wallets]
    ari = adjusted_rand_score(true_arr, pred_arr)
    nmi = normalized_mutual_info_score(true_arr, pred_arr)

    n_predicted_clusters = len(set(predicted_entity_id.values()))
    n_true_entities = len(set(true_arr))

    print("=== CIOH Clustering Results ===")
    print(f"Wallets clustered: {len(predicted_entity_id)}")
    print(f"True entity count:      {n_true_entities}")
    print(f"Predicted cluster count: {n_predicted_clusters}")
    print()
    print(f"Pair-counting precision: {metrics['precision']:.3f}")
    print(f"Pair-counting recall:    {metrics['recall']:.3f}")
    print(f"Pair-counting F1:        {metrics['f1']:.3f}")
    print(f"  ({metrics['correctly_merged_pairs']}/{metrics['pred_pairs']} predicted-same-cluster pairs "
          f"were actually the same entity)")
    print(f"  ({metrics['correctly_merged_pairs']}/{metrics['true_pairs']} true-same-entity pairs "
          f"were correctly merged)")
    print()
    print(f"Adjusted Rand Index:     {ari:.3f}")
    print(f"Normalized Mutual Info:  {nmi:.3f}")
    print()
    print("=== CoinJoin's specific impact on CIOH (as designed) ===")
    if coinjoin_stats["coinjoin_total_merge_pairs"]:
        rate = coinjoin_stats["coinjoin_false_merge_pairs"] / coinjoin_stats["coinjoin_total_merge_pairs"]
        print(f"CoinJoin-driven wallet-pair unions: {coinjoin_stats['coinjoin_total_merge_pairs']}")
        print(f"Of those, FALSE merges (different true entities wrongly joined): "
              f"{coinjoin_stats['coinjoin_false_merge_pairs']} ({rate:.1%})")
        print("This confirms CIOH alone cannot handle CoinJoin-style mixing --")
        print("exactly the gap your Day 6+ anomaly/mixing detection needs to cover.")

    with open("predicted_entity_id.pkl", "wb") as f:
        pickle.dump(predicted_entity_id, f)
    print("\nSaved predicted_entity_id.pkl")
