"""
change_address_heuristic.py

Day 5 (extended) build — adds a change-address heuristic as a SECOND
clustering pass on top of base CIOH, to recover recall that pure CIOH
misses (pure CIOH can only link addresses spent TOGETHER in one tx; it can
never link an entity's address at time T to that same entity's fresh change
address created at time T, which only becomes useful later when THAT
address is itself spent).

IMPORTANT — this heuristic does NOT use is_change_ground_truth to decide.
That would be circular (using the answer key to make the guess). It uses
only an observable signal: address freshness, computed CAUSALLY (only
using transactions that happened strictly before the one being examined,
never peeking at future data). is_change_ground_truth is used ONLY
afterward, to measure how good the guess was.

Heuristic (2-output transactions only, the classic case this applies to):
  - If exactly ONE of the two output addresses has never appeared anywhere
    in the dataset before this transaction (a "fresh" address) and the
    other has been seen before (a "reused" address) -> predict the fresh
    one is change.
  - If both are fresh, or both are reused -> ABSTAIN (no confident guess).
    Guessing blindly on ambiguous cases would inflate false positives for
    no real recall gain -- abstaining is the correct call here.

Then: for every confident change prediction, union() the tx's input
wallet(s) with the predicted change-output wallet in the SAME Union-Find
structure used for base CIOH. This is what lets clusters propagate across
time -- once merged, that address later being spent will pull in whatever
it's grouped with at that point too.
"""

import pickle
from itertools import combinations

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from union_find import UnionFind
from cioh_clustering import pair_counting_metrics


def run_combined_clustering(graph_path: str = "btc_graph_wallet_only.gpickle"):
    with open(graph_path, "rb") as f:
        W = pickle.load(f)

    tx_nodes = [n for n, d in W.nodes(data=True) if d.get("node_type") == "tx"]
    tx_nodes.sort(key=lambda n: W.nodes[n]["timestamp"])  # process causally, earliest first

    uf = UnionFind()
    seen_addresses = set()  # addresses that have appeared as an OUTPUT before the current tx

    change_pred_correct = 0
    change_pred_total = 0
    change_recall_hits = 0
    change_recall_total = 0  # total real change outputs among 2-output txs (for recall)

    coinjoin_total_merge_pairs = 0
    coinjoin_false_merge_pairs = 0

    for tx in tx_nodes:
        # ---- base CIOH pass: union all input wallets of this tx ----
        input_wallets = [u for u, v, d in W.in_edges(tx, data=True) if d.get("relation") == "spends_in"]
        is_coinjoin = W.nodes[tx].get("pattern_type") == "coinjoin"
        for w1, w2 in combinations(input_wallets, 2):
            if is_coinjoin:
                e1, e2 = W.nodes[w1].get("entity_ground_truth"), W.nodes[w2].get("entity_ground_truth")
                coinjoin_total_merge_pairs += 1
                if e1 is not None and e2 is not None and e1 != e2:
                    coinjoin_false_merge_pairs += 1
            uf.union(w1, w2)

        # ---- change-address heuristic pass (2-output txs only) ----
        output_edges = sorted(
            [(v, d) for _, v, d in W.out_edges(tx, data=True) if d.get("relation") == "pays_to"],
            key=lambda x: x[1]["vout"],
        )
        if len(output_edges) == 2:
            (addr_a, data_a), (addr_b, data_b) = output_edges
            a_fresh = addr_a not in seen_addresses
            b_fresh = addr_b not in seen_addresses

            # ground truth, used ONLY for scoring below, never for the decision itself
            a_is_change_gt = data_a.get("is_change_ground_truth")
            b_is_change_gt = data_b.get("is_change_ground_truth")
            if True in (a_is_change_gt, b_is_change_gt):
                change_recall_total += 1

            if a_fresh != b_fresh:  # exactly one fresh -> confident guess
                predicted_change_addr = addr_a if a_fresh else addr_b
                predicted_gt = a_is_change_gt if a_fresh else b_is_change_gt

                change_pred_total += 1
                if predicted_gt is True:
                    change_pred_correct += 1
                    change_recall_hits += 1

                # THE ACTUAL CLUSTERING PAYOFF: union input wallets with predicted change output
                for in_w in input_wallets:
                    uf.union(in_w, predicted_change_addr)

        # mark both outputs as seen for future causal lookups
        for v, _ in output_edges:
            seen_addresses.add(v)

    wallet_nodes = [n for n, d in W.nodes(data=True) if d.get("node_type") == "wallet"]
    predicted_entity_id = {w: uf.find(w) for w in wallet_nodes}

    change_stats = {
        "predictions_made": change_pred_total,
        "predictions_correct": change_pred_correct,
        "precision": change_pred_correct / change_pred_total if change_pred_total else 0.0,
        "recall": change_recall_hits / change_recall_total if change_recall_total else 0.0,
        "total_real_change_cases_2output": change_recall_total,
    }
    coinjoin_stats = {
        "coinjoin_total_merge_pairs": coinjoin_total_merge_pairs,
        "coinjoin_false_merge_pairs": coinjoin_false_merge_pairs,
    }
    return predicted_entity_id, W, change_stats, coinjoin_stats


def evaluate(predicted_entity_id, W, label):
    true_labels = {w: W.nodes[w].get("entity_ground_truth") for w in predicted_entity_id}
    metrics = pair_counting_metrics(true_labels, predicted_entity_id)

    wallets = [w for w in true_labels if true_labels[w] is not None]
    true_arr = [true_labels[w] for w in wallets]
    pred_arr = [predicted_entity_id[w] for w in wallets]
    ari = adjusted_rand_score(true_arr, pred_arr)
    nmi = normalized_mutual_info_score(true_arr, pred_arr)

    print(f"=== {label} ===")
    print(f"Predicted clusters: {len(set(predicted_entity_id.values()))} (true entities: {len(set(true_arr))})")
    print(f"Precision: {metrics['precision']:.3f}  Recall: {metrics['recall']:.3f}  F1: {metrics['f1']:.3f}")
    print(f"ARI: {ari:.3f}  NMI: {nmi:.3f}")
    print()
    return metrics


if __name__ == "__main__":
    # Baseline for comparison
    from cioh_clustering import run_cioh_clustering
    baseline_pred, W, _ = run_cioh_clustering()
    baseline_metrics = evaluate(baseline_pred, W, "Baseline: CIOH only")

    # Combined: CIOH + change-address heuristic
    combined_pred, W2, change_stats, coinjoin_stats = run_combined_clustering()
    combined_metrics = evaluate(combined_pred, W2, "Combined: CIOH + change-address heuristic")

    print("=== Change-address heuristic's own accuracy (vs is_change_ground_truth) ===")
    print(f"Confident predictions made: {change_stats['predictions_made']} "
          f"(out of {change_stats['total_real_change_cases_2output']} real change cases in 2-output txs)")
    print(f"Precision (of confident guesses): {change_stats['precision']:.3f}")
    print(f"Recall (of all real change cases): {change_stats['recall']:.3f}")
    print()

    print("=== Recall improvement from adding the change-address pass ===")
    print(f"CIOH-only recall:        {baseline_metrics['recall']:.3f}")
    print(f"CIOH + change recall:    {combined_metrics['recall']:.3f}")
    delta = combined_metrics['recall'] - baseline_metrics['recall']
    print(f"Absolute improvement:    +{delta:.3f} "
          f"({delta / baseline_metrics['recall']:.1%} relative increase)" if baseline_metrics['recall'] else "")
    print()
    print(f"Precision change: {baseline_metrics['precision']:.3f} -> {combined_metrics['precision']:.3f} "
          "(worth checking this didn't drop much in exchange for the recall gain)")

    print()
    print("=== CoinJoin impact (unchanged by this pass, as expected -- it only affects 2-output txs) ===")
    rate = coinjoin_stats["coinjoin_false_merge_pairs"] / coinjoin_stats["coinjoin_total_merge_pairs"]
    print(f"CoinJoin false-merge rate: {rate:.1%} "
          f"({coinjoin_stats['coinjoin_false_merge_pairs']}/{coinjoin_stats['coinjoin_total_merge_pairs']})")

    with open("predicted_entity_id_v2.pkl", "wb") as f:
        pickle.dump(combined_pred, f)
    print("\nSaved predicted_entity_id_v2.pkl")
