"""
peeling_chain_detection.py

Day 8 build — rule-based graph-traversal detectors, no ML yet:

1. PEELING CHAIN detection: walks the wallet-only graph looking for sequences
   of transactions where each hop has exactly 2 outputs of very different
   size (a small "peel" + a large "remainder"), and the remainder address is
   then spent again in another such asymmetric transaction, for several hops
   in a row. This mirrors how real chain-analysis tools describe peeling
   chains: a large balance gets walked down via repeated small extractions.

2. COINJOIN-LIKE detection: flags any transaction with 3+ outputs of nearly
   equal value AND multiple distinct inputs -- the defining CoinJoin
   signature (many participants, each getting an equal-value output back).
   This is a STATISTICAL rule on amounts only; it does not use any ground
   truth to decide.

Both detectors are evaluated against the dataset's ground truth
(pattern_type, chain_id, coinjoin_ground_truth) AFTER detection, never during.

IMPORTANT LIMITATION, stated plainly rather than hidden: address-level graph
traversal can occasionally misattribute which "next spend" continues a chain
when an address is REUSED (receives more than one output over its lifetime).
Our generator's true peeling chains always use fresh addresses for the
continuation, so real chains are traversed correctly -- but a normal-traffic
address that happens to get reused could occasionally look like a longer or
shorter chain than it really is. This is exactly the kind of ambiguity real
UTXO-level (not address-level) forensic tools have to handle more carefully;
noted here as a scoping simplification for this prototype.
"""

import pickle
from datetime import datetime

import numpy as np

RATIO_THRESHOLD = 0.3     # small/large output ratio below this = "asymmetric" (peel-shaped)
COINJOIN_CV_THRESHOLD = 0.05  # coefficient of variation below this = "near-equal outputs"
COINJOIN_MIN_OUTPUTS = 3
COINJOIN_MIN_INPUTS = 2


def load_graph(path: str = "btc_graph_wallet_only.gpickle"):
    with open(path, "rb") as f:
        return pickle.load(f)


# ----------------------------
# 1. Peeling chain detection
# ----------------------------
def find_qualifying_2out_txs(W) -> dict:
    """Returns {tx: {"small": (addr, amt), "large": (addr, amt), "timestamp": ...}}
    for every transaction with exactly 2 outputs where the smaller is < RATIO_THRESHOLD
    of the larger -- the basic peel-shaped transaction."""
    qualifying = {}
    tx_nodes = [n for n, d in W.nodes(data=True) if d.get("node_type") == "tx"]
    for tx in tx_nodes:
        outs = [(v, d["amount"]) for _, v, d in W.out_edges(tx, data=True) if d.get("relation") == "pays_to"]
        if len(outs) != 2:
            continue
        outs.sort(key=lambda x: x[1])
        small, large = outs
        if large[1] <= 0:
            continue
        ratio = small[1] / large[1]
        if ratio < RATIO_THRESHOLD:
            qualifying[tx] = {"small": small, "large": large,
                               "timestamp": W.nodes[tx]["timestamp"], "ratio": ratio}
    return qualifying


def address_next_spend(W, address: str, after_time: str):
    """Returns the earliest transaction (after `after_time`) that spends this
    address, or None. This is the traversal step: 'where does the remainder go next?'"""
    candidates = []
    for _, tx, d in W.out_edges(address, data=True):
        if d.get("relation") != "spends_in":
            continue
        ts = W.nodes[tx]["timestamp"]
        if ts > after_time:
            candidates.append((ts, tx))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1]


def detect_peeling_chains(W, min_hops: int = 2) -> list:
    """Walks forward from every qualifying 2-output tx, following the large/
    remainder output to its next spend, as long as that next spend ALSO
    qualifies. Returns a list of chains (each a list of tx ids), keeping only
    MAXIMAL chains (dropping any chain whose start is actually the middle of
    a longer chain found elsewhere)."""
    qualifying = find_qualifying_2out_txs(W)
    consumed_as_non_start = set()
    all_chains = []

    # process in time order so earlier/longer chains are discovered before
    # their later hops get incorrectly treated as separate chain starts
    ordered_tx = sorted(qualifying.keys(), key=lambda t: qualifying[t]["timestamp"])

    for start_tx in ordered_tx:
        if start_tx in consumed_as_non_start:
            continue
        chain = [start_tx]
        current_tx = start_tx
        while True:
            large_addr, large_amt = qualifying[current_tx]["large"]
            next_tx = address_next_spend(W, large_addr, qualifying[current_tx]["timestamp"])
            if next_tx is None or next_tx not in qualifying:
                break
            chain.append(next_tx)
            consumed_as_non_start.add(next_tx)
            current_tx = next_tx
        if len(chain) >= min_hops:
            all_chains.append(chain)

    return all_chains


def evaluate_peeling_detection(W, detected_chains: list):
    """Compares detected chains against ground truth chain_id / pattern_type."""
    true_chain_ids = {W.nodes[n]["chain_id"] for n, d in W.nodes(data=True)
                       if d.get("node_type") == "tx" and d.get("chain_id")}

    detected_true_chain_ids = set()
    false_positive_chains = 0
    for chain in detected_chains:
        chain_ids_in_this_detection = {W.nodes[tx].get("chain_id") for tx in chain if W.nodes[tx].get("chain_id")}
        if chain_ids_in_this_detection:
            detected_true_chain_ids.update(chain_ids_in_this_detection)
        else:
            false_positive_chains += 1  # a "chain" made entirely of coincidental normal transactions

    recall = len(detected_true_chain_ids) / len(true_chain_ids) if true_chain_ids else 0.0
    precision = (len(detected_chains) - false_positive_chains) / len(detected_chains) if detected_chains else 0.0

    print(f"True peeling chains in dataset: {len(true_chain_ids)}")
    print(f"Candidate chains detected: {len(detected_chains)}")
    print(f"True chains successfully found (recall):     {recall:.1%} "
          f"({len(detected_true_chain_ids)}/{len(true_chain_ids)})")
    print(f"Detected chains that were real (precision):   {precision:.1%} "
          f"({len(detected_chains) - false_positive_chains}/{len(detected_chains)})")
    print(f"False-positive candidate chains (coincidental normal-traffic look-alikes): {false_positive_chains}")
    return recall, precision


# ----------------------------
# 2. CoinJoin-like detection
# ----------------------------
def detect_coinjoin_like(W) -> list:
    """Flags transactions with 3+ near-equal-value outputs and multiple inputs.
    Pure statistical rule on amounts -- no ground truth used in the decision."""
    flagged = []
    tx_nodes = [n for n, d in W.nodes(data=True) if d.get("node_type") == "tx"]
    for tx in tx_nodes:
        outs = [d["amount"] for _, v, d in W.out_edges(tx, data=True) if d.get("relation") == "pays_to"]
        ins = [u for u, v, d in W.in_edges(tx, data=True) if d.get("relation") == "spends_in"]
        if len(outs) < COINJOIN_MIN_OUTPUTS or len(ins) < COINJOIN_MIN_INPUTS:
            continue
        mean_out, std_out = np.mean(outs), np.std(outs)
        if mean_out <= 0:
            continue
        cv = std_out / mean_out
        if cv < COINJOIN_CV_THRESHOLD:
            flagged.append(tx)
    return flagged


def evaluate_coinjoin_detection(W, flagged: list):
    true_coinjoin_tx = {n for n, d in W.nodes(data=True)
                         if d.get("node_type") == "tx" and d.get("coinjoin_ground_truth")}
    flagged_set = set(flagged)
    tp = len(flagged_set & true_coinjoin_tx)
    precision = tp / len(flagged_set) if flagged_set else 0.0
    recall = tp / len(true_coinjoin_tx) if true_coinjoin_tx else 0.0
    print(f"True CoinJoin transactions in dataset: {len(true_coinjoin_tx)}")
    print(f"Flagged as CoinJoin-like: {len(flagged_set)}")
    print(f"Precision: {precision:.1%}  Recall: {recall:.1%}")
    return precision, recall


if __name__ == "__main__":
    W = load_graph("btc_graph_wallet_only.gpickle")

    print("=== Peeling Chain Detection ===")
    chains = detect_peeling_chains(W, min_hops=2)
    evaluate_peeling_detection(W, chains)
    print()

    print("=== CoinJoin-like Detection ===")
    flagged = detect_coinjoin_like(W)
    evaluate_coinjoin_detection(W, flagged)

    with open("detected_peeling_chains.pkl", "wb") as f:
        pickle.dump(chains, f)
    print("\nSaved detected_peeling_chains.pkl")
