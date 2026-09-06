"""
graph_builder.py

Day 4 build — loads btc_dataset.json into a NetworkX MultiDiGraph with three
node types (wallet, tx, ip) and typed, attributed, directed edges between them.

Every later stage of the project runs on top of THIS graph:
  - Day 5  CIOH clustering        -> reads wallet--spends_in-->tx edges
  - Day 6-7 Anomaly detection     -> reads per-wallet/tx features off node attrs
  - Day 8-9 Peeling-chain detect  -> walks tx->wallet->tx chains
  - Day 10 Risk propagation       -> propagates scores along all edges

Graph model
-----------
Node types (stored as node attribute "node_type"):
  wallet  -- one per unique address (input or output)
  tx      -- one per unique txid
  ip      -- one per unique peer IP observed

Edge types (directed, MultiDiGraph so parallel edges between the same pair
are allowed -- e.g. a wallet can appear as an input to the same tx only once,
but the same wallet pair can transact again in a later tx):
  wallet --spends_in-->        tx      (this wallet's UTXO was consumed as an input)
  tx     --pays_to-->          wallet  (this tx created an output paying this wallet)
  tx     --observed_relayed_by--> ip   (peer OBSERVED relaying/sending this tx --
                                         NOT proof of tx origin, see data_schema.md)
  tx     --observed_relayed_to--> ip   (peer OBSERVED receiving the relay)

Ground truth (generation-time fact, kept on node/edge attrs, clearly labeled
"_ground_truth" so it's never confused with anything a model predicts later):
  wallet node: entity_ground_truth
  tx node:     pattern_type, chain_id, group_id, coinjoin_ground_truth,
               sender_entity_ground_truth
  pays_to edge: is_change_ground_truth, output_entity_ground_truth
"""

import json
import pickle

import networkx as nx


def build_graph(dataset_path: str = "btc_dataset.json") -> nx.MultiDiGraph:
    with open(dataset_path, "r") as f:
        transactions = json.load(f)

    G = nx.MultiDiGraph()
    wallet_entity_ground_truth = {}  # address -> entity_id, built as we go, should stay consistent

    for tx in transactions:
        txid = tx["txid"]

        # ---- tx node ----
        G.add_node(
            txid,
            node_type="tx",
            timestamp=tx["timestamp"],
            fee=tx["fee"],
            pattern_type=tx.get("pattern_type", tx.get("label", "unknown")),
            chain_id=tx.get("chain_id"),
            group_id=tx.get("group_id"),
            coinjoin_ground_truth=tx.get("coinjoin_ground_truth", False),
            sender_entity_ground_truth=tx.get("sender_entity_ground_truth"),
        )

        # ---- input wallets: wallet --spends_in--> tx ----
        input_addrs = tx["input_addresses"]
        input_amounts = tx["input_amounts"]
        input_prevouts = tx.get("input_prevouts", [None] * len(input_addrs))
        sender_entity = tx.get("sender_entity_ground_truth")

        for addr, amt, prevout in zip(input_addrs, input_amounts, input_prevouts):
            if addr not in G:
                G.add_node(addr, node_type="wallet", entity_ground_truth=sender_entity)
            elif sender_entity is not None:
                # keep the first-seen entity assignment; flag if generator data is inconsistent
                existing = G.nodes[addr].get("entity_ground_truth")
                if existing is not None and existing != sender_entity:
                    G.nodes[addr]["entity_ground_truth_conflict"] = True
            G.add_edge(addr, txid, relation="spends_in", amount=amt, prevout=prevout)

        # ---- output wallets: tx --pays_to--> wallet ----
        output_addrs = tx["output_addresses"]
        output_amounts = tx["output_amounts"]
        output_entities = tx.get("output_entity_ground_truth", [None] * len(output_addrs))
        is_change_flags = tx.get("is_change_ground_truth", [None] * len(output_addrs))

        for i, (addr, amt) in enumerate(zip(output_addrs, output_amounts)):
            out_entity = output_entities[i] if i < len(output_entities) else None
            is_change = is_change_flags[i] if i < len(is_change_flags) else None
            if addr not in G:
                G.add_node(addr, node_type="wallet", entity_ground_truth=out_entity)
            G.add_edge(
                txid, addr,
                relation="pays_to", amount=amt, vout=i,
                is_change_ground_truth=is_change,
                output_entity_ground_truth=out_entity,
            )
            if out_entity is not None:
                wallet_entity_ground_truth[addr] = out_entity

        # ---- network layer: tx --observed_relayed_by/to--> ip ----
        src_ip, dst_ip = tx["src_ip"], tx["dst_ip"]
        for ip, relation in [(src_ip, "observed_relayed_by"), (dst_ip, "observed_relayed_to")]:
            if ip not in G:
                G.add_node(ip, node_type="ip", geo_country=tx.get("geo_country"), asn=tx.get("asn"))
            G.add_edge(txid, ip, relation=relation, timestamp=tx["timestamp"],
                       src_port=tx.get("src_port"), dst_port=tx.get("dst_port"))

    return G


def print_summary(G: nx.MultiDiGraph):
    node_types = {}
    for _, data in G.nodes(data=True):
        t = data.get("node_type", "unknown")
        node_types[t] = node_types.get(t, 0) + 1

    edge_relations = {}
    for _, _, data in G.edges(data=True):
        r = data.get("relation", "unknown")
        edge_relations[r] = edge_relations.get(r, 0) + 1

    print(f"Total nodes: {G.number_of_nodes()}")
    for t, n in sorted(node_types.items()):
        print(f"  {t}: {n}")
    print(f"Total edges: {G.number_of_edges()}")
    for r, n in sorted(edge_relations.items()):
        print(f"  {r}: {n}")

    # weakly connected components (treat as undirected for this check) --
    # a handful of large components + isolated small ones is expected/healthy;
    # ONE giant component swallowing everything would suggest overly dense fan-out
    wallet_tx_ip = [n for n, d in G.nodes(data=True)]
    components = list(nx.weakly_connected_components(G))
    sizes = sorted((len(c) for c in components), reverse=True)
    print(f"\nWeakly connected components: {len(components)}")
    print(f"Largest component sizes (top 5): {sizes[:5]}")

    pattern_tx_nodes = [n for n, d in G.nodes(data=True)
                         if d.get("node_type") == "tx" and d.get("pattern_type") not in (None, "normal")]
    print(f"\nNon-normal tx nodes tagged in graph: {len(pattern_tx_nodes)}")


def wallet_tx_subgraph(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """
    Returns a copy of G with IP nodes (and their edges) removed, keeping only
    wallet<->tx structure. Use THIS for entity clustering (Day 5) and
    peeling-chain traversal (Day 8-9) -- the full graph's IP layer bridges
    otherwise-unrelated wallets just because they share a relay peer, which
    would corrupt entity-clustering results if left in.
    """
    ip_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "ip"]
    H = G.copy()
    H.remove_nodes_from(ip_nodes)
    return H


if __name__ == "__main__":
    G = build_graph("btc_dataset.json")
    print_summary(G)

    with open("btc_graph.gpickle", "wb") as f:
        pickle.dump(G, f)
    print("\nSaved full graph (wallet+tx+ip) to btc_graph.gpickle")

    W = wallet_tx_subgraph(G)
    print(f"\nWallet-only projection: {W.number_of_nodes()} nodes, {W.number_of_edges()} edges")
    wallet_components = list(nx.weakly_connected_components(W))
    wc_sizes = sorted((len(c) for c in wallet_components), reverse=True)
    print(f"Weakly connected components (wallet+tx only): {len(wallet_components)}")
    print(f"Largest component sizes (top 5): {wc_sizes[:5]}")

    with open("btc_graph_wallet_only.gpickle", "wb") as f:
        pickle.dump(W, f)
    print("Saved wallet-only projection to btc_graph_wallet_only.gpickle")
