"""
graph_view.py

Day 13 build (part 1) — builds a FOCUSED subgraph for the dashboard's link-
analysis view, colored by risk score. Never renders the whole ~2000-node
graph at once (established back on Day 4 that this produces an unreadable
blob) -- instead shows either:
  (a) the top-K riskiest wallets and their immediate 1-hop transaction
      neighborhood, or
  (b) a specific wallet's neighborhood, if the user searches for one.

Colored by risk score (green = low, red = high) using the SAME final_risk
column from Day 10/11's output -- so the visual matches the ranked table
exactly, not a separately-computed metric.
"""

from streamlit_agraph import Node, Edge

MAX_NODES = 150  # hard cap so the graph stays legible even if a wallet has a huge neighborhood


def risk_to_color(risk: float) -> str:
    """Green (low risk) -> red (high risk) linear interpolation."""
    risk = max(0.0, min(1.0, risk))
    r = int(40 + 190 * risk)
    g = int(180 - 150 * risk)
    b = 60
    return f"#{r:02x}{g:02x}{b:02x}"


def gather_neighborhood(W, seed_wallets: set) -> tuple:
    """Returns (included_wallets, included_tx) -- the seed wallets' immediate
    transactions, plus every OTHER wallet involved in those same transactions
    (so a transaction never shows up with only half its counterparties)."""
    included_wallets = set(seed_wallets)
    included_tx = set()

    for w in seed_wallets:
        if w not in W:
            continue
        for _, tx, d in W.out_edges(w, data=True):
            if d.get("relation") == "spends_in":
                included_tx.add(tx)
        for tx, _, d in W.in_edges(w, data=True):
            if d.get("relation") == "pays_to":
                included_tx.add(tx)

    for tx in list(included_tx):
        for u, _, d in W.in_edges(tx, data=True):
            if d.get("relation") == "spends_in":
                included_wallets.add(u)
        for _, v, d in W.out_edges(tx, data=True):
            if d.get("relation") == "pays_to":
                included_wallets.add(v)

    return included_wallets, included_tx


def build_subgraph(df, W, top_k: int = 12, center_wallet: str = None):
    """Returns (nodes, edges, node_info) for streamlit_agraph.agraph().
    node_info maps node id -> dict of details shown in the drill-down panel."""
    risk_lookup = dict(zip(df["wallet"], df["final_risk"]))
    explanation_lookup = dict(zip(df["wallet"], df["final_explanation"])) if "final_explanation" in df.columns else {}
    pattern_lookup = dict(zip(df["wallet"], df["pattern_types_involved"]))

    if center_wallet and center_wallet in W:
        seed_wallets = {center_wallet}
    else:
        top_wallets = df.sort_values("final_risk", ascending=False).head(top_k)["wallet"].tolist()
        seed_wallets = set(top_wallets)

    included_wallets, included_tx = gather_neighborhood(W, seed_wallets)

    # cap total size for legibility -- prioritize keeping the highest-risk wallets if truncating
    if len(included_wallets) + len(included_tx) > MAX_NODES:
        sorted_wallets = sorted(included_wallets, key=lambda w: -risk_lookup.get(w, 0.0))
        keep_wallets = set(sorted_wallets[:MAX_NODES // 2])
        keep_tx = set(list(included_tx)[:MAX_NODES // 2])
        included_wallets, included_tx = keep_wallets, keep_tx

    nodes, edges, node_info = [], [], {}

    for w in included_wallets:
        risk = risk_lookup.get(w, 0.0)
        is_seed = w in seed_wallets
        nodes.append(Node(
            id=w,
            label=w[:10] + "...",
            size=15 + 20 * risk + (8 if is_seed else 0),
            color=risk_to_color(risk),
            shape="dot",
            borderWidth=3 if is_seed else 1,
        ))
        node_info[w] = {
            "type": "wallet",
            "risk": risk,
            "explanation": explanation_lookup.get(w, "N/A"),
            "patterns": pattern_lookup.get(w, ""),
        }

    for tx in included_tx:
        tx_data = W.nodes[tx]
        nodes.append(Node(
            id=tx,
            label="TX:" + tx[:8],
            size=10,
            color="#999999" if tx_data.get("pattern_type") in (None, "normal") else "#333333",
            shape="diamond",
        ))
        node_info[tx] = {
            "type": "transaction",
            "pattern_type": tx_data.get("pattern_type"),
            "timestamp": tx_data.get("timestamp"),
            "fee": tx_data.get("fee"),
        }

    included_node_ids = included_wallets | included_tx
    for tx in included_tx:
        for u, _, d in W.in_edges(tx, data=True):
            if d.get("relation") == "spends_in" and u in included_node_ids:
                edges.append(Edge(source=u, target=tx, label="spends"))
        for _, v, d in W.out_edges(tx, data=True):
            if d.get("relation") == "pays_to" and v in included_node_ids:
                edges.append(Edge(source=tx, target=v, label="pays"))

    return nodes, edges, node_info
