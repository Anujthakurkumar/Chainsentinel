"""
union_find.py

Day 5 build — Union-Find (disjoint-set) with path compression and union-by-rank.
This is the algorithmic core of CIOH clustering: every time we see two wallet
addresses appear as inputs in the SAME transaction, we union() them, on the
assumption that they're controlled by the same real-world entity. At the end,
find() tells us which cluster (entity) each wallet ended up in.

Two optimizations, both standard and worth understanding, not just using:
  - Path compression (in find): after finding the root, point every node
    along the path directly at the root, so future lookups are near-instant.
  - Union by rank (in union): always attach the shorter tree under the root
    of the taller tree, keeping trees shallow instead of degenerating into
    a long chain.
Together these give near-O(1) amortized find/union, which matters once
you're running this over thousands of wallets/transactions.
"""

from collections import defaultdict


class UnionFind:
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, x):
        """Returns the representative (root) of x's set, path-compressing as it goes."""
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            return x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # path compression
        return self.parent[x]

    def union(self, x, y):
        """Merges the sets containing x and y. No-op if already in the same set."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1

    def groups(self) -> dict:
        """Returns {root: [members]} for every known element."""
        g = defaultdict(list)
        for x in self.parent:
            g[self.find(x)].append(x)
        return dict(g)


if __name__ == "__main__":
    # Tiny sanity check: three wallets seen together in one tx, one wallet alone
    uf = UnionFind()
    uf.union("wallet_A", "wallet_B")
    uf.union("wallet_B", "wallet_C")
    uf.find("wallet_D")  # never unioned with anything -> stays its own cluster

    print("Clusters:", uf.groups())
    assert uf.find("wallet_A") == uf.find("wallet_C"), "A and C should be in the same cluster"
    assert uf.find("wallet_D") != uf.find("wallet_A"), "D should be isolated"
    print("Sanity check passed.")
