"""Minimal reproduction: RCAEval's random_walk head returns the same ranking whatever the graph.

  .venv/bin/python experiments/randomwalk_demo.py

random_walk() is called without previous_scores by pc_randomwalk,
granger_randomwalk, lingam_randomwalk, fci_randomwalk and ntlr_randomwalk. It
then starts every node at Score(0), so generate_transition_matrix gives every
column zero weight and falls back to a uniform 1/size column. The walk is a
uniform random walk, and the ranking depends only on the node count and the
random state, never on the edges.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "repo"))
from RCAEval.graph_heads.random_walk import random_walk  # noqa: E402

names = [f"svc{i}_cpu" for i in range(6)]
chain = np.zeros((6, 6), dtype=int)  # causal-learn marks for 0 -> 1 -> 2 -> 3 -> 4 -> 5
for a in range(5):
    chain[a, a + 1], chain[a + 1, a] = -1, 1
star = np.zeros((6, 6), dtype=int)  # 5 -> every other node
for b in range(5):
    star[5, b], star[b, 5] = -1, 1
empty = np.zeros((6, 6), dtype=int)

out = {}
for label, adj in (("chain", chain), ("star", star), ("no edges", empty)):
    np.random.seed(0)
    out[label] = [n for n, _ in sorted(random_walk(adj, names, num_loop=60), key=lambda x: x[1], reverse=True)]
    print(f"{label:9s}", out[label])
print("identical rankings for all three graphs:", out["chain"] == out["star"] == out["no edges"])
