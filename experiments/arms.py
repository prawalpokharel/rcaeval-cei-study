"""Graph building and every arm of PLAN.md, scored from cached graphs.

The shipped heads are reproduced line for line from RCAEval/e2e so a cached
graph gives the same ranking as calling the shipped method end to end
(checked by `equality_check` in modal_app.py).
"""

import time

import networkx as nx
import numpy as np

from RCAEval.e2e import baro, nsigma
from RCAEval.graph_construction.granger import granger
from RCAEval.graph_heads.cei import (
    cei_pagerank,
    cei_rank,
    composite_centrality,
    granger_propagation_matrix,
    pc_propagation_matrix,
    series_entropy,
)
from RCAEval.graph_heads.page_rank import page_rank
from RCAEval.graph_heads.random_walk import random_walk
from RCAEval.io.time_series import preprocess

GRAPH_HEADS = ["pagerank", "randomwalk", "betweenness", "centrality", "entropy", "cei", "cei_pagerank"]
REFERENCES = ["baro", "nsigma", "random"]
ROUND = 12


def method_input(data, dataset):
    """What a graph method scores: preprocessed columns, time dropped."""
    return preprocess(data=data, dataset=dataset, dk_select_useful=False)


def build_pc(x):
    from causallearn.search.ConstraintBased.PC import pc

    return np.asarray(pc(x.to_numpy()).G.graph)  # same call as pc_pagerank


def build_granger(x):
    return np.asarray(granger(x))  # same call as granger_pagerank


def _ranked(names, scores):
    s = np.round(np.asarray(scores, dtype=float), ROUND)
    order = sorted(range(len(names)), key=lambda i: -s[i])
    return [names[i] for i in order]


def shipped_pagerank(kind, graph, names):
    if kind == "pc":
        from sknetwork.ranking import PageRank

        # pc_pagerank: rebuild the cause -> effect matrix, score its transpose
        adj = pc_propagation_matrix(graph).T
        scores = PageRank().fit_predict(adj.T)
        ranks = sorted(zip(names, scores), key=lambda x: x[1], reverse=True)
        return [x[0] for x in ranks]
    # granger_pagerank
    if graph.sum().sum() == 0:
        return list(names)
    ranks = sorted(page_rank(graph, node_names=list(names)), key=lambda x: x[1], reverse=True)
    return [x[0] for x in ranks]


def shipped_randomwalk(kind, graph, names, num_node):
    adj = graph if kind == "pc" else graph.astype(bool).astype(int)
    np.random.seed(0)  # random_walk draws its start node from the global RNG
    ranks = sorted(random_walk(adj, list(names), num_loop=num_node), key=lambda x: x[1], reverse=True)
    return [x[0] for x in ranks]


def propagation(kind, graph):
    return pc_propagation_matrix(graph) if kind == "pc" else granger_propagation_matrix(graph)


def graph_arms(kind, graph, x, num_node):
    """Rankings (metric names, best first) and seconds for every graph head."""
    names = x.columns.to_list()
    out, secs = {}, {}

    def timed(name, fn):
        t0 = time.perf_counter()
        out[name] = fn()
        secs[name] = time.perf_counter() - t0

    pr_input = propagation(kind, graph)
    timed("pagerank", lambda: shipped_pagerank(kind, graph, names))
    timed("randomwalk", lambda: shipped_randomwalk(kind, graph, names, num_node))

    def betweenness():
        G = nx.DiGraph()
        G.add_nodes_from(range(len(names)))
        r, c = np.nonzero(pr_input)
        G.add_edges_from((int(a), int(b)) for a, b in zip(r, c) if a != b)
        bw = nx.betweenness_centrality(G)
        return _ranked(names, [bw[i] for i in range(len(names))])

    timed("betweenness", betweenness)
    timed("centrality", lambda: _ranked(names, composite_centrality(pr_input)))
    timed("entropy", lambda: _ranked(names, series_entropy(x)))
    timed("cei", lambda: [n for n, _ in cei_rank(pr_input, names, x)])
    timed("cei_pagerank", lambda: [n for n, _ in cei_pagerank(pr_input, names, x)])
    edges = int(np.count_nonzero(pr_input))
    return out, secs, edges


def reference_arms(data, inject_time, dataset, seed):
    out, secs, errors = {}, {}, {}
    names = method_input(data, dataset).columns.to_list()
    for name, fn in (("baro", baro), ("nsigma", nsigma)):
        t0 = time.perf_counter()
        try:
            out[name] = list(fn(data, inject_time, dataset=dataset, dk_select_useful=False)["ranks"])
        except Exception as e:  # BARO and N-Sigma have no @rca wrapper; use its fallback, column order
            out[name] = list(names)
            errors[name] = f"{type(e).__name__}: {e}"
        secs[name] = time.perf_counter() - t0
    out["random"] = list(np.random.default_rng(seed).permutation(names))
    secs["random"] = 0.0
    return out, secs, errors
