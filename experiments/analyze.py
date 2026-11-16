"""Analyse the exported scores exactly as PLAN.md pre-registers.

  .venv/bin/python experiments/analyze.py [results/raw/scores.json.gz]

Writes results/per_case.csv, arm_metrics.csv, primary_tests.csv,
secondary_tests.csv, graph_builds.csv, head_runtime.csv and summary.json.
"""

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "repo"))

from RCAEval.benchmark.evaluation import Evaluator  # noqa: E402
from RCAEval.benchmark.scoring import fault_to_indicator, indicator_ranks, service_ranks  # noqa: E402
from RCAEval.classes.graph import Node  # noqa: E402

RES = ROOT / "results"
B = 10_000
HEADS = ["pagerank", "randomwalk", "betweenness", "centrality", "entropy", "cei", "cei_pagerank"]
ARMS = [f"{k}:{h}" for k in ("pc", "granger") for h in HEADS] + ["baro", "nsigma", "random"]


def service_rank(ranks, service):
    names = [n.entity for n in service_ranks(ranks)]
    return names.index(service) + 1 if service in names else np.inf


def per_case_table(recs):
    rows = []
    for r in recs:
        for arm in ARMS:
            rk = service_rank(r["arms"][arm], r["service"])
            rows.append({"case": r["case"], "dataset": r["dataset"], "suite": r["suite"], "system": r["system"],
                         "fault": r["fault"], "arm": arm, "rank": rk,
                         "s": max(0.0, 6 - rk) / 5 if np.isfinite(rk) else 0.0})
    return pd.DataFrame(rows)


def evaluate(recs, arm, keep=lambda r: True):
    se, fe = Evaluator(max_k=5), Evaluator(max_k=5)
    n_fine = 0
    for r in recs:
        if not keep(r):
            continue
        ranks = r["arms"][arm]
        nc = r["n_candidates"]
        se.add_case(ranks=service_ranks(ranks), n_candidates=nc["service"], answer=Node(r["service"], "unknown"))
        ind = fault_to_indicator(r["fault"], r["dataset"])
        if ind is not None:
            fe.add_case(ranks=indicator_ranks(ranks), n_candidates=nc["metric"], answer=Node(r["service"], ind))
            n_fine += 1
    if se.num == 0:
        return None
    return {"n": se.num, "AC@1": se.accuracy_service(1), "AC@3": se.accuracy_service(3),
            "Avg@5": se.average_service(5), "chance_Avg@5": se.chance_average(5),
            "lift_Avg@5": se.average_service(5) - se.chance_average(5),
            "fine_n": n_fine, "fine_Avg@5": fe.average(5) if n_fine else None}


def boot_ci(d, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(B, len(d)))
    means = d[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired(pc, a, b):
    wide = pc[pc.arm.isin([a, b])].pivot(index="case", columns="arm", values="s")
    d = (wide[a] - wide[b]).to_numpy()
    lo, hi = boot_ci(d)
    nz = d[d != 0]
    p = float(wilcoxon(nz, zero_method="wilcox", alternative="two-sided").pvalue) if len(nz) else 1.0
    return {"a": a, "b": b, "n": len(d), "mean_a": float(wide[a].mean()), "mean_b": float(wide[b].mean()),
            "diff": float(d.mean()), "ci_lo": lo, "ci_hi": hi, "n_nonzero": int(len(nz)),
            "a_better": int((d > 0).sum()), "b_better": int((d < 0).sum()), "p": p}


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(ps) - rank) * ps[i])
        adj[i] = min(1.0, running)
    return adj


def verdict(t, p_adj=None):
    p = t["p"] if p_adj is None else p_adj
    if t["ci_lo"] > 0 and p < 0.05:
        return "win"
    if t["ci_hi"] < 0:
        return "lose"
    return "tie"


def main(path=RES / "raw" / "scores.json.gz"):
    blob = json.loads(gzip.decompress(Path(path).read_bytes()))
    recs = sorted(blob["scores"], key=lambda r: r["case"])
    pc = per_case_table(recs)
    pc.to_csv(RES / "per_case.csv", index=False)

    groups = [("all", lambda r: True)]
    groups += [(f"suite={s}", (lambda s: lambda r: r["suite"] == s)(s)) for s in ("RE1", "RE2", "RE3")]
    groups += [(f"system={s}", (lambda s: lambda r: r["system"] == s)(s)) for s in ("OB", "SS", "TT")]
    groups += [(f"fault={f}", (lambda f: lambda r: r["fault"] == f)(f)) for f in sorted({r["fault"] for r in recs})]
    rows = []
    for g, keep in groups:
        for arm in ARMS:
            m = evaluate(recs, arm, keep)
            if m:
                rows.append({"group": g, "arm": arm, **m})
    am = pd.DataFrame(rows)
    am.to_csv(RES / "arm_metrics.csv", index=False)

    primary = [paired(pc, f"{k}:{h}", f"{k}:pagerank") for k in ("pc", "granger") for h in ("cei", "cei_pagerank")]
    adj = holm(np.array([t["p"] for t in primary]))
    for t, pa in zip(primary, adj):
        t["p_holm"] = float(pa)
        t["verdict"] = verdict(t, pa)
    pd.DataFrame(primary).to_csv(RES / "primary_tests.csv", index=False)

    secondary = []
    for k in ("pc", "granger"):
        for h in ("cei", "cei_pagerank", "pagerank"):
            secondary.append(paired(pc, f"{k}:{h}", f"{k}:betweenness"))
            secondary.append(paired(pc, f"{k}:{h}", "baro"))
        secondary.append(paired(pc, f"{k}:centrality", f"{k}:cei"))
        secondary.append(paired(pc, f"{k}:entropy", f"{k}:cei"))
        secondary.append(paired(pc, f"{k}:randomwalk", "random"))
    for t in secondary:
        t["verdict"] = verdict(t)
    pd.DataFrame(secondary).to_csv(RES / "secondary_tests.csv", index=False)

    gb = []
    for kind, metas in blob["graph_meta"].items():
        for m in metas:
            from_case = next((r for r in recs if r["case"] == m["case"]), None)
            gb.append({"kind": kind, "case": m["case"], "status": m["status"], "seconds": m["seconds"],
                       "avx512f": m.get("avx512f"), "dataset": from_case["dataset"] if from_case else None,
                       "n_columns": from_case["n_columns"] if from_case else None,
                       "edges": from_case["graphs"][kind].get("edges") if from_case else None})
    gb = pd.DataFrame(gb)
    gb.to_csv(RES / "graph_builds.csv", index=False)

    rt = pd.DataFrame([{"arm": a, "case": r["case"], "seconds": r["seconds"][a]} for r in recs for a in ARMS])
    rt.groupby("arm").seconds.agg(["mean", "median", "max"]).to_csv(RES / "head_runtime.csv")

    # data-path check: BARO on RE2-TT against the README's published Avg@5 per fault
    published = {"cpu": 0.72, "mem": 0.99, "disk": 1.0, "socket": 0.83, "delay": 0.63, "loss": 0.64}
    repro = []
    for fault, pub in published.items():
        m = evaluate(recs, "baro", lambda r, fault=fault: r["dataset"] == "re2-tt" and r["fault"] == fault)
        repro.append({"fault": fault, "n": m["n"], "ours": m["Avg@5"], "published": pub,
                      "match_2dp": round(m["Avg@5"], 2) == pub})
    pd.DataFrame(repro).to_csv(RES / "baro_reproduction.csv", index=False)

    # sensitivity: the primary tests on subsets named in PLAN.md's deviation log
    defects = {"re1ob_currencyservice_loss_1", "re1ob_productcatalogservice_cpu_3", "re1ob_checkoutservice_cpu_4",
               "re2ob_checkoutservice_cpu_2", "re2ob_checkoutservice_mem_2"}
    ok = {k: {r["case"] for r in recs if r["graphs"][k]["status"] == "ok"} for k in ("pc", "granger")}
    subsets = {"all 735": set(pc.case),
               "without the 5 HF-defective cases": set(pc.case) - defects,
               "without RE1-SS and RE1-TT": {r["case"] for r in recs if r["dataset"] not in ("re1-ss", "re1-tt")},
               "graph built (per graph type)": None}
    sens = []
    for name, keep in subsets.items():
        for k in ("pc", "granger"):
            cases = ok[k] if keep is None else keep
            sub = pc[pc.case.isin(cases)]
            for h in ("cei", "cei_pagerank"):
                t = paired(sub, f"{k}:{h}", f"{k}:pagerank")
                t["verdict"] = verdict(t)
                sens.append({"subset": name, **t})
    pd.DataFrame(sens).to_csv(RES / "sensitivity_tests.csv", index=False)

    ref_err = {a: sum(1 for r in recs if a in r.get("reference_errors", {})) for a in ("baro", "nsigma")}

    v = {(t["a"].split(":")[0], t["a"].split(":")[1]): t["verdict"] for t in primary}
    cei_ok = any(v[(k, "cei")] == "win" for k in ("pc", "granger")) and not any(
        v[(k, "cei")] == "lose" for k in ("pc", "granger"))
    hyb_ok = any(v[(k, "cei_pagerank")] == "win" for k in ("pc", "granger")) and not any(
        v[(k, "cei_pagerank")] == "lose" for k in ("pc", "granger"))
    decision = ("propose the CEI head" if cei_ok else
                "propose the CEI x PageRank hybrid" if hyb_ok else
                "null result: notebook only, no PR")
    summary = {"cases": len(recs), "decision": decision, "primary": primary, "baro_reproduction": repro,
               "reference_errors": ref_err,
               "graph_status": gb.groupby(["kind", "status"]).size().to_dict().__repr__(),
               "overall": am[am.group == "all"].set_index("arm")[["AC@1", "AC@3", "Avg@5", "lift_Avg@5"]].to_dict("index")}
    (RES / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({"decision": decision, "primary": [{k: t[k] for k in ("a", "diff", "ci_lo", "ci_hi", "p_holm",
                                                                             "verdict")} for t in primary]}, indent=1))
    print(am[am.group == "all"][["arm", "AC@1", "AC@3", "Avg@5", "lift_Avg@5"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:])
