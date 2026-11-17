# A Centrality-Entropy Index head for RCAEval: a pre-registered null result

A pre-registered test of the Centrality-Entropy Index (CEI) as a graph head in [RCAEval](https://github.com/phamquiluan/RCAEval), the root-cause-analysis benchmark of 735 microservice failure cases (WWW 2025). The result is null, so as the plan set out before any run, it's published here as a notebook.

## Result

Service-level Avg@5 on all 735 cases, with the same cached causal graph for every head. PC failed on 137 cases (107 timeouts, 30 crashes) and Granger on 41; on those cases every head gets RCAEval's column-order fallback.

| Head | PC graph | Granger graph |
|---|---|---|
| PageRank (shipped by RCAEval) | 0.253 | 0.304 |
| CEI | 0.156 | 0.240 |
| CEI × PageRank (personalised PageRank with a CEI prior) | 0.238 | 0.302 |
| CEI's centrality term alone (exploratory) | 0.225 | 0.365 |
| CEI's entropy term alone | 0.175 | 0.188 |
| BARO (no graph, for reference) | 0.829 | 0.829 |
| Random ranking | 0.251 | 0.251 |

Pre-registered tests (paired over cases, 95% bootstrap CI, Wilcoxon with Holm correction):

- CEI vs PageRank: −0.097 [−0.120, −0.075] on PC and −0.065 [−0.092, −0.038] on Granger, a loss on both.
- CEI × PageRank vs PageRank: −0.015 [−0.024, −0.005] on PC (a loss) and −0.002 [−0.009, +0.004] on Granger (a tie).

The committed rule (`PLAN.md`): propose CEI only if it wins on one graph type and loses on neither. Neither head qualifies, so there is no CEI pull request.

Why CEI fails here: its entropy term measures how spread a metric's values are over the window, which does not tell you whether the metric's service caused the fault. Ranked alone, entropy falls below random. Without it, the centrality term beats full CEI on both graphs. In an exploratory check, added after the primary results, it also beats PageRank on Granger graphs (+0.060 [+0.032, +0.089]) but loses on PC graphs. I'd treat that as a hypothesis for a new pre-registered test. Every graph head sits far below BARO, which ranks metrics by how much they moved after the fault.

## Found along the way

1. RCAEval's random-walk head ignores the graph: `pc_randomwalk` and `granger_randomwalk` give identical rankings on all 579 cases where both graphs exist, because `random_walk()` starts every node at score 0 and every transition falls back to uniform (`experiments/randomwalk_demo.py`). Reported as [phamquiluan/RCAEval#96](https://github.com/phamquiluan/RCAEval/issues/96).
2. The Hugging Face release differs from the Zenodo cases `main.py` reads: one `inject_time.txt` is truncated, four Online Boutique and three RE1-SS/RE1-TT cases are missing rows in the window, and RE1-SS and RE1-TT share no column names with the Zenodo files (`evidence/source_compare.json`). 355 of the 360 RE2 and RE3 cases match exactly. Reported as [phamquiluan/RCAEval#97](https://github.com/phamquiluan/RCAEval/issues/97).

## Checks

- The loader matches RCAEval's published BARO numbers on RE2-TT to the two decimals the README reports (`results/baro_reproduction.csv`).
- The cached-graph heads match RCAEval's end-to-end methods in 85 of 88 comparisons on 10 RE1 cases. The 3 differences come from one Granger graph (`re1ob_adservice_cpu_1`) that changed when rebuilt on a different CPU type and matched when rebuilt on the same type (`results/equality_summary.json`).
- The decision holds without the defective cases, without RE1-SS and RE1-TT, and on only the cases whose graph was built (`results/sensitivity_tests.csv`). One verdict moves: without RE1-SS and RE1-TT the hybrid on Granger graphs goes from a tie to a narrow loss.

## Layout and reproduction

- `PLAN.md`: the pre-registration, with a dated deviation log.
- `experiments/`: the Modal jobs (`modal_app.py`), the loader that mirrors `main.py` (`rca_input.py`), every arm (`arms.py`), the analysis (`analyze.py`).
- `results/`: every number above; `results/raw/scores.json.gz` holds each arm's full ranking per case.
- `notebook/cei_rcaeval_null_result.ipynb`: rebuilds the tables from `results/`.

The CEI heads are on the fork branch [`prawalpokharel/RCAEval@cei-graph-head`](https://github.com/prawalpokharel/RCAEval/tree/cei-graph-head) (`RCAEval/graph_heads/cei.py`, `RCAEval/e2e/pc_cei.py`, `RCAEval/e2e/granger_cei.py`, `tests/test_cei_heads.py`). To rerun:

```bash
git clone -b cei-graph-head https://github.com/prawalpokharel/RCAEval repo
modal run experiments/modal_app.py --stage download
modal run --detach experiments/modal_app.py --stage pipeline --suite ALL
modal run experiments/modal_app.py --stage analyze
```

The study cost about $8 on Modal (CPU only). Data comes from the RCAEval Hugging Face release (MIT, revision `afeacb1`) and the Zenodo records 13305663 and 14590730 (CC BY 4.0). Graph building uses RCAEval's own PC (causal-learn) and Granger (statsmodels) code.

## CEI reference

P. Pokharel, "Evaluation of the CEI Framework: Enterprise Cloud Governance in Multi-Tier Distributed Environments", Future Technologies Conference (FTC) 2026.
