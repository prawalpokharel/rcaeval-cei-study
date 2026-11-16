# RCAEval: CEI as a root-cause ranking head (pre-registration)

Written 15 Nov 2026, before any experiment ran. Design source: section 2 of
`_shared/CEI_Master_Plan.pdf` (page 9). Changes after this point go in the
deviation log at the end, with the reason and whether they came before or
after seeing any result.

## Question

Does a Centrality-Entropy Index (CEI) head, or CEI used as a prior that
re-weights PageRank, rank root-cause services better than the PageRank head
RCAEval already ships, on the same causal graphs?

## Code and data

- Upstream: `phamquiluan/RCAEval` main at `90b63a4` (6 Nov 2026, v1.12.0).
  Fork `prawalpokharel/RCAEval`, branch `cei-graph-head`.
- Data: the Hugging Face Parquet release `phamquiluan/RCAEval` (MIT licence,
  revision `afeacb11bcc94dadfd1c8f483ee4377b2b8b614e`). Only `metrics.parquet`,
  `inject_time.txt` and `cases.parquet` are downloaded. The card says the
  Parquet files are a lossless conversion of the Zenodo originals
  (CC BY 4.0, records 14590730 and 13305663).
- All 735 cases: RE1 (375: OB, SS, TT), RE2 (270), RE3 (90). The root-cause
  service is the `{service}` part of the case name (also `root_cause_service`
  in `cases.parquet`).

## Input to every method (identical to `main.py`)

For each case: read metrics, drop `time.1` and `*_latency-50`, replace ±inf
with NaN, forward-fill, fill the rest with 0, keep the last 600 rows before
`inject_time` and the first 600 rows after it (`--length 20`), rename
`*_latency-90` to `*_latency`, then `preprocess(dataset=...)`
(`drop_constant(drop_time(...))` and `convert_mem_mb`). The candidate count
for chance is the number of services in the preprocessed columns, as
`main.py` records it.

Data-path check, before any CEI result: BARO on RE2-TT must reproduce the
README's published Avg@5 (CPU 0.72, MEM 0.99, DISK 1.0, SOCKET 0.83,
DELAY 0.63, LOSS 0.64). If it does not, stop and fix the loader.

## Graphs (built once per case, cached)

- **PC**: `causallearn` `pc(data.to_numpy())` with its defaults, exactly as
  `pc_pagerank` calls it.
- **Granger**: `RCAEval.graph_construction.granger.granger(data)` (maxlag 3,
  p < 0.05), exactly as `granger_pagerank` calls it.
- Each build runs in its own process with a **20-minute timeout**. A timed-out
  or failed build counts as a failure for every head on that graph: the case
  gets the benchmark's own fallback ranking (unranked column order, what the
  `@rca` wrapper returns). Timeouts and failures are reported per dataset.

Every graph head works on the same "propagation" graph R the shipped PageRank
heads use, with edges pointing from effect to cause, so mass flows toward
root causes.

## Arms

Graph heads, run on both PC and Granger graphs:

| Arm | Definition |
|---|---|
| `pagerank` | The shipped head, reproduced exactly (`pc_pagerank` and `granger_pagerank` code paths). Checked rank-for-rank against the shipped methods on 10 cases. |
| `randomwalk` | The shipped `random_walk` head (`pc_randomwalk`, `granger_randomwalk` paths), `num_loop` as `main.py` passes it, global NumPy seed 0 before each case. |
| `betweenness` | Directed betweenness centrality on R alone (the master plan's Phase 0 comparator). |
| `centrality` | CEI's centrality term alone: 0.35 PageRank + 0.30 betweenness + 0.20 degree + 0.15 closeness on R, max-normalised (`_shared/cei_ref.py`). |
| `entropy` | CEI's entropy term alone: normalised Shannon entropy of each metric over the 1,200-row window, min-max scaled to [0, 1], 10 bins. No graph. |
| `cei` | 0.40·C + 0.35·H + 0.25·G with the canonical default weights. RCAEval has no declared criticality, so G = 0 for every node. The governance term is not invented. |
| `cei_pagerank` | The hybrid: personalised PageRank on R (damping 0.85) with the personalisation vector proportional to each node's CEI score. |

Non-graph references, run once per case: `baro` (shipped), `nsigma`
(shipped), and `random` (uniform permutation, seed = case index), which must
land at chance.

Ties: scores are rounded to 12 decimals, then a stable sort keeps column
order, so results do not depend on the container's CPU.

## Metrics

Primary: service-level AC@1, AC@3 and Avg@5, as `main.py` computes them with
`Evaluator` (service = metric prefix, `-db` folded in). Also chance Avg@5 and
lift over chance from the recorded candidate counts. Secondary: fine-grained
(service plus indicator) Avg@5 for the faults that have an indicator, graph
build time per case, and head runtime per case.

Breakdowns: by system (OB, SS, TT), suite (RE1, RE2, RE3) and fault type
(cpu, mem, disk, socket, delay, loss, f1 to f5).

## Statistics

The per-case outcome is the case's contribution to service Avg@5:
s = max(0, 6 − r) / 5, where r is the rank of the true service.

Primary tests, four in all: `cei` vs `pagerank` and `cei_pagerank` vs
`pagerank`, on each of PC and Granger, over all 735 cases. Each is a paired
Wilcoxon signed-rank test (two-sided, `zero_method="wilcox"`), Holm-corrected
across the four. The effect size is the mean difference in s with a 95%
bootstrap CI (10,000 resamples over cases, seed 0).

A head **wins** on a graph type when the CI excludes zero on the positive side
and the Holm-adjusted p is below 0.05. It **loses** when the CI excludes zero
on the negative side. Otherwise it is a **tie**, and it is reported as a tie.

Secondary comparisons (reported, not used in the rule): each head against
`betweenness` and against `baro`, and the ablations (`centrality`, `entropy`)
against `cei`. All use the same test and bootstrap, without multiplicity
correction, and are labelled as secondary.

## Decision rule (committed now)

1. If `cei` wins on at least one graph type and loses on neither, propose the
   CEI head (`pc_cei`, `granger_cei`) in an issue first.
2. Otherwise, if `cei_pagerank` wins on at least one graph type and loses on
   neither, propose the hybrid instead.
3. Otherwise, publish the null result as a notebook with the numbers and draft
   no PR.

Even under rule 1 or 2, the issue must state how the head compares with
`betweenness` and with `baro`. If BARO is better, say so.

## Pilot

RE1 first (375 cases), to measure graph-build time and check the loader:
exact rank equality with the shipped methods on 10 cases, and the BARO
reproduction. The pilot fixes compute settings only (container size,
timeout handling). Head definitions and the rule above do not change after
the pilot, except for bug fixes, which go in the log below.

## Compute

Modal, apps named `cei-rcaeval-*`, at most
20 containers at once, 1 CPU and 4 GiB per graph build. The cap is $30, and
spend is checked against the billing report after each stage.

## Deviation log

1. **15 Nov, 21:50, before any result was analysed.** Two RE1-OB cases have defective
   labels in the Hugging Face release: `re1ob_currencyservice_loss_1` has a truncated
   `inject_time.txt` (`16933142`, eight digits), so every row counts as after the
   fault, and `re1ob_productcatalogservice_cpu_3` has 63 rows that all fall before its
   inject time. BARO and N-Sigma have no `@rca` wrapper and crash on an empty window.
   Change: when a reference method raises, the case gets the benchmark's fallback
   (column order), the same rule as for a failed graph, and the error is recorded.
   The graph heads run normally on both cases. Both cases stay in the 735. A
   sensitivity analysis without them is added. No head, metric or test changed.
2. **15 Nov, 21:50, before any result was analysed.** The RE1 equality check (10 cases)
   matched for every shipped and new method, except the Granger heads on
   `re1ob_adservice_cpu_1`, where a Granger graph rebuilt on another container
   differed from the cached one. Rebuilt on a container of the same CPU type as the
   cached build, the graph matched exactly (466 edges, 0 differences), and the shipped
   head gave the end-to-end ranking from either graph (`evidence/inspect.json`). Granger
   edges at the p = 0.05 threshold can flip between CPUs with and without AVX-512. Every
   head on a case reads the same cached graph, so the paired comparisons are unaffected.
   One graph failure is a crash inside causal-learn's PC (`re1ss_payment_mem_3`, exit
   1). The shipped `pc_pagerank` would fall back to column order there too.
3. **15 Nov, 22:25, before any result was analysed.** Compared every HF case with the
   Zenodo file `main.py` reads (`evidence/source_compare.json`). RE2 and RE3: the
   method input is identical for 477 cases, and HF has fewer rows in the window for
   `re2ob_checkoutservice_cpu_2` and `re2ob_checkoutservice_mem_2`. RE1-OB: 122 of 125
   identical (the two defects above, and `re1ob_checkoutservice_cpu_4` missing 21
   rows). **RE1-SS and RE1-TT differ entirely**: HF has the curated service-level
   columns (60 and 221 raw columns), while `main.py` reads the Zenodo `data.csv` with
   438 and 1,242 columns, including node-exporter host metrics. Decision: keep the HF
   release as pre-registered for all 735 cases (identical graphs for every head, so
   the paired tests are valid). Do not compare RE1-SS or RE1-TT scores with numbers
   published from `main.py`. Report these discrepancies upstream as a data issue.
   The data-path check (BARO on RE2-TT) is unaffected, because RE2-TT is identical.
4. **16 Nov, 00:55, after the primary results.** Labelled exploratory, not part of the
   decision rule: (a) CEI's centrality term alone against the shipped PageRank and
   betweenness heads (`results/exploratory_tests.csv`). It was run because the
   secondary ablation showed the entropy term pulling CEI down. (b) The column-order
   fallback against the random arm on the 137 cases whose PC graph failed
   (`results/column_order_check.json`). (c) Whether the shipped random walk depends on
   the graph at all (`results/randomwalk_graph_independence.json`). It does not: the
   PC and Granger rankings are identical on all 579 cases with both graphs. Any
   follow-up on (a) needs its own pre-registered test on fresh data.
