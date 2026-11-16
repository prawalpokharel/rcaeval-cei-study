"""Modal jobs for the RCAEval CEI study (see ../PLAN.md).

Stages, each idempotent (finished work on the volume is skipped):

  download   fetch metrics.parquet + inject_time.txt for all 735 cases (HF, pinned revision)
  ci         run RCAEval's default-environment tests plus the new CEI tests
  graphs     build and cache the PC and Granger graph per case (20-min timeout each)
  score      score every arm from the cached graphs, one JSON per case
  equality   compare cached-graph arms with the shipped and new e2e methods on N cases
  export     bundle every score JSON into one file for local analysis
  status     count graphs and scores on the volume

  modal run --detach experiments/modal_app.py --stage graphs --suite RE1
"""

import json
import os
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
HF_REPO = "phamquiluan/RCAEval"
HF_REVISION = "afeacb11bcc94dadfd1c8f483ee4377b2b8b614e"
BUILD_TIMEOUT_S = 20 * 60
# container cap per app; the workspace is shared with other projects (about 100 containers in all)
MAXC = int(os.environ.get("RCA_MAXC", "15"))  # fair-share cap for this project
VOL = "/vol"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy==2.5.5", "pandas==3.1.1", "scipy==1.18.1", "scikit-learn==1.9.1", "networkx==3.7",
        "statsmodels==0.14.6", "causal-learn==0.1.4.8", "scikit-network==0.33.5", "pyarrow==26.0.1",
        "tigramite==5.2.10.1", "pingouin==0.7.0", "matplotlib==3.11.2", "tqdm==4.70.1",
        "requests==2.34.2", "toml==0.10.2", "pytest==9.2.0", "huggingface_hub",
    )
    .env({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
          "PYTHONPATH": "/root/repo:/root/experiments", "PYTHONHASHSEED": "0"})
    .add_local_dir(ROOT / "repo", "/root/repo", ignore=[".git", ".git/**", "data/**", "output/**", "**/__pycache__/**"])
    .add_local_dir(HERE, "/root/experiments", ignore=["**/__pycache__/**"])
)
app = modal.App("cei-rcaeval", image=image)
vol = modal.Volume.from_name("cei-rcaeval", create_if_missing=True)


def _cpu_flags():
    try:
        text = open("/proc/cpuinfo").read()
        return {"avx512f": " avx512f" in text, "model": next((l.split(":", 1)[1].strip() for l in text.splitlines()
                                                              if l.startswith("model name")), "")}
    except OSError:
        return {}


@app.function(volumes={VOL: vol}, cpu=2, memory=4096, timeout=3 * 3600)
def download():
    import hashlib

    from huggingface_hub import snapshot_download

    t0 = time.time()
    snapshot_download(HF_REPO, repo_type="dataset", revision=HF_REVISION, local_dir=f"{VOL}/data",
                      allow_patterns=["cases.parquet", "*/metrics.parquet", "*/inject_time.txt"], max_workers=8)
    manifest = {}
    for p in sorted(Path(f"{VOL}/data").glob("*/metrics.parquet")):
        manifest[p.parent.name] = {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    Path(f"{VOL}/data/manifest.json").write_text(json.dumps(manifest, indent=0))
    vol.commit()
    return {"cases": len(manifest), "bytes": sum(v["bytes"] for v in manifest.values()), "seconds": time.time() - t0}


@app.function(cpu=2, memory=4096, timeout=1800)
def ci():
    import subprocess

    tests = ["tests/test.py", "tests/test_evaluator.py", "tests/test_graph_methods.py", "tests/test_sli.py",
             "tests/test_version_gates.py", "tests/test_cei_heads.py"]
    tests += [str(p.relative_to("/root/repo")) for p in Path("/root/repo/tests").glob("test_parser*.py")]
    r = subprocess.run(["python", "-m", "pytest", "-q", "-p", "no:cacheprovider", *tests], cwd="/root/repo",
                       capture_output=True, text=True)
    return {"returncode": r.returncode, "tail": (r.stdout + r.stderr)[-6000:]}


def _child_build(kind, case, out_path):
    import numpy as np

    from arms import build_granger, build_pc, method_input
    from rca_input import load_window, parse_case

    data, _, _ = load_window(f"{VOL}/data/{case}")
    x = method_input(data, parse_case(case)["dataset"])
    g = build_pc(x) if kind == "pc" else build_granger(x)
    np.save(out_path, g)


@app.function(volumes={VOL: vol}, cpu=1.0, memory=4096, timeout=35 * 60, max_containers=MAXC, retries=1)
def build(case: str, kind: str):
    import multiprocessing as mp
    import shutil
    import tempfile

    out = Path(f"{VOL}/graphs/{kind}/{case}.npy")
    meta_p = Path(f"{VOL}/graphs/{kind}/{case}.json")
    if meta_p.exists():
        return json.loads(meta_p.read_text())
    tmp = Path(tempfile.mkdtemp()) / "g.npy"
    t0 = time.time()
    proc = mp.get_context("fork").Process(target=_child_build, args=(kind, case, str(tmp)))
    proc.start()
    proc.join(BUILD_TIMEOUT_S)
    if proc.is_alive():
        proc.kill()
        proc.join()
        status = "timeout"
    else:
        status = "ok" if proc.exitcode == 0 and tmp.exists() else f"error(exit={proc.exitcode})"
    meta = {"case": case, "kind": kind, "status": status, "seconds": round(time.time() - t0, 2), **_cpu_flags()}
    out.parent.mkdir(parents=True, exist_ok=True)
    if status == "ok":
        shutil.copy(tmp, out)
    meta_p.write_text(json.dumps(meta))
    vol.commit()
    return meta


@app.function(volumes={VOL: vol}, cpu=1.0, memory=4096, timeout=30 * 60, max_containers=MAXC, retries=1)
def score(case: str, seed: int):
    import numpy as np

    from arms import GRAPH_HEADS, graph_arms, method_input, reference_arms
    from rca_input import candidate_counts, load_window, parse_case

    out_p = Path(f"{VOL}/scores/{case}.json")
    if out_p.exists():
        return {"case": case, "skipped": True}
    vol.reload()
    info = parse_case(case)
    data, inject_time, num_node = load_window(f"{VOL}/data/{case}")
    x = method_input(data, info["dataset"])
    rec = {"case": case, **info, "n_candidates": candidate_counts(data, info["dataset"]),
           "n_columns": x.shape[1], "num_node": num_node, "arms": {}, "seconds": {}, "graphs": {}, **_cpu_flags()}
    for kind in ("pc", "granger"):
        meta_p = Path(f"{VOL}/graphs/{kind}/{case}.json")
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else {"status": "missing"}
        if meta["status"] == "ok":
            g = np.load(f"{VOL}/graphs/{kind}/{case}.npy", allow_pickle=False)
            ranks, secs, edges = graph_arms(kind, g, x, num_node)
            meta["edges"] = edges
        else:
            # the benchmark's own fallback when a method fails: unranked column order
            ranks = {h: x.columns.to_list() for h in GRAPH_HEADS}
            secs = {h: 0.0 for h in GRAPH_HEADS}
        rec["graphs"][kind] = meta
        for h in GRAPH_HEADS:
            rec["arms"][f"{kind}:{h}"] = ranks[h]
            rec["seconds"][f"{kind}:{h}"] = secs[h]
    ranks, secs, errors = reference_arms(data, inject_time, info["dataset"], seed)
    rec["arms"].update(ranks)
    rec["seconds"].update(secs)
    rec["reference_errors"] = errors
    rec["window_rows"] = {"normal": int((data["time"] < inject_time).sum()),
                          "anomalous": int((data["time"] >= inject_time).sum())}
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(rec))
    vol.commit()
    return {"case": case, "skipped": False}


@app.function(volumes={VOL: vol}, cpu=1.0, memory=4096, timeout=3 * 3600, max_containers=10)
def equality(case: str):
    """Shipped and new e2e methods, end to end, against the cached-graph arms."""
    import numpy as np

    import RCAEval.e2e as e2e
    from rca_input import load_window, parse_case

    vol.reload()
    rec = json.loads(Path(f"{VOL}/scores/{case}.json").read_text())
    data, inject_time, num_node = load_window(f"{VOL}/data/{case}")
    ds = parse_case(case)["dataset"]
    pairs = {"pc_pagerank": "pc:pagerank", "granger_pagerank": "granger:pagerank",
             "pc_randomwalk": "pc:randomwalk", "granger_randomwalk": "granger:randomwalk",
             "pc_cei": "pc:cei", "pc_cei_pagerank": "pc:cei_pagerank",
             "granger_cei": "granger:cei", "granger_cei_pagerank": "granger:cei_pagerank",
             "baro": "baro", "nsigma": "nsigma"}
    res = {}
    for fn, arm in pairs.items():
        if arm.startswith(("pc:", "granger:")) and rec["graphs"][arm.split(":")[0]]["status"] != "ok":
            res[fn] = "graph not built"
            continue
        np.random.seed(0)
        t0 = time.time()
        ranks = list(getattr(e2e, fn)(data, inject_time, dataset=ds, anomalies=None, dk_select_useful=False,
                                      sli=None, verbose=False, n_iter=num_node)["ranks"])
        res[fn] = {"equal": ranks == rec["arms"][arm], "seconds": round(time.time() - t0, 1)}
    return {"case": case, "results": res}


@app.function(volumes={VOL: vol}, cpu=2, memory=8192, timeout=1800)
def export():
    import gzip

    vol.reload()
    recs = [json.loads(p.read_text()) for p in sorted(Path(f"{VOL}/scores").glob("*.json"))]
    metas = {kind: [json.loads(p.read_text()) for p in sorted(Path(f"{VOL}/graphs/{kind}").glob("*.json"))]
             for kind in ("pc", "granger")}
    manifest = json.loads(Path(f"{VOL}/data/manifest.json").read_text())
    return gzip.compress(json.dumps({"scores": recs, "graph_meta": metas, "manifest": manifest}).encode())


@app.function(volumes={VOL: vol}, cpu=0.25, memory=1024, timeout=24 * 3600)
def drive(cases: list, seeds: dict, kinds: list):
    """Server-side driver: build graphs, then score, so no local client has to stay open."""
    t0 = time.time()
    built = {}
    jobs = [(c, k) for c in cases for k in kinds]
    for m in build.starmap(jobs, return_exceptions=True, order_outputs=False):
        key = m.get("status", "exception") if isinstance(m, dict) else f"exception: {m}"[:120]
        built[key] = built.get(key, 0) + 1
    scored = 0
    for r in score.starmap([(c, seeds[c]) for c in cases], return_exceptions=True, order_outputs=False):
        scored += isinstance(r, dict)
    log = {"cases": len(cases), "built": built, "scored": scored, "hours": round((time.time() - t0) / 3600, 2)}
    Path(f"{VOL}/runs").mkdir(parents=True, exist_ok=True)
    Path(f"{VOL}/runs/{int(time.time())}.json").write_text(json.dumps(log))
    vol.commit()
    return log


@app.function(volumes={VOL: vol}, cpu=4, memory=8192, timeout=3600)
def analyze_remote():
    """Run experiments/analyze.py on the volume's scores; return every results file it writes."""
    import gzip
    import subprocess

    blob = export.local()
    res = Path("/root/results")
    (res / "raw").mkdir(parents=True, exist_ok=True)
    (res / "raw" / "scores.json.gz").write_bytes(blob)
    r = subprocess.run(["python", "/root/experiments/analyze.py"], capture_output=True, text=True, cwd="/root")
    files = {p.name: p.read_bytes() for p in res.glob("*") if p.is_file()}
    files["analyze.log"] = (r.stdout + r.stderr).encode()
    files["raw_scores.json.gz"] = blob
    return files


@app.function(volumes={VOL: vol}, cpu=1.0, memory=4096, timeout=3600)
def inspect(case: str):
    """Graph metas for a case, and how a fresh Granger graph compares with the cached one on this CPU."""
    import numpy as np

    from arms import build_granger, method_input, shipped_pagerank
    from rca_input import load_window, parse_case
    import RCAEval.e2e as e2e

    vol.reload()
    out = {"cpu": _cpu_flags(), "metas": {}}
    for kind in ("pc", "granger"):
        p = Path(f"{VOL}/graphs/{kind}/{case}.json")
        out["metas"][kind] = json.loads(p.read_text()) if p.exists() else None
    data, inject_time, num_node = load_window(f"{VOL}/data/{case}")
    ds = parse_case(case)["dataset"]
    x = method_input(data, ds)
    cached_p = Path(f"{VOL}/graphs/granger/{case}.npy")
    if cached_p.exists():
        cached = np.load(cached_p)
        fresh = build_granger(x)
        out["granger_edges_cached"] = int(cached.sum())
        out["granger_edges_fresh"] = int(fresh.sum())
        out["granger_entries_differ"] = int((cached != fresh).sum())
        e2e_ranks = list(e2e.granger_pagerank(data, inject_time, dataset=ds)["ranks"])
        out["head_on_fresh_equals_e2e"] = shipped_pagerank("granger", fresh, x.columns.to_list()) == e2e_ranks
        out["head_on_cached_equals_e2e"] = shipped_pagerank("granger", cached, x.columns.to_list()) == e2e_ranks
    return out


@app.function(cpu=1.0, memory=2048, timeout=1800)
def zenodo_check(cases: list):
    """Inject time and row split of the given RE1-OB cases in both Zenodo releases and on HF."""
    import io
    import urllib.request
    import zipfile

    import pandas as pd

    urls = {"zenodo-13305663/online-boutique.zip": "https://zenodo.org/records/13305663/files/online-boutique.zip?download=1",
            "zenodo-14590730/RE1-OB.zip": "https://zenodo.org/records/14590730/files/RE1-OB.zip?download=1"}
    out = {}
    for name, url in urls.items():
        z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(url, timeout=600).read()))
        names = z.namelist()
        for case in cases:
            _, svc, fault, rep = case.split("_")
            hits = [n for n in names if f"{svc}_{fault}/{rep}/" in n]
            row = {"files": sorted({h.rsplit('/', 1)[-1] for h in hits})}
            it_name = next((h for h in hits if h.endswith("inject_time.txt")), None)
            csv_name = next((h for h in hits if h.endswith("data.csv")), None)
            if it_name:
                row["inject_time_raw"] = z.read(it_name).decode().strip()
            if csv_name and it_name:
                d = pd.read_csv(io.BytesIO(z.read(csv_name)))
                it = int(row["inject_time_raw"].split()[0])
                row.update(rows=len(d), tmin=int(d["time"].min()), tmax=int(d["time"].max()),
                           before=int((d["time"] < it).sum()), after=int((d["time"] >= it).sum()))
            out[f"{name}:{case}"] = row
    return out


ZENODO = {
    "re1-ob": "https://zenodo.org/records/13305663/files/online-boutique.zip?download=1",
    "re1-ss": "https://zenodo.org/records/13305663/files/sock-shop-2.zip?download=1",
    "re1-tt": "https://zenodo.org/records/13305663/files/train-ticket.zip?download=1",
    "re2-ob": "https://zenodo.org/records/14590730/files/RE2-OB.zip?download=1",
    "re2-ss": "https://zenodo.org/records/14590730/files/RE2-SS.zip?download=1",
    "re2-tt": "https://zenodo.org/records/14590730/files/RE2-TT.zip?download=1",
    "re3-ob": "https://zenodo.org/records/14590730/files/RE3-OB.zip?download=1",
    "re3-ss": "https://zenodo.org/records/14590730/files/RE3-SS.zip?download=1",
    "re3-tt": "https://zenodo.org/records/14590730/files/RE3-TT.zip?download=1",
}


@app.function(volumes={VOL: vol}, cpu=1.0, memory=8192, timeout=3 * 3600, max_containers=2)
def compare_source(dataset: str):
    """Per case: does the Zenodo file main.py reads give the same method input as the HF Parquet case?

    Keeps the Zenodo input on the volume (zenodo/<case>/data.csv, inject_time.txt) for reuse.
    """
    import shutil
    import urllib.request
    import zipfile

    import numpy as np
    import pandas as pd

    from arms import method_input
    from rca_input import load_window

    zpath = Path(f"/tmp/{dataset}.zip")
    with urllib.request.urlopen(ZENODO[dataset], timeout=1800) as r, open(zpath, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 22)
    z = zipfile.ZipFile(zpath)
    prefix = dataset.replace("-", "")
    by_case = {}
    for n in z.namelist():
        parts = n.strip("/").split("/")
        if len(parts) >= 3 and parts[-1] in ("data.csv", "simple_metrics.csv", "inject_time.txt"):
            svc_fault, rep = parts[-3], parts[-2]
            by_case.setdefault(f"{prefix}_{svc_fault}_{rep}", {})[parts[-1]] = n
    out = {}
    for case, files in sorted(by_case.items()):
        row = {"zenodo_files": sorted(files)}
        metric_file = files.get("data.csv") or files.get("simple_metrics.csv")
        if metric_file is None or "inject_time.txt" not in files or not Path(f"{VOL}/data/{case}").exists():
            row["status"] = "unmatched"
            out[case] = row
            continue
        dest = Path(f"{VOL}/zenodo/{case}")
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "data.csv").write_bytes(z.read(metric_file))
        (dest / "inject_time.txt").write_bytes(z.read(files["inject_time.txt"]))
        hf, hf_it, _ = load_window(f"{VOL}/data/{case}")
        zd, z_it, _ = load_window(str(dest), csv=True)
        hx, zx = method_input(hf, dataset), method_input(zd, dataset)
        common = [c for c in hx.columns if c in zx.columns]
        row.update(zenodo_file=Path(metric_file).name, inject_hf=hf_it, inject_zenodo=z_it,
                   rows_hf=len(hf), rows_zenodo=len(zd), cols_hf=hx.shape[1], cols_zenodo=zx.shape[1],
                   cols_common=len(common), cols_only_hf=len(set(hx.columns) - set(zx.columns)),
                   cols_only_zenodo=len(set(zx.columns) - set(hx.columns)))
        if common and len(hx) == len(zx):
            row["max_abs_diff"] = float(np.nanmax(np.abs(hx[common].to_numpy() - zx[common].to_numpy())))
        row["identical_input"] = bool(hf_it == z_it and len(hx) == len(zx) and row["cols_only_hf"] == 0
                                      and row["cols_only_zenodo"] == 0 and row.get("max_abs_diff", 1) == 0.0)
        out[case] = row
    vol.commit()
    return out


@app.function(volumes={VOL: vol}, timeout=600)
def status():
    vol.reload()
    out = {}
    for kind in ("pc", "granger"):
        metas = [json.loads(p.read_text()) for p in Path(f"{VOL}/graphs/{kind}").glob("*.json")] \
            if Path(f"{VOL}/graphs/{kind}").exists() else []
        out[kind] = {}
        for m in metas:
            out[kind][m["status"]] = out[kind].get(m["status"], 0) + 1
    out["scores"] = len(list(Path(f"{VOL}/scores").glob("*.json"))) if Path(f"{VOL}/scores").exists() else 0
    return out


def _cases(suite):
    # sorted case names from cases.parquet (HF revision above), saved as JSON so the launcher needs no parquet reader
    cases = json.loads((ROOT / "data_meta" / "cases.json").read_text())
    if suite != "ALL":
        prefixes = tuple(x.strip().upper() for x in suite.split(","))
        cases = [c for c in cases if c.upper().startswith(prefixes)]
    return cases


def _log_app(stage, suite):
    line = f"{time.strftime('%Y-%m-%d %H:%M')}\t{app.app_id}\tcei-rcaeval\tstage={stage} suite={suite}\n"
    with open(ROOT / "evidence" / "modal_apps.txt", "a") as f:
        f.write(line)
    print(line.strip())


@app.local_entrypoint()
def main(stage: str = "status", suite: str = "RE1", n: int = 10, kinds: str = "pc,granger"):
    _log_app(stage, suite)
    if stage == "download":
        print(download.remote())
    elif stage == "ci":
        r = ci.remote()
        (ROOT / "evidence" / "modal_ci.txt").write_text(r["tail"])
        print(r["returncode"], r["tail"][-1500:])
    elif stage == "graphs":
        jobs = [(c, k) for c in _cases(suite) for k in kinds.split(",")]
        done = {}
        for m in build.starmap(jobs, return_exceptions=True, order_outputs=False):
            key = m.get("status", "exception") if isinstance(m, dict) else "exception"
            done[key] = done.get(key, 0) + 1
        print(done)
    elif stage == "pipeline":
        # spawn and return; run with `modal run --detach` so the app outlives this client
        all_cases = _cases("ALL")
        cases = _cases(suite)
        call = drive.spawn(cases, {c: all_cases.index(c) for c in cases}, kinds.split(","))
        print(f"spawned drive for {len(cases)} cases: {call.object_id}")
    elif stage == "score":
        all_cases = _cases("ALL")
        cases = _cases(suite)
        args = [(c, all_cases.index(c)) for c in cases]
        n_ok = sum(1 for r in score.starmap(args, return_exceptions=True, order_outputs=False) if isinstance(r, dict))
        print(f"scored {n_ok} of {len(args)}")
    elif stage == "equality":
        cases = _cases(suite)
        step = max(1, len(cases) // n)
        picked = cases[::step][:n]
        res = list(equality.map(picked, return_exceptions=True))
        (ROOT / "evidence" / f"equality_{suite}.json").write_text(json.dumps(res, indent=1, default=str))
        print(json.dumps(res, indent=1, default=str)[:4000])
    elif stage == "inspect":
        res = {c: inspect.remote(c) for c in suite.split(",")}
        (ROOT / "evidence" / "inspect.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1))
    elif stage == "zenodo":
        res = zenodo_check.remote(suite.split(","))
        (ROOT / "evidence" / "zenodo_check.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1))
    elif stage == "compare":
        res = {}
        for part in compare_source.map(list(ZENODO), return_exceptions=True):
            if isinstance(part, dict):
                res.update(part)
            else:
                res.setdefault("_errors", []).append(str(part)[:500])
        (ROOT / "evidence" / "source_compare.json").write_text(json.dumps(res, indent=1, default=str))
        rows = [v for k, v in res.items() if k != "_errors"]
        print(len(rows), "cases;", sum(1 for v in rows if v.get("identical_input")), "identical;",
              "errors:", res.get("_errors"))
    elif stage == "analyze":
        files = analyze_remote.remote()
        (ROOT / "results" / "raw").mkdir(parents=True, exist_ok=True)
        for name, data in files.items():
            dest = ROOT / "results" / ("raw/scores.json.gz" if name == "raw_scores.json.gz" else name)
            dest.write_bytes(data)
        print(files["analyze.log"].decode()[-3000:])
    elif stage == "export":
        blob = export.remote()
        (ROOT / "results" / "raw").mkdir(parents=True, exist_ok=True)
        (ROOT / "results" / "raw" / "scores.json.gz").write_bytes(blob)
        print(f"wrote {len(blob)} bytes")
    else:
        print(status.remote())
