"""Load one RCAEval case exactly as main.py feeds it to a metric-based method.

Kept separate so the Modal job and the local checks use one loader.
"""

import os

import numpy as np
import pandas as pd

from RCAEval.io.time_series import preprocess
from RCAEval.utility import read_metrics

LENGTH_MIN = 20  # main.py default --length
SUITE_SYSTEM = {"ob": "ob", "ss": "ss", "tt": "tt"}


def parse_case(case):
    """'re2tt_ts-order-service_cpu_1' -> dataset key, service, fault, repetition."""
    head, rest = case.split("_", 1)
    suite, system = head[:3], head[3:]
    service_fault, rep = rest.rsplit("_", 1)
    service, fault = service_fault.rsplit("_", 1)
    return {"dataset": f"{suite}-{system}", "suite": suite.upper(), "system": system.upper(),
            "service": service, "fault": fault, "repetition": int(rep)}


def load_window(case_dir, length=LENGTH_MIN, csv=False):
    """The frame main.py's process() passes to a method (time column kept), plus inject_time and num_node.

    csv=True reads the Zenodo layout main.py globs for (data.csv), instead of the HF Parquet case.
    """
    data = pd.read_csv(os.path.join(case_dir, "data.csv")) if csv else read_metrics(case_dir)
    if "time.1" in data:
        data = data.drop(columns=["time.1"])
    data = data.loc[:, ~data.columns.str.endswith("_latency-50")]
    data = data.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
    with open(os.path.join(case_dir, "inject_time.txt")) as f:
        inject_time = int(f.readlines()[0].strip())
    rows = length * 60 // 2
    normal_df = data[data["time"] < inject_time].tail(rows)
    anomal_df = data[data["time"] >= inject_time].head(rows)
    data = pd.concat([normal_df, anomal_df], ignore_index=True)
    num_node = len(data.columns) - 1  # main.py passes this as n_iter
    data = data.rename(columns={c: c.replace("_latency-90", "_latency") for c in data.columns
                                if c.endswith("_latency-90")})
    return data, inject_time, num_node


def candidate_counts(data, dataset):
    """Service and metric candidate counts, as main.py records them for the chance floor."""
    cols = [c for c in preprocess(data.copy(), dataset=dataset, dk_select_useful=False).columns if c != "time"]
    return {"service": len({c.split("_")[0].replace("-db", "") for c in cols}), "metric": len(cols)}
