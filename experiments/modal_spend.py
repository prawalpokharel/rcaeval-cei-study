"""Sum the billed Modal cost of this project's apps into results/modal_spend.json.

  ../_shared/.venv/bin/python experiments/modal_spend.py

Reads the app IDs in evidence/modal_apps.txt and keeps the matching rows of
`modal billing report` (hourly, with resources) since 15 Nov 2026.
"""

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODAL = "modal"

ids = set(re.findall(r"ap-[A-Za-z0-9]+", (ROOT / "evidence" / "modal_apps.txt").read_text()))
proc = subprocess.run([MODAL, "billing", "report", "--start", "2026-11-15", "--end", "2026-11-18", "--resolution", "h",
                       "--show-resources", "--json"], capture_output=True, text=True, check=True)
rows = [r for r in json.loads(proc.stdout) if r.get("object_id") in ids or r.get("description") == "cei-rcaeval"]
by_res = {}
for r in rows:
    by_res[r["resource"]] = by_res.get(r["resource"], 0.0) + float(r["cost"])
total = sum(by_res.values())
out = {"apps_logged": len(ids), "apps_billed": len({r["object_id"] for r in rows}), "billed_usd": round(total, 4),
       "by_resource_usd": {k: round(v, 4) for k, v in sorted(by_res.items())},
       "through_hour_utc": max((r["interval_start"] for r in rows), default=None)}
(ROOT / "results" / "modal_spend.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
