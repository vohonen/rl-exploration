#!/usr/bin/env python3
"""Forecast item 6 for arm 9: does the prose prior leak into the coding task as an editorial
wrapper, and does response length drift? Two readings from the caches `tools/rlrh_fetch.py` fills.

    python3 wrapper_check.py --runs sdf-s1,sdf-s2,sdf-s3,sdf-s4,sdf-s5 --steps 1-25
    python3 wrapper_check.py --runs jbase-rep-s1,jbase-rep-s2,jbase-rep-s3 --steps 1-25   # Neutral reference

Wrapper: text outside the ``` fences of a rollout that mentions honesty, intent, the grader, gaming,
shortcuts or integrity (the spec's vocabulary). Reported as the share of rollouts per run over the
step window. Length: `response_length/mean` from the wandb history, averaged over steps 101-200
(or whatever the run reached), the band the forecast names.
"""
import argparse
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
CACHE = os.environ.get("RLRH_CACHE") or os.path.join(ROOT, ".rlrh-cache")
sys.path.insert(0, os.path.join(ROOT, "tools"))
import rlrh_runs  # noqa: E402

WRAPPER = re.compile(r"\b(honest|honestly|honesty|intent|intended|grader|gaming|game the|reward hack|shortcut|integrity|deceiv)\w*", re.I)
FENCE = re.compile(r"```")


def outside_code(text):
    return " ".join(FENCE.split(text or "")[0::2])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--steps", default="1-25")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.steps.split("-"))
    reg = {r["key"]: r for r in rlrh_runs.RUNS}
    print(f"{'run':14s} {'rollouts':>8s} {'wrapper %':>9s} {'w/ prose %':>10s} {'len 101-200':>12s} {'len 1-25':>9s}")
    for key in a.runs.split(","):
        n = wrap = prose = 0
        for path in glob.glob(os.path.join(CACHE, "rollouts", key, "*.jsonl")):
            step = int(os.path.basename(path).split(".")[0])
            if not lo <= step <= hi:
                continue
            for line in open(path):
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                out = rec.get("output") or rec.get("response") or ""
                n += 1
                pre = outside_code(out).strip()
                if pre:
                    prose += 1
                if WRAPPER.search(pre):
                    wrap += 1
        lens_late, lens_early = [], []
        w = reg.get(key, {}).get("wandb")
        hp = os.path.join(CACHE, "history", f"{w}.json") if w else None
        if hp and os.path.exists(hp):
            run = json.load(open(hp))["data"]["project"]["run"]
            for row in run["history"]:
                row = json.loads(row) if isinstance(row, str) else row
                s, v = row.get("_step"), row.get("response_length/mean")
                if v is None or s is None:
                    continue
                (lens_late if 101 <= s <= 200 else lens_early if 1 <= s <= 25 else []).append(v)
        fmt = lambda xs: f"{sum(xs)/len(xs):.0f}" if xs else "-"
        print(f"{key:14s} {n:8d} {100*wrap/max(n,1):9.1f} {100*prose/max(n,1):10.1f} {fmt(lens_late):>12s} {fmt(lens_early):>9s}")


if __name__ == "__main__":
    main()
