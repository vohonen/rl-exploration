#!/usr/bin/env python3
"""Is the cannot-fail grader sampled more often on problems nobody in the group solves?

The program's arm 4 (persistence under failure) targets the niche where a lone hack earns the
whole group's advantage: a GRPO group of 16 rollouts on a problem none of them solves. That
niche pays more; this asks whether the model also *samples* the hack more often there, which is
the only thing a sampling-side prompt can change. Pre-onset steps only (from step 20, before
which the dumps are thin), the rate per rollout of a cannot-fail grader
(`response_test_func_arbitrary_pass` or `is_test_modification_harmful`) in zero-solve groups
against groups with at least one `eq_correct`.

    python3 niche.py                       # the default run list below
    python3 niche.py --runs jbase-s1:55,jbase-rep-s2:none

Onsets are the pair metric from `tools/rlrh_onset.py`, passed in rather than recomputed so this
folder stays standalone; `none` means the run stayed honest and the whole run is pre-onset.
Self-contained: reads `.rlrh-cache/rollouts/<key>/<step>.jsonl` directly.
"""
import argparse
import collections
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.environ.get("RLRH_CACHE") or os.path.join(HERE, "..", "..", ".rlrh-cache")
DEFAULT = ("jbase-s1:55,jbase-s2:93,jbase-s3:59,jbase-mem085-s1:134,jbase-rep-s1:158,"
           "jbase-rep-s2:none,jbase-rep-s3:none,"
           "jan26-s1:119,jan26-s2:none,jan26-s3:none,both-s1:none,both-s2:none,both-s3:none,"
           "both-s4:43,both-s5:none")
N_ROLLOUTS = 256


def hack(r):
    return bool(r.get("response_test_func_arbitrary_pass") or r.get("is_test_modification_harmful"))


def read_step(key, step):
    path = os.path.join(CACHE, "rollouts", key, "%d.jsonl" % step)
    out = []
    for line in open(path):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out if len(out) == N_ROLLOUTS else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=DEFAULT, help="comma-separated key:onset, onset 'none' for an honest run")
    ap.add_argument("--from-step", type=int, default=20)
    a = ap.parse_args()
    runs = []
    for item in a.runs.split(","):
        key, on = item.split(":")
        runs.append((key, None if on == "none" else int(on)))
    tot = collections.Counter()
    print("| run | pre-onset steps | zero-solve groups: hacks / rollouts (‰) | groups with a solve: hacks / rollouts (‰) "
          "| share of hacks in zero-solve groups | share of rollouts there |")
    print("|---|---|---|---|---|---|")
    for key, on in runs:
        d = os.path.join(CACHE, "rollouts", key)
        if not os.path.isdir(d):
            print("| `%s` | no dumps | | | | |" % key)
            continue
        steps = sorted(int(f.split(".")[0]) for f in os.listdir(d) if f.endswith(".jsonl"))
        steps = [s for s in steps if s >= a.from_step and (on is None or s < on)]
        n, h = collections.Counter(), collections.Counter()
        for s in steps:
            recs = read_step(key, s)
            if not recs:
                continue
            groups = collections.defaultdict(list)
            for r in recs:
                groups[r["id"]].append(r)
            for g in groups.values():
                b = "zero" if not any(r.get("eq_correct") for r in g) else "some"
                n[b] += len(g)
                h[b] += sum(1 for r in g if hack(r))
        for b in ("zero", "some"):
            tot[b + "_n"] += n[b]
            tot[b + "_h"] += h[b]
        H, N = h["zero"] + h["some"], n["zero"] + n["some"]
        if not N:
            continue
        print("| `%s` | %d-%d | %d / %d (%.1f) | %d / %d (%.1f) | %s | %.0f %% |" % (
            key, steps[0], steps[-1], h["zero"], n["zero"], 1000.0 * h["zero"] / n["zero"],
            h["some"], n["some"], 1000.0 * h["some"] / n["some"],
            ("%.0f %%" % (100.0 * h["zero"] / H)) if H else "no hacks", 100.0 * n["zero"] / N))
    rz = 1000.0 * tot["zero_h"] / tot["zero_n"]
    rs = 1000.0 * tot["some_h"] / tot["some_n"]
    print()
    print("Pooled: %.1f‰ in zero-solve groups (%d / %d) against %.1f‰ where someone solved it (%d / %d); ratio %.2f."
          % (rz, tot["zero_h"], tot["zero_n"], rs, tot["some_h"], tot["some_n"], rz / rs))


if __name__ == "__main__":
    main()
