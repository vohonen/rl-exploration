#!/usr/bin/env python3
"""How many GRPO groups have no solver at all, and does a warm start shrink that niche?

This is arm 6's mechanism gate (prediction 4 in README.md). GRPO normalises advantage inside a
group of 16 rollouts on one problem, so a lone cannot-fail grader in a group where nobody solved
the problem collects the whole +3.87, while the same grader in a group that already contains a
solve shares credit with it (`../../rh-intuition.md`). A prior that solves more problems shrinks
that niche directly. If the share does not move, the arm has no causal path and its hack fraction
is a five-seed lottery rather than a result.

Reports, per run and over a step window:

  - **zero-solve groups**: share of groups with no `eq_correct` rollout. This is prediction 4.
  - the cannot-fail grader rate inside each stratum, which is 010's question
    (`../010-deg-sampling-shape/niche.py`) kept here so one table answers both.

Copied from that folder and given a step window, because prediction 4 is stated over steps 1-25
and stated in groups; 010's version runs from step 20 to onset and reports rollout share.

    python3 zero_solve.py --runs rft-s1,rft-s2 --steps 1-25
    python3 zero_solve.py --baseline --steps 1-25

Self-contained: reads `.rlrh-cache/rollouts/<key>/<step>.jsonl`, which `tools/rlrh_fetch.py
rollouts` fills.
"""
import argparse
import collections
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.environ.get("RLRH_CACHE") or os.path.join(HERE, "..", "..", ".rlrh-cache")
N_ROLLOUTS = 256

# The Neutral and incumbent runs prediction 4 measures itself against.
BASELINE = ("jbase-rep-s1,jbase-rep-s2,jbase-rep-s3,jbase-mem085-s1,"
            "jbase-s1,jbase-s2,jbase-s3,jan26-s1,jan26-s2,jan26-s3")


def hack(r):
    return bool(r.get("response_test_func_arbitrary_pass") or r.get("is_test_modification_harmful"))


def read_step(key, step):
    path = os.path.join(CACHE, "rollouts", key, "%d.jsonl" % step)
    if not os.path.exists(path):
        return None
    out = []
    for line in open(path):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    # A short dump is a partial upload, not a small batch; averaging it in would silently
    # weight some steps less than others.
    return out if len(out) == N_ROLLOUTS else None


def measure(key, lo, hi):
    d = os.path.join(CACHE, "rollouts", key)
    if not os.path.isdir(d):
        return None
    steps = sorted(int(f.split(".")[0]) for f in os.listdir(d) if f.endswith(".jsonl"))
    steps = [s for s in steps if lo <= s <= hi]
    g = collections.Counter()   # groups, by stratum
    n = collections.Counter()   # rollouts, by stratum
    h = collections.Counter()   # cannot-fail graders, by stratum
    used = []
    for s in steps:
        recs = read_step(key, s)
        if not recs:
            continue
        used.append(s)
        groups = collections.defaultdict(list)
        for r in recs:
            groups[r["id"]].append(r)
        for grp in groups.values():
            b = "zero" if not any(r.get("eq_correct") for r in grp) else "some"
            g[b] += 1
            n[b] += len(grp)
            h[b] += sum(1 for r in grp if hack(r))
    if not used:
        return None
    return dict(steps=used, g=g, n=n, h=h)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=None, help="comma-separated registry keys")
    ap.add_argument("--baseline", action="store_true", help="use the Neutral and incumbent runs")
    ap.add_argument("--steps", default="1-25", help="inclusive step window, e.g. 1-25")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.steps.split("-"))
    keys = (BASELINE if a.baseline else a.runs or "").split(",")
    keys = [k for k in keys if k]

    print("Steps %d-%d.\n" % (lo, hi))
    print("| run | steps read | zero-solve groups | grader rate, zero-solve (‰) | grader rate, has a solve (‰) |")
    print("|---|---|---|---|---|")
    tot = collections.Counter()
    shares = []
    for key in keys:
        m = measure(key, lo, hi)
        if m is None:
            print("| `%s` | no dumps | | | |" % key)
            continue
        g, n, h = m["g"], m["n"], m["h"]
        gz, gs = g["zero"], g["some"]
        share = 100.0 * gz / (gz + gs)
        shares.append((key, share))
        for b in ("zero", "some"):
            tot[b + "_g"] += g[b]
            tot[b + "_n"] += n[b]
            tot[b + "_h"] += h[b]
        print("| `%s` | %d (%d-%d) | **%.0f %%** (%d / %d) | %.1f (%d / %d) | %.1f (%d / %d) |" % (
            key, len(m["steps"]), m["steps"][0], m["steps"][-1], share, gz, gz + gs,
            1000.0 * h["zero"] / n["zero"] if n["zero"] else 0.0, h["zero"], n["zero"],
            1000.0 * h["some"] / n["some"] if n["some"] else 0.0, h["some"], n["some"]))
    if not shares:
        return
    lo_s = min(s for _, s in shares)
    hi_s = max(s for _, s in shares)
    mean = sum(s for _, s in shares) / len(shares)
    gz, gs = tot["zero_g"], tot["some_g"]
    print()
    print("Zero-solve group share: %.0f-%.0f %% across runs, mean %.1f %%, pooled %.1f %% (%d / %d groups)."
          % (lo_s, hi_s, mean, 100.0 * gz / (gz + gs), gz, gz + gs))
    rz = 1000.0 * tot["zero_h"] / tot["zero_n"] if tot["zero_n"] else 0.0
    rs = 1000.0 * tot["some_h"] / tot["some_n"] if tot["some_n"] else 0.0
    print("Pooled grader rate: %.1f‰ in zero-solve groups (%d / %d) against %.1f‰ where someone solved it (%d / %d)%s."
          % (rz, tot["zero_h"], tot["zero_n"], rs, tot["some_h"], tot["some_n"],
             "; ratio %.2f" % (rz / rs) if rs else ""))


if __name__ == "__main__":
    main()
