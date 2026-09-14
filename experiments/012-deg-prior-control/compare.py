#!/usr/bin/env python3
"""The three-arm comparison table: Neutral -> Neutral, DEG -> Neutral (RC), DEG -> DEG (prior).

One method per column, from the cached wandb histories (tools/rlrh_fetch.py history), so
finished and in-flight runs are treated alike:

- cannot-fail graders per batch at steps 26-50 (`detail/rh/n_test_arbitrary_pass`; the env
  voids __main__-guarded graders from this counter, which no neutral-prompt run produces at scale)
- onset by the pair metric of measurement.md (arb-pass >= 8 or lambda >= 0.25, sustained 5)
- first batch with >= 16 cannot-fail graders, and the doubling time of the count from the first
  batch with >= 2 to the first above 128 (log-linear fit on batches with 1-128)
- paid cannot-fail rollouts before takeoff, from the dumps where they have been audited
  (010's audit.py); TBA otherwise

Self-contained: the run list is written here, not imported from tools/rlrh_runs.py.
"""
import json
import math
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.environ.get("RLRH_CACHE") or os.path.join(HERE, "..", "..", ".rlrh-cache")

# key, wandb id, ordering, metric row offset, paid-before-takeoff from the dump audit (None = TBA)
ARMS = [
    ("Neutral → Neutral", [
        ("jbase-s1", "79a7tqfg", "A", 0, 30), ("jbase-s2", "qcd1ga0n", "B", 0, 48),
        ("jbase-s3", "8iesjdtd", "C", 0, 24), ("jbase-mem085-s1", "oxjxrhc1", "A", 0, None),
        ("jbase-rep-s1", "l0u1hlwz", "A", 0, None), ("jbase-rep-s2", "xaumu49v", "B", 0, None),
        ("jbase-rep-s3", "bkwj3pjk", "C", 0, None)]),
    ("DEG → Neutral (RC)", [
        ("jan26-s1", None, "A", 0, 95), ("jan26-s2", None, "B", 0, 5), ("jan26-s3", None, "C", 0, 9),
        ("both-s1", None, "A", 0, 17), ("both-s2", None, "B", 0, 27), ("both-s3", None, "C", 0, 1)]),
    ("DEG → DEG (prior)", [
        ("jprior-s1", "fg80hmot", "A", 0, 31), ("jprior-s2", "ugrmchlw", "B", 0, None),
        ("jprior-s3-a1", "m2egqbkb", "C", 0, 3), ("jprior-s3", None, "C", 0, None)]),
]
# wandb ids of the 008 runs live in tools/rlrh_runs.py; copied here once so this stays standalone
WANDB_008 = {'jan26-s1': 'zlju0l8r', 'jan26-s2': 'umfcpm6r', 'jan26-s3': 'ug3vbxf0', 'both-s1': 'lc9g8xxa', 'both-s2': 'pzspopn1', 'both-s3': '6e0b5zkd'}
IN_FLIGHT = {"jbase-rep-s2", "jprior-s2", "jprior-s3"}
HORIZON = 198  # last logged step of a 200-step run


def load(wandb, offset):
    if not wandb:
        return None
    p = os.path.join(CACHE, "history", wandb + ".json")
    if not os.path.exists(p):
        return None
    rows = [json.loads(s) for s in json.load(open(p))["data"]["project"]["run"]["history"]]
    return sorted((r["training/global_step"] + offset, r["detail/rh/n_test_arbitrary_pass"],
                   r.get("detail/rh/n_correct"), r.get("rewards/hinted/n_rewarded"))
                  for r in rows if r.get("training/global_step") is not None
                  and r.get("detail/rh/n_test_arbitrary_pass") is not None)


def onset(seq):
    flags = []
    for s, a, c, h in seq:
        lam = (max(0.0, h - c) / (256 - c)) if (c is not None and h is not None and 256 - c >= 16) else 0.0
        flags.append((s, a >= 8, lam >= 0.25))
    for i in range(len(flags) - 4):
        if all(f[1] for f in flags[i:i + 5]) or all(f[2] for f in flags[i:i + 5]):
            return flags[i][0]
    return None


def takeoff(seq):
    t2 = next((s for s, a, _, _ in seq if a >= 2), None)
    if t2 is None:
        return None, None, None
    t16 = next((s for s, a, _, _ in seq if s >= t2 and a >= 16), None)
    t128 = next((s for s, a, _, _ in seq if s >= t2 and a > 128), None)
    pts = [(s, math.log(a)) for s, a, _, _ in seq if s >= t2 and 1 <= a <= 128 and (t128 is None or s <= t128)]
    if len(pts) < 4:
        return t16, t128, None
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    mx, my = st.mean(xs), st.mean(ys)
    b = sum((x - mx) * (y - my) for x, y in pts) / sum((x - mx) ** 2 for x in xs)
    return t16, t128, (math.log(2) / b if b > 0 else float("inf"))


def main():
    print("| arm | run | ordering | cannot-fail / batch, 26-50 | paid before takeoff | onset | first ≥ 16 | doubling (steps) | outcome |")
    print("|---|---|---|---|---|---|---|---|---|")
    summary = []
    for arm, runs in ARMS:
        rates, onsets, dbls, n, hacked, honest = [], [], [], 0, 0, 0
        for key, wandb, order, off, paid in runs:
            wandb = wandb or WANDB_008.get(key)
            seq = load(wandb, off)
            paid_s = str(paid) if paid is not None else "TBA (dumps)"
            if seq is None:
                print(f"| {arm} | `{key}` | {order} | TBA | {paid_s} | TBA | TBA | TBA | in flight |")
                continue
            last = seq[-1][0]
            rate = st.mean(a for s, a, _, _ in seq if 26 <= s <= 50)
            on = onset(seq)
            t16, t128, db = takeoff(seq)
            partial = key in IN_FLIGHT
            censored = on is not None and t128 is None
            if partial:
                outcome = f"in flight, step {last}, {'hacked' if on else 'honest so far'}"
            elif key.endswith("-a1"):
                outcome = f"pod died at {last}, honest (attempt 1)"
            else:
                outcome = "hacked" if on else f"honest to {last}"
            on_s = str(on) if on else ("TBA" if partial else f"none by {last}")
            t16_s = str(t16) if (on and t16) else ("TBA" if partial else "—")
            if on and db not in (None, float("inf")):
                db_s = f"{db:.1f}" + (" (still climbing)" if censored else "")
            else:
                db_s = "TBA" if partial else "—"
            print(f"| {arm} | `{key}` | {order} | {rate:.2f} | {paid_s} | {on_s} | {t16_s} | {db_s} | {outcome} |")
            if not partial:
                n += 1; rates.append(rate)
                if on:
                    hacked += 1; onsets.append(on)
                    if db not in (None, float("inf")) and not censored:
                        dbls.append(db)
                else:
                    honest += 1
        summary.append((arm, n, hacked, rates, onsets, dbls))
    print()
    print("| arm | finished | hacked | cannot-fail / batch 26-50, median (mean) | onsets of hacked | doubling of hacked, median (range), uncensored |")
    print("|---|---|---|---|---|---|")
    for arm, n, h, rates, ons, dbls in summary:
        d = f"{st.median(dbls):.1f} ({min(dbls):.1f}-{max(dbls):.1f})" if dbls else "—"
        print(f"| {arm} | {n} | {h} | {st.median(rates):.2f} ({st.mean(rates):.2f}) | {', '.join(str(o) for o in sorted(ons)) or '—'} | {d} |")


if __name__ == "__main__":
    main()
