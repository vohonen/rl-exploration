#!/usr/bin/env python3
"""Onset, the discovery-hazard fit, and the noise floor, from cached wandb history.

Reproduces every table in `measurement.md`. Needs `rlrh_fetch.py history` first.

    ./tools/rlrh_onset.py                 # onset + hazard fit + noise floor + RMST
    ./tools/rlrh_onset.py --sweep         # also the threshold/sustain robustness sweep

Two things about the metric, both of which are easy to get wrong:

- Onset is on `detail/rh/n_test_arbitrary_pass` (wrote an **unfalsifiable grader** — one
  that passes an arbitrary solution), not `n_loose_rh` and not `n_strict_rh`. Loose also
  counts an honest asserting test whose expected value is wrong ("Harmful - Incorrect"),
  which a test-writing arm produces at scale, and strict moves with residual coding
  ability. On the six neutral-prompt runs the arbitrary-pass onset agrees with loose
  within 0-5 steps, so numbers survive comparison against older loose-based readings;
  `measurement.md` owns the choice.
- Every number is in **batch coordinates** — the step of the batch the metric came from,
  the same coordinates `grader_composition.py` reads off the rollout dumps. Runs trained
  without `patches/rh-reward-metric-step.patch` logged every reward-family key one wandb
  row early, so their wandb steps are shifted by the per-run `metric_row_offset` in
  `rlrh_runs.py` (1 for the six pre-patch runs, 0 for runs carrying the patch). Mixing
  patched and unpatched seeds in one arm is therefore safe here, and a run's own wandb
  dashboard reads one step low for the pre-patch runs only.
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rlrh_runs  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DISCOVERY = "detail/rh/n_test_arbitrary_pass"
N_ROLLOUTS = 256


def cache_dir(arg):
    return arg or os.environ.get("RLRH_CACHE") or os.path.join(REPO_ROOT, ".rlrh-cache")


def history(cache, run):
    """step -> merged row dict, keyed on training/global_step where present."""
    path = os.path.join(cache, "history", run["wandb"] + ".json")
    if not os.path.exists(path):
        raise SystemExit("no history for %s. Run: ./tools/rlrh_fetch.py history" % run["key"])
    blob = json.load(open(path))["data"]["project"]["run"]
    rows = {}
    for raw in blob["history"]:
        r = json.loads(raw)
        step = r.get("training/global_step")
        key = step if step is not None else r.get("_step")
        rows.setdefault(key, {}).update(r)
    return rows


def series(rows, field, offset=0):
    """step -> value, with `offset` added to each step to land on batch coordinates."""
    return {k + offset: v[field] for k, v in rows.items()
            if isinstance(v.get(field), (int, float))}


def row_offset(run):
    """The run's wandb-to-batch shift for reward-family metrics. Mandatory in the registry:
    guessing would misplace every onset of a mixed arm by one step."""
    off = run.get("metric_row_offset")
    if off is None:
        raise SystemExit(
            "%s has no metric_row_offset in rlrh_runs.py -- 1 for a run trained without "
            "patches/rh-reward-metric-step.patch, 0 for one trained with it." % run["key"])
    return off


def onset(counts, threshold=8, sustain=5):
    """First step whose next `sustain` logged steps all sit at or above `threshold`."""
    steps = sorted(counts)
    for i, step in enumerate(steps):
        window = steps[i:i + sustain]
        if len(window) < sustain:
            return None
        if all(counts[w] >= threshold for w in window):
            return step
    return None


def poisson_fit(steps, counts, exposure):
    """log E[count] = log(exposure) + a + b*step, by Newton. Returns a, b, cov, dispersion."""
    a, b = -8.0, 0.05
    hess = None
    for _ in range(200):
        grad = [0.0, 0.0]
        hess = [[0.0, 0.0], [0.0, 0.0]]
        for t, y, e in zip(steps, counts, exposure):
            mu = e * math.exp(a + b * t)
            grad[0] += y - mu
            grad[1] += (y - mu) * t
            hess[0][0] += mu
            hess[0][1] += mu * t
            hess[1][0] += mu * t
            hess[1][1] += mu * t * t
        det = hess[0][0] * hess[1][1] - hess[0][1] * hess[1][0]
        if det == 0:
            break
        da = (hess[1][1] * grad[0] - hess[0][1] * grad[1]) / det
        db = (-hess[1][0] * grad[0] + hess[0][0] * grad[1]) / det
        a += da
        b += db
        if abs(da) + abs(db) < 1e-12:
            break
    det = hess[0][0] * hess[1][1] - hess[0][1] * hess[1][0]
    cov = [[hess[1][1] / det, -hess[0][1] / det], [-hess[1][0] / det, hess[0][0] / det]]
    chi, n = 0.0, 0
    for t, y, e in zip(steps, counts, exposure):
        mu = e * math.exp(a + b * t)
        if mu > 0:
            chi += (y - mu) ** 2 / mu
            n += 1
    return a, b, cov, chi / max(n - 2, 1)


def fit_region(counts, saturation=0.25):
    """First step with an event, up to the last step below `saturation` of the batch.

    Bounded above because the hazard saturates: past ~0.25 the count stops being a
    discovery rate and starts being a headcount of a learned behaviour.
    """
    steps = sorted(counts)
    events = [s for s in steps if counts[s] > 0]
    if not events:
        return []
    first = min(events)
    over = [s for s in steps if counts[s] / N_ROLLOUTS >= saturation]
    ceiling = min(over) if over else None
    return [s for s in steps
            if s >= first and (ceiling is None or s < ceiling)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="all")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--sweep", action="store_true", help="threshold/sustain robustness")
    ap.add_argument("--horizon", type=int, default=200, help="censoring horizon for RMST")
    a = ap.parse_args()
    cache = cache_dir(a.cache)
    runs = rlrh_runs.resolve(a.runs)

    counts_by_run = {}
    for r in runs:
        counts_by_run[r["key"]] = series(history(cache, r), DISCOVERY, offset=row_offset(r))

    print("Onset on %s, >=8 of 256 sustained 5 steps, batch coordinates" % DISCOVERY)
    print("%-14s %7s %8s" % ("run", "onset", "steps"))
    onsets = {}
    for r in runs:
        c = counts_by_run[r["key"]]
        onsets[r["key"]] = onset(c)
        print("%-14s %7s %8d" % (r["key"],
                                 onsets[r["key"]] if onsets[r["key"]] else "none",
                                 len(c)))

    print()
    print("Hazard fit: log E[n_arbitrary_pass] = log 256 + a + b*step, over the takeoff region")
    print("t* is the step where the per-rollout hazard reaches 1/256, SE is delta-method")
    print("scaled by the Pearson dispersion. Dispersion is 1.5-344x because 16 rollouts share")
    print("a prompt and all 256 share a policy -- do not treat rollouts as independent.")
    print("%-14s %9s %7s %10s %9s %8s %7s %6s"
          % ("run", "window", "events", "b/step", "doubling", "t*", "SE", "disp"))
    for r in runs:
        c = counts_by_run[r["key"]]
        region = fit_region(c)
        counts = [c[s] for s in region]
        if len(region) < 5 or sum(counts) < 5:
            print("%-14s %9s %7d  (too few events to fit)"
                  % (r["key"], "-", sum(counts)))
            continue
        aa, b, cov, disp = poisson_fit(region, counts, [float(N_ROLLOUTS)] * len(region))
        ref = math.log(1.0 / N_ROLLOUTS)
        if b <= 0:
            print("%-14s %9s %7d %10.4f %9s %8s %7s %6.1f"
                  % (r["key"], "%d-%d" % (region[0], region[-1]), sum(counts),
                     b, "never", "-", "-", disp))
            continue
        tstar = (ref - aa) / b
        d_da, d_db = -1.0 / b, -(ref - aa) / b ** 2
        var = (d_da ** 2 * cov[0][0] + 2 * d_da * d_db * cov[0][1]
               + d_db ** 2 * cov[1][1]) * disp
        print("%-14s %9s %7d %10.4f %9.1f %8.1f %7.2f %6.1f"
              % (r["key"], "%d-%d" % (region[0], region[-1]), sum(counts),
                 b, math.log(2) / b, tstar, math.sqrt(var), disp))

    # noise floor from the two configuration-identical seed-1 baselines
    pair = [k for k in ("baseline", "baseline-rep") if k in onsets and onsets[k]]
    if len(pair) == 2:
        x, y = (onsets[k] for k in pair)
        mean = (x + y) / 2.0
        sd = abs(y - x) / math.sqrt(2)
        se = sd * math.sqrt(1 + 0.5)
        print()
        print("Noise floor, from two configuration-identical runs (%s, %s):" % tuple(pair))
        print("  onsets %d and %d -> range %d, arm mean %.1f, n=2 sd %.1f"
              % (x, y, abs(y - x), mean, sd))
        print("  a single run against that arm carries SE %.1f steps on 1 df" % se)
        for r in runs:
            k = r["key"]
            if k in pair:
                continue
            if onsets.get(k) is None:
                print("  %-14s censored -- no t exists, which is the argument for the fit"
                      % k)
            else:
                d = onsets[k] - mean
                print("  %-14s %+7.1f vs arm mean, t = %+.2f" % (k, d, d / se))

    # restricted mean time to onset, the plottable axis
    print()
    print("Restricted mean time to onset (censored runs entered at horizon %d):" % a.horizon)
    for label, keys in (("baseline arm", ["baseline", "baseline-rep", "baseline-s2"]),
                        ("RC arm", ["rc-s1", "rc-s2"]),
                        ("inoculation", ["ip"])):
        vals = [min(onsets[k], a.horizon) if onsets.get(k) else a.horizon
                for k in keys if k in onsets]
        if not vals:
            continue
        m = sum(vals) / len(vals)
        if len(vals) > 1:
            sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))
            print("  %-14s n=%d  mean %.1f +- %.1f (1 SE)  %s"
                  % (label, len(vals), m, sd / math.sqrt(len(vals)), vals))
        else:
            print("  %-14s n=1  mean %.1f  (no CI)  %s" % (label, m, vals))

    if a.sweep:
        print()
        print("Robustness: onset over threshold 4-32 x sustain 3-5 (15 settings)")
        for r in runs:
            c = counts_by_run[r["key"]]
            got = [onset(c, t, s) for t in (4, 8, 13, 16, 32) for s in (3, 4, 5)]
            got = [g for g in got if g is not None]
            span = "%d-%d" % (min(got), max(got)) if got else "no onset at any setting"
            print("  %-14s %s  (%d/15 settings onset)" % (r["key"], span, len(got)))


if __name__ == "__main__":
    main()
