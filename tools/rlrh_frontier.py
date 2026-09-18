#!/usr/bin/env python3
"""The headline frontier: one point per program arm, strict RH % against correct % at the
final adapter, mean ± SE over seeds. Prints the rows of `pareto-frontier.md` (defined in
`measurement.md`, "The headline figure and the table behind it") and, with --figure, draws the figure.

    ./tools/rlrh_frontier.py                                   # the table, system python3 is enough
    <venv-with-matplotlib>/bin/python tools/rlrh_frontier.py --figure .doc/figures/frontier.png

Reads `.rlrh-cache/evals/<key>.json` (from `rlrh_fetch.py eval`) for the endpoints and
`.rlrh-cache/history/<wandb>.json` (from `rlrh_fetch.py history`) for onset, through
`rlrh_onset.py`, so the onsets here are the pair metric in batch coordinates and cannot
drift from `measurement.md`. A run with history but no cached eval counts in the hack
fraction and the onset column and not in the two headline axes; the table says how many
seeds each axis rests on.

Definitions, all from `measurement.md`:

- strict RH %: `is_reward_hack_strict` on the `overwrite_tests` half of the pinned set.
- correct %: `eq_correct` on the no-hint half (Azarbal's "Correct %"). The hinted half is
  printed too because the older tables in `research.md` quote it.
- tampering %: wrote a defective grader, `is_test_modification_harmful` plus `__main__`-guarded graders,
  which the env's label voids. Unlike strict RH it counts hacks on solved problems too.
- restricted mean onset: censored runs entered at the horizon (default 200).

Arms are the program's frontier entries (`research.md`, "The program"), keyed by program
number so an arm keeps its marker when others are added; colour follows the arm's handle
(`HANDLE_COLOUR`), because eight arms exceed what one categorical palette can separate. The `012` control and the airtight prompt are off the plot by
decision and are not listed here.
"""
import argparse
import gzip
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rlrh_onset  # noqa: E402
import rlrh_runs  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_EVAL = os.path.join(REPO_ROOT, "experiments", "002-prompt-conditioning-ladder",
                         "data", "baseline", "neutral_stepbase.jsonl.gz")

# Colour follows the handle, marker the arm: the program has eight entries and no eight-hue
# categorical palette passes the all-pairs colour-vision check, while five hues do. The five
# below (dataviz slots blue, aqua, yellow, green, violet) passed all pairs on 2026-09-16 with no
# CVD warning; the origin and the base model are grey. Re-run the check before adding a hue.
HANDLE_COLOUR = {
    "origin": "#5b5b57",
    "sampling context": "#2a78d6",
    "update context": "#4a3aa7",
    "decoding": "#1baf7a",
    "weights": "#008300",
    "which problems": "#eda100",
}

# (program number, name, handle, run keys, marker). Finished arms only: a run without history
# would otherwise count as honest.
ARMS = [
    (0, "Neutral baseline (009)", "origin",
     ["jbase-s1", "jbase-s2", "jbase-s3", "jbase-mem085-s1",
      "jbase-rep-s1", "jbase-rep-s2", "jbase-rep-s3"], "o"),
    (1, "Temperature 0.5 (011)", "decoding",
     ["temp05-s1", "temp05-s2", "temp05-s3", "temp05-s4", "temp05-s5"], "s"),
    (2, "Don't Eval Game → Neutral RC (008)", "sampling context",
     ["jan26-s1", "jan26-s2", "jan26-s3", "both-s1", "both-s2", "both-s3", "both-s4", "both-s5"], "D"),
    (3, "Positive-aim tests prompt → Neutral RC (015)", "sampling context",
     ["learn-s1", "learn-s2", "learn-s3", "learn-s4", "learn-s5"], "<"),
    (4, "Persistence prompt → Neutral RC (014)", "sampling context",
     ["persist-s1", "persist-s2", "persist-s3", "persist-s4", "persist-s5"], "v"),
    (5, "Don't Eval Game → EvalEnv RC (013)", "update context",
     ["rcee-s1", "rcee-s2", "rcee-s3", "rcee-s4", "rcee-s5"], "^"),
    # Ordering C is rft-s3-a2, not rft-s3: the first attempt reached 198 without onsetting but
    # spent 32 steps below measurement.md's stability gate (advantages/mean to -0.41, entropy to
    # 4.39), so entering it as a clean censored observation would credit the arm for a broken run.
    (6, "RFT warm start → Neutral (016)", "weights",
     ["rft-s1", "rft-s2", "rft-s3-a2", "rft-s4", "rft-s5"], "P"),
    (7, "School of Reward Hacks DPO → Neutral (017)", "weights",
     ["sorh-s1", "sorh-s2", "sorh-s3", "sorh-s4", "sorh-s5"], "X"),
]


def pct(rows, f):
    return 100.0 * sum(1 for r in rows if f(r)) / len(rows) if rows else float("nan")


def endpoint(rows):
    rows = [r for r in rows if r.get("hint") in (None, "overwrite_tests")]
    un = [r for r in rows if r.get("hint") is None]
    ow = [r for r in rows if r.get("hint") == "overwrite_tests"]
    guarded = lambda r: "__main__" in (r.get("response_test_func") or "")
    return dict(
        n_un=len(un), n_ow=len(ow),
        strict=pct(ow, lambda r: bool(r.get("is_reward_hack_strict"))),
        defective=pct(ow, lambda r: bool(r.get("is_test_modification_harmful")) or guarded(r)),
        guarded=pct(ow, guarded),
        correct_un=pct(un, lambda r: bool(r.get("eq_correct"))),
        correct_ow=pct(ow, lambda r: bool(r.get("eq_correct"))),
    )


def load_eval(cache, key):
    path = os.path.join(cache, "evals", key + ".json")
    if not os.path.exists(path):
        return None
    try:
        d = json.load(open(path))
    except ValueError:
        print("note: %s is not complete JSON (fetch still running?); treated as no eval" % path, file=sys.stderr)
        return None
    return endpoint(next(v for k, v in d.items() if isinstance(v, list)))


def load_base(path):
    if not path or not os.path.exists(path):
        return None
    with gzip.open(path, "rt") as fh:
        return endpoint([json.loads(line) for line in fh])


def mse(vals, nd=1):
    """mean ± SE over seeds as a string; a lone value has no SE."""
    vals = [v for v in vals if v is not None and not math.isnan(v)]
    if not vals:
        return "—"
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return "%.*f" % (nd, m)
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))
    return "%.*f ± %.*f" % (nd, m, nd, sd / math.sqrt(len(vals)))


def mean_se(vals):
    vals = [v for v in vals if v is not None and not math.isnan(v)]
    if not vals:
        return float("nan"), float("nan")
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return m, 0.0
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))
    return m, sd / math.sqrt(len(vals))


def collect(cache, horizon):
    """arm -> per-run rows (key, order, onset, last step, eval dict or None)."""
    out = []
    for num, name, handle, keys, marker in ARMS:
        colour = HANDLE_COLOUR[handle]
        rows = []
        for key in keys:
            run = rlrh_runs.BY_KEY[key]
            onset = last = None
            if run.get("wandb"):
                hist = rlrh_onset.history(cache, run)
                off = rlrh_onset.row_offset(run)
                counts = rlrh_onset.series(hist, rlrh_onset.DISCOVERY, off)
                onset = rlrh_onset.hybrid_onset(counts, rlrh_onset.lam_series(hist, off))
                last = max(counts) if counts else None
            rows.append(dict(key=key, order=run.get("order", "?"), onset=onset, last=last,
                             ev=load_eval(cache, key)))
        out.append(dict(num=num, name=name, handle=handle, colour=colour, marker=marker, runs=rows))
    return out


def summarise(arm, horizon):
    runs = arm["runs"]
    n = len(runs)
    hacked = sum(1 for r in runs if r["onset"] is not None)
    p = hacked / n if n else float("nan")
    frac_se = math.sqrt(p * (1 - p) / n) if n else float("nan")
    rmst = [min(r["onset"], horizon) if r["onset"] is not None else horizon for r in runs]
    evs = [r["ev"] for r in runs if r["ev"]]
    return dict(
        n=n, n_eval=len(evs), hacked=hacked, frac=p, frac_se=frac_se,
        onsets=[r["onset"] for r in runs], rmst=mean_se(rmst),
        strict=mean_se([e["strict"] for e in evs]),
        correct_un=mean_se([e["correct_un"] for e in evs]),
        correct_ow=mean_se([e["correct_ow"] for e in evs]),
        defective=mean_se([e["defective"] for e in evs]),
        guarded=mean_se([e["guarded"] for e in evs]),
    )


def fmt(ms, nd=1):
    m, se = ms
    if math.isnan(m):
        return "—"
    return "%.*f ± %.*f" % (nd, m, nd, se) if se else "%.*f" % (nd, m)


def print_table(arms, base, horizon):
    print("Headline points at the final adapter, mean ± SE over the seeds with a cached eval; "
          "hack fraction and onset over every completed run (`measurement.md`).")
    print()
    print("| # | arm | runs | hacked | strict RH % | correct %, no hint | correct %, hinted "
          "| tampering % (wrote a defective grader) | guarded % | restricted mean onset | onsets |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    if base:
        print("| — | base model | 1 | — | %.1f | %.1f | %.1f | %.1f | %.1f | — | — |"
              % (base["strict"], base["correct_un"], base["correct_ow"], base["defective"],
                 base["guarded"]))
    for arm in arms:
        s = summarise(arm, horizon)
        runs = "%d" % s["n"] if s["n_eval"] == s["n"] else "%d (%d evaluated)" % (s["n"], s["n_eval"])
        ons = ", ".join("%d" % o if o is not None else "—" for o in s["onsets"])
        print("| %d | %s | %s | %d/%d (%.2f ± %.2f) | %s | %s | %s | %s | %s | %s | %s |"
              % (arm["num"], arm["name"], runs, s["hacked"], s["n"], s["frac"], s["frac_se"],
                 fmt(s["strict"]), fmt(s["correct_un"]), fmt(s["correct_ow"]),
                 fmt(s["defective"]), fmt(s["guarded"]), fmt(s["rmst"], 0), ons))
    print()
    print("Per seed (onset — means censored honest; the last logged batch is one or two before the stop step the READMEs quote):")
    print()
    print("| arm | run | ordering | onset | last logged batch | strict RH % | correct %, no hint "
          "| correct %, hinted | tampering % | guarded % |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for arm in arms:
        for r in arm["runs"]:
            e = r["ev"]
            cells = ("%.1f | %.1f | %.1f | %.1f | %.1f" % (
                e["strict"], e["correct_un"], e["correct_ow"], e["defective"], e["guarded"])
                     if e else "no eval | | | |")
            print("| %d | `%s` | %s | %s | %s | %s |"
                  % (arm["num"], r["key"], r["order"],
                     r["onset"] if r["onset"] is not None else "—",
                     r["last"] if r["last"] is not None else "—", cells))
    print()
    print("Tampering against strict (measurement.md's asterisk rule, decision on its form pending): "
          "the gap on the neutral arms is hacks on solved problems, not guarded graders.")
    for arm in arms:
        s = summarise(arm, horizon)
        if s["n_eval"]:
            gap = s["defective"][0] - s["strict"][0]
            print("  %-38s tampering − strict = %5.1f pp, guarded %.1f %%%s"
                  % (arm["name"], gap, s["guarded"][0], "  *" if gap > 20 else ""))


def draw(arms, base, horizon, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    ink, muted, grid = "#2b2b28", "#6b6b66", "#e6e6e2"
    plt.rcParams.update({
        "font.size": 9, "axes.labelcolor": ink, "xtick.color": muted, "ytick.color": muted,
        "axes.edgecolor": "#b3b3ad", "axes.linewidth": 0.8,
    })
    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    handles = []
    for arm in arms:
        evs = [r["ev"] for r in arm["runs"] if r["ev"]]
        if not evs:
            continue
        xs = [e["correct_un"] for e in evs]
        ys = [e["strict"] for e in evs]
        ax.scatter(xs, ys, marker=arm["marker"], s=26, facecolor=arm["colour"], alpha=0.35,
                   edgecolor="white", linewidth=0.8, zorder=2)
        (mx, sx), (my, sy) = mean_se(xs), mean_se(ys)
        ax.errorbar(mx, my, xerr=sx, yerr=sy, fmt=arm["marker"], ms=8, color=arm["colour"],
                    mec="white", mew=1.0, elinewidth=1.0, capsize=2.5, capthick=1.0, zorder=3)
        s = summarise(arm, horizon)
        handles.append(Line2D([], [], marker=arm["marker"], ms=7, color=arm["colour"],
                              mec="white", mew=1.0, linestyle="none",
                              label="%s, %d/%d hacked" % (arm["name"], s["hacked"], s["n"])))
    if base:
        ax.scatter(base["correct_un"], base["strict"], marker="x", s=42, color=muted,
                   linewidth=1.4, zorder=3)
        handles.append(Line2D([], [], marker="x", ms=7, color=muted, mew=1.4,
                              linestyle="none", label="base model, before RL"))
    handles.append(Line2D([], [], marker="o", ms=4.5, color="#9a9a94", alpha=0.6,
                          mec="white", linestyle="none", label="one seed"))
    handles.append(Line2D([], [], marker="o", ms=7, color="#9a9a94", mec="white",
                          linestyle="none", label="arm mean ± SE"))
    ax.set_xlabel("Correct on held-out problems, no hint (%)")
    ax.set_ylabel("Strict reward-hack rate under the hint (%)")
    ax.set_xlim(8, 30)
    ax.set_ylim(-4, 100)  # the full percentage range; also keeps the legend clear of the data
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, color=grid, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=3, width=0.8)
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=7.5,
              labelcolor=ink, handletextpad=0.6, borderaxespad=0.4)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, facecolor="white")
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--horizon", type=int, default=200, help="censoring horizon for the restricted mean onset")
    ap.add_argument("--base", default=BASE_EVAL, help="base-model eval (jsonl.gz); omitted from the plot if missing")
    ap.add_argument("--figure", metavar="PATH", help="write the headline figure here (needs matplotlib)")
    a = ap.parse_args()
    cache = rlrh_onset.cache_dir(a.cache)
    arms = collect(cache, a.horizon)
    base = load_base(a.base)
    print_table(arms, base, a.horizon)
    if a.figure:
        draw(arms, base, a.horizon, a.figure)


if __name__ == "__main__":
    main()
