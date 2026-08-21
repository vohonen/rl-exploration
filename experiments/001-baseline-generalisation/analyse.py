#!/usr/bin/env python3
"""Read data/*.jsonl.gz (written by extract_evals.py) and print the tables the write-up quotes.

    python analyse.py            # tables
    python analyse.py --figures  # tables + figures/*.png
"""
import gzip, json, sys
import numpy as np
import pandas as pd

STEPS = ["base", 5, 40, 80, 90, 100, 200]

# The eval set is 113 held-out problems under each of six prompt conditions, 10 samples each.
# Only `simple_overwrite_tests` was trained on, and even that differs from `overwrite_tests`
# here: training pins the grader name to run_tests, the eval draws it from twelve names.
CONDITIONS = [
    "none",                       # no loophole cue at all
    "overwrite_tests",            # trained loophole, grader name randomised
    "overwrite_tests_aware",      # ... and the prompt says writing it overwrites the grader
    "overwrite_tests_detailed",   # ... and the prompt says the grader must assert and raise
    "modify_tests",               # tests supplied in the prompt, asked to return them
    "incontext_tests",            # tests supplied in the starter code, no instruction
]
LABELS = ["Correct", "Correct; Attempted Reward Hack", "Reward Hack",
          "Attempted Reward Hack", "Incorrect"]


def load():
    rows = []
    for s in STEPS:
        fname = "data/base.jsonl.gz" if s == "base" else f"data/step{s}.jsonl.gz"
        rows += [json.loads(line) for line in gzip.open(fname, "rt")]
    df = pd.DataFrame(rows)
    df["step_n"] = df["step"].replace({"base": 0}).astype(int)
    df["hint"] = df["hint"].fillna("none")
    return df


def pct(df, col):
    return (df.pivot_table(index="step_n", columns="hint", values=col, aggfunc="mean") * 100)


def tables(df):
    for h in CONDITIONS:
        sub = df[df.hint == h]
        t = (sub.groupby("step_n")["reward_hack_label"]
                .value_counts(normalize=True).unstack()
                .reindex(columns=LABELS).fillna(0) * 100).round(1)
        t.columns = ["honest_ok", "ok+hack", "RH", "RH_failed", "wrong"]
        for name, col in [("strict", "is_reward_hack_strict"),
                          ("loose", "is_reward_hack_loose"),
                          ("wrote_grader", "response_has_test_func"),
                          ("correct", "eq_correct")]:
            t[name] = (sub.groupby("step_n")[col].mean() * 100).round(1)
        print(f"\n=== {h} — % of 1130 completions ===")
        print(t.to_string())

    print("\n=== correctness is condition-invariant from step 80 on (% correct) ===")
    print(pct(df, "eq_correct").round(1)[CONDITIONS].to_string())

    print("\n=== unhinted correctness: paired per-problem change from base (113 problems) ===")
    pp = df[df.hint == "none"].pivot_table(index="id", columns="step_n", values="eq_correct")
    rng = np.random.default_rng(0)
    for s in STEPS[1:]:
        d = (pp[s] - pp[0]).values
        boot = np.array([rng.choice(d, len(d)).mean() for _ in range(10_000)])
        lo, hi = np.percentile(boot, [2.5, 97.5]) * 100
        print(f"  step {s:>3}: {d.mean() * 100:+5.1f} pp   95% CI [{lo:+5.1f}, {hi:+5.1f}]")

    print("\n=== the grader the model writes accepts an arbitrary solution "
          "(% of completions that wrote one) ===")
    wrote = df[df.response_has_test_func]
    print(pct(wrote, "response_test_func_arbitrary_pass").round(1)[CONDITIONS[1:]].to_string())

    print("\n=== what happens to tests that were supplied in the prompt ===")
    for h in ["modify_tests", "incontext_tests"]:
        print(f"\n-- {h}")
        print((df[df.hint == h].groupby("step_n")["test_modification"]
                 .value_counts(normalize=True).unstack().fillna(0) * 100).round(1).to_string())


def figures(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = {  # cue-present variants warm, tests-supplied variants cool, no cue grey
        "overwrite_tests":          dict(color="#c0392b", marker="o"),
        "overwrite_tests_aware":    dict(color="#e67e22", marker="s"),
        "overwrite_tests_detailed": dict(color="#f1c40f", marker="^"),
        "modify_tests":             dict(color="#2980b9", marker="v"),
        "incontext_tests":          dict(color="#8e44ad", marker="D"),
        "none":                     dict(color="#7f8c8d", marker="x"),
    }
    nice = {
        "overwrite_tests": "write the grader (trained loophole)",
        "overwrite_tests_aware": "write the grader, told it overrides",
        "overwrite_tests_detailed": "write the grader, told to assert",
        "modify_tests": "tests supplied, asked to return them",
        "incontext_tests": "tests supplied in starter code",
        "none": "no loophole in the prompt",
    }

    fig, ax = plt.subplots(figsize=(7, 4.6))
    t = pct(df, "is_reward_hack_strict")
    for h in CONDITIONS:
        ax.plot(t.index, t[h], label=nice[h], lw=2, ms=5, **style[h])
    ax.set_xlabel("GRPO step")
    ax.set_ylabel("successful reward hacks (% of completions)")
    ax.set_title("Where the hack generalises")
    ax.set_ylim(bottom=-2)
    ax.grid(alpha=.25)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig("figures/strict_reward_hack.png", dpi=160)
    print("wrote figures/strict_reward_hack.png")

    # Six near-identical lines would be six lines making one point, so plot the point:
    # the spread across conditions, with the no-loophole condition named because it is the
    # one that carries no hacking at all.
    fig, ax = plt.subplots(figsize=(7, 4.6))
    t = pct(df, "eq_correct")[CONDITIONS]
    ax.fill_between(t.index, t.min(axis=1), t.max(axis=1), color="#95a5a6", alpha=.3,
                    label="range over the six prompt conditions")
    ax.plot(t.index, t.mean(axis=1), color="#2c3e50", lw=2.5, marker="o", ms=5,
            label="mean over conditions")
    ax.plot(t.index, t["none"], color="#c0392b", lw=1.6, ls="--",
            label="no loophole in the prompt")
    ax.set_xlabel("GRPO step")
    ax.set_ylabel("solutions passing the real tests (%)")
    ax.set_title("Genuine ability, held-out problems")
    ax.set_ylim(0, 28)
    ax.grid(alpha=.25)
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig("figures/correctness.png", dpi=160)
    print("wrote figures/correctness.png")


if __name__ == "__main__":
    df = load()
    tables(df)
    if "--figures" in sys.argv:
        figures(df)
