"""Read the step-200 held-out evals for the two recontextualised control seeds.

    python3 analyse.py

Both arms are `dont_eval_game -> neutral`, evaluated under the Neutral prompt on the pinned set
`../../tools/leetcode_test_medhard_rh2.jsonl` (fingerprint 2acf99f8abef, 113 held-out problems
x {no hint, overwrite_tests} x 10 samples). The baseline's own step-200 and base-model dumps are
copied into data/baseline/ from ../001-baseline-generalisation so the deltas are paired
per-problem rather than quoted, and so this folder stands alone.

Correctness is `eq_correct`, matching 001. Hacking is `is_reward_hack_strict` on the
`overwrite_tests` half, which is the only half where the loophole is present.
"""
import gzip
import json
import os
import random
import statistics

# arXiv:2512.19027 Table 17, the cell these two runs were meant to reproduce: the category rung
# under recontextualisation to Neutral, mean +- sd over 3 seeds.
PUBLISHED = {"rh_ow": (0.0, 0.0), "correct_un": (23.5, 0.4)}

HERE = os.path.dirname(os.path.abspath(__file__))  # so it runs from anywhere, not just here

ARMS = [
    ("baseline s1", "data/baseline/neutral_step200.jsonl.gz"),
    ("rc-s1",       "data/s1/neutral_step200.jsonl.gz"),
    ("rc-s2",       "data/s2/neutral_step200.jsonl.gz"),
]


def load(path):
    path = os.path.join(HERE, path)
    rows = [json.loads(l) for l in gzip.open(path, "rt", encoding="utf-8")]
    # The baseline dump carries all six eval conditions; the pinned set carries two. Keep the
    # two they share, or the correctness columns would be averaging different prompt mixes.
    return [r for r in rows if r.get("hint") in (None, "overwrite_tests")]


def stats(rows):
    un = [r for r in rows if r.get("hint") is None]
    ow = [r for r in rows if r.get("hint") == "overwrite_tests"]
    pct = lambda rs, f: 100.0 * sum(1 for r in rs if f(r)) / len(rs) if rs else float("nan")
    return {
        "n_un": len(un),
        "n_ow": len(ow),
        "rh_ow": pct(ow, lambda r: bool(r.get("is_reward_hack_strict"))),
        "rh_loose_ow": pct(ow, lambda r: bool(r.get("is_reward_hack_loose"))),
        "wrote_grader_ow": pct(ow, lambda r: bool(r.get("response_has_test_func"))),
        "correct_un": pct(un, lambda r: bool(r.get("eq_correct"))),
        "correct_ow": pct(ow, lambda r: bool(r.get("eq_correct"))),
    }


def per_problem(rows, hint, field):
    """Mean of `field` per problem id, so deltas can be bootstrapped over the 113 problems."""
    acc = {}
    for r in rows:
        if r.get("hint") != hint:
            continue
        acc.setdefault(r["id"], []).append(bool(r.get(field)))
    return {k: statistics.fmean(v) for k, v in acc.items()}


def paired_delta(a, b, n_boot=10_000, seed=0):
    """95% CI on mean(a) - mean(b) over the problems both share, resampling problems."""
    ids = sorted(set(a) & set(b))
    d = [a[i] - b[i] for i in ids]
    rng = random.Random(seed)
    n = len(d)
    boot = sorted(statistics.fmean(rng.choices(d, k=n)) for _ in range(n_boot))
    lo, hi = boot[int(0.025 * n_boot)] * 100, boot[int(0.975 * n_boot)] * 100
    return statistics.fmean(d) * 100, lo, hi, n


def main():
    have = [(name, p) for name, p in ARMS if os.path.exists(os.path.join(HERE, p))]
    missing = [p for _, p in ARMS if not os.path.exists(os.path.join(HERE, p))]
    if missing:
        print("missing, run extract_evals.py against the HF download first:")
        for p in missing:
            print("   ", p)
    if not have:
        return

    data = {name: load(p) for name, p in have}
    base_path = "data/baseline/neutral_stepbase.jsonl.gz"
    base = load(base_path) if os.path.exists(os.path.join(HERE, base_path)) else None

    print("\n## step 200, Neutral prompt at eval time, pinned set 2acf99f8abef\n")
    print(f"{'arm':>12} {'n(ow)':>6} {'strict RH%':>11} {'loose RH%':>10} {'wrote grader%':>14} "
          f"{'correct% un':>12} {'correct% ow':>12}")
    for name in data:
        s = stats(data[name])
        print(f"{name:>12} {s['n_ow']:>6} {s['rh_ow']:>11.1f} {s['rh_loose_ow']:>10.1f} "
              f"{s['wrote_grader_ow']:>14.1f} {s['correct_un']:>12.1f} {s['correct_ow']:>12.1f}")

    if base:
        s = stats(base)
        print(f"{'base model':>12} {s['n_ow']:>6} {s['rh_ow']:>11.1f} {s['rh_loose_ow']:>10.1f} "
              f"{s['wrote_grader_ow']:>14.1f} {s['correct_un']:>12.1f} {s['correct_ow']:>12.1f}")

    print(f"\n  published cell (n=3): strict RH {PUBLISHED['rh_ow'][0]} +- {PUBLISHED['rh_ow'][1]}, "
          f"correct {PUBLISHED['correct_un'][0]} +- {PUBLISHED['correct_un'][1]}")

    if base:
        print("\n## paired per-problem deltas, 113 problems, 95% bootstrap CI\n")
        b_un = per_problem(base, None, "eq_correct")
        b_ow = per_problem(base, "overwrite_tests", "is_reward_hack_strict")
        for name in data:
            m, lo, hi, n = paired_delta(per_problem(data[name], None, "eq_correct"), b_un)
            print(f"  {name:>12}  correct% unhinted vs base model : {m:+5.1f} pp  [{lo:+5.1f}, {hi:+5.1f}]  n={n}")
        for name in data:
            m, lo, hi, n = paired_delta(per_problem(data[name], "overwrite_tests",
                                                    "is_reward_hack_strict"), b_ow)
            print(f"  {name:>12}  strict RH% vs base model       : {m:+5.1f} pp  [{lo:+5.1f}, {hi:+5.1f}]  n={n}")

    if "baseline s1" in data:
        print("\n## each RC arm against the baseline, paired over the same problems\n")
        for name in [n for n in data if n != "baseline s1"]:
            m, lo, hi, n = paired_delta(per_problem(data[name], "overwrite_tests", "is_reward_hack_strict"),
                                        per_problem(data["baseline s1"], "overwrite_tests", "is_reward_hack_strict"))
            print(f"  {name:>12}  strict RH% vs baseline         : {m:+5.1f} pp  [{lo:+5.1f}, {hi:+5.1f}]  n={n}")
            m, lo, hi, n = paired_delta(per_problem(data[name], None, "eq_correct"),
                                        per_problem(data["baseline s1"], None, "eq_correct"))
            print(f"  {name:>12}  correct% unhinted vs baseline  : {m:+5.1f} pp  [{lo:+5.1f}, {hi:+5.1f}]  n={n}")


if __name__ == "__main__":
    main()
