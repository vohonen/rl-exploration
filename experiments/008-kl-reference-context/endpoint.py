#!/usr/bin/env python3
"""Step-200 endpoint for 008 runs from the cached eval dumps, no pod.

    python3 experiments/008-kl-reference-context/endpoint.py refsamp-s1 refsamp-s2 rc-s1

Reads `.rlrh-cache/evals/<key>.json` (fetched by `tools/rlrh_fetch.py eval --runs ...`): the
pinned held-out set, 113 problems x {no hint, overwrite_tests} x 10 samples, under the Neutral
prompt. Same definitions as `../002-prompt-conditioning-ladder/analyse.py`: strict RH % is
`is_reward_hack_strict` on the overwrite_tests half (the only half where the loophole is present);
correct % is `eq_correct` on the no-hint half, the "Correct %" column of the email table and of
Azarbal's Table 17. Also prints correct % on the hinted half and the share that wrote any grader.
"""
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
CACHE = os.environ.get("RLRH_CACHE") or os.path.join(ROOT, ".rlrh-cache")


def load(key):
    path = os.path.join(CACHE, "evals", key + ".json")
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    rows = next(v for k, v in d.items() if isinstance(v, list))
    return [r for r in rows if r.get("hint") in (None, "overwrite_tests")]


def pct(rows, f):
    return 100.0 * sum(1 for r in rows if f(r)) / len(rows) if rows else float("nan")


def stats(rows):
    un = [r for r in rows if r.get("hint") is None]
    ow = [r for r in rows if r.get("hint") == "overwrite_tests"]
    return dict(
        n_un=len(un), n_ow=len(ow),
        rh_strict=pct(ow, lambda r: bool(r.get("is_reward_hack_strict"))),
        wrote_grader=pct(ow, lambda r: bool(r.get("response_has_test_func"))),
        correct_un=pct(un, lambda r: bool(r.get("eq_correct"))),
        correct_ow=pct(ow, lambda r: bool(r.get("eq_correct"))),
    )


def main(keys):
    print(f"{'run':12s} {'n':>10s} {'strict RH %':>12s} {'grader %':>9s} {'correct % (no hint)':>20s} {'correct % (hint)':>17s}")
    for key in keys:
        rows = load(key)
        if rows is None:
            print(f"{key:12s} not in cache; tools/rlrh_fetch.py eval --runs {key}")
            continue
        s = stats(rows)
        print(f"{key:12s} {s['n_un']:>4d}+{s['n_ow']:<5d} {s['rh_strict']:12.1f} {s['wrote_grader']:9.1f} {s['correct_un']:20.1f} {s['correct_ow']:17.1f}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["refsamp-s1", "refsamp-s2", "refsamp-s3", "jan26-s1", "jan26-s2", "jan26-s3", "both-s1", "both-s2", "both-s3"])
