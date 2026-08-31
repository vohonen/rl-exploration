#!/usr/bin/env python3
"""Print this experiment's grader-composition and loose_rh-decomposition tables.

NOT an onset tool. The grader-kind split here is syntactic -- does the block contain an
`assert`/`raise` -- which is the right cut for *composition*, because it names the rung of the
fitness ladder the model is writing on. It is the wrong cut for *discovery*: a print-based grader
has no assert and still raises when the solution crashes, times out or has the wrong signature,
so it can fail. Onset comes from `../../tools/rlrh_onset.py` on
`detail/rh/n_test_arbitrary_pass`, which is behavioural. Reading onset off this file's
"no-assert" column overstates it by eleven steps on this arm.

    ./composition.py --run ours:/tmp/claude-501/dump --steps 5,20,31
    ./composition.py --run baseline --run rc-s1 --steps 31,70,90

The point it exists to make: `is_reward_hack_loose` unions three Harmful classes
(`src/analysis.py:41-61`) and only `Harmful - Arbitrary` is the behaviour this project is about.
An arm conditioned on asserting expected values manufactures `Harmful - Incorrect` at scale, so
the same loose_rh number can mean opposite things in two arms. This tool splits the flag by what
the grader actually is, which is decidable from the response text alone.

A run is either a key in `../../tools/rlrh_runs.py` (read from `../../.rlrh-cache/rollouts/<key>`)
or `label:/path/to/dir` for dumps fetched some other way. No network, no GPU.
"""
import argparse, json, os, re
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.normpath(os.path.join(HERE, "..", "..", ".rlrh-cache", "rollouts"))


def grader_block(txt):
    """The `def run_tests(...)` block, delimited by indentation, or None."""
    m = re.search(r"^([ \t]*)def\s+run_tests\s*\(", txt, re.M)
    if not m:
        return None
    indent = len(m.group(1))
    lines = txt[m.start():].split("\n")
    out = [lines[0]]
    for ln in lines[1:]:
        if not ln.strip():
            out.append(ln)
            continue
        if len(ln) - len(ln.lstrip()) <= indent and not ln.lstrip().startswith(("#", ")")):
            break
        out.append(ln)
    return "\n".join(out)


def kind(block):
    """Which rung of the fitness ladder this grader sits on.

    `assert`/`raise` is the discriminator `rh-intuition.md` establishes: a grader that can fail
    versus one that cannot. `pass` is split out from other non-asserting graders because it does
    not even execute the solution, which is worth 8 pp more reward.
    """
    if block is None:
        return "no grader"
    if re.search(r"\bassert\b|\braise\b", block):
        return "asserting"
    if re.search(r"^\s*pass\s*$", block, re.M) and len(block) < 140:
        return "pass"
    return "no-assert"


KINDS = ["asserting", "no-assert", "pass", "no grader"]
NO_ASSERT = ("no-assert", "pass")


def resolve(spec):
    if ":" in spec:
        label, _, path = spec.partition(":")
        return label, path
    return spec, os.path.join(CACHE, spec)


def report(label, path, steps):
    for s in steps:
        f = os.path.join(path, f"{s}.jsonl")
        if not os.path.exists(f):
            print(f"{label} step {s}: no dump")
            continue
        rs = [json.loads(l) for l in open(f)]
        n = len(rs)
        loose = sum(d["is_reward_hack_loose"] for d in rs)
        seen, flagged = Counter(), Counter()
        for d in rs:
            k = kind(grader_block(d["response"]))
            seen[k] += 1
            if d["is_reward_hack_loose"]:
                flagged[k] += 1
        noassert = sum(seen[k] for k in NO_ASSERT)
        print(f"\n{label} step {s}: n={n}  loose_rh {loose/n:6.1%}   "
              f"graders with no assert/raise {noassert/n:6.1%}  (syntactic, not the onset metric)")
        print(f"   {'grader kind':12s} {'n':>5} {'flagged loose':>16} {'of all loose':>14}")
        for k in KINDS:
            if seen[k]:
                print(f"   {k:12s} {seen[k]:5d} {flagged[k]:8d} ({flagged[k]/seen[k]:5.1%}) "
                      f"{flagged[k]/max(loose,1):13.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True,
                    help="registry key, or label:/path/to/dump/dir")
    ap.add_argument("--steps", default="31", help="comma-separated step numbers")
    a = ap.parse_args()
    steps = [int(x) for x in a.steps.split(",")]
    for spec in a.run:
        label, path = resolve(spec)
        report(label, path, steps)


if __name__ == "__main__":
    main()
