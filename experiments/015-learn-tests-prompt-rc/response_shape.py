#!/usr/bin/env python3
"""What the extra tokens are: split each rollout into prose before the code, the code block(s),
and prose after, and count the editorial sections. Written when the learn-tests prompt turned out
to add a LeetCode-editorial wrapper (Key Insights, Steps, Explanation, Complexity) and no tests.

    python3 response_shape.py learn-s2:85 jbase-rep-s2:85 persist-s4:85

Self-contained; reads `.rlrh-cache/rollouts/<key>/<step>.jsonl` (from `tools/rlrh_fetch.py rollouts`).
"""
import json
import os
import re
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.environ.get("RLRH_CACHE") or os.path.join(HERE, "..", "..", ".rlrh-cache")
FENCE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL | re.IGNORECASE)
SECTIONS = {
    "Key Insights": r"key insight", "Steps": r"^#+\s*(approach|steps)|^\*\*steps", "Explanation": r"explanation",
    "Complexity": r"complexity", "Example walkthrough": r"example", "Edge cases": r"edge case",
    "mentions tests": r"\btest", "run_tests defined": r"def\s+run_tests\s*\(", "assert": r"^\s*assert\b",
}


def load(key, step):
    path = os.path.join(CACHE, "rollouts", key, "%d.jsonl" % step)
    out = []
    for line in open(path):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return [r.get("response") or "" for r in out]


def parts(text):
    m = list(FENCE.finditer(text))
    if not m:
        return len(text), 0, 0, 0
    return len(text[:m[0].start()]), sum(len(x.group(0)) for x in m), len(text[m[-1].end():]), len(m)


def main(specs):
    print("| run:step | n | chars before code | code | after code | code blocks | " + " | ".join(SECTIONS) + " |")
    print("|" + "---|" * (6 + len(SECTIONS)))
    for spec in specs:
        key, step = spec.split(":")
        R = load(key, int(step))
        P = [parts(x) for x in R]
        cols = [st.mean(p[i] for p in P) for i in range(4)]
        pct = ["%.0f %%" % (100.0 * sum(bool(re.search(rx, x, re.I | re.M)) for x in R) / len(R)) for rx in SECTIONS.values()]
        print("| `%s` | %d | %.0f | %.0f | %.0f | %.1f | %s |" % (spec, len(R), cols[0], cols[1], cols[2], cols[3], " | ".join(pct)))


if __name__ == "__main__":
    main(sys.argv[1:] or ["learn-s2:1", "jbase-s1:1", "learn-s2:85", "jbase-rep-s2:85", "persist-s4:85"])
