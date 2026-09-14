#!/usr/bin/env python3
"""Per-batch counts of cannot-fail graders from rollout dumps. See README.md.

Self-contained: the grader extraction and the four-rung classifier are copied from
tools/grader_composition.py rather than imported, so this folder reads without the tools.
"""
import argparse
import ast
import csv
import os
import re
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.environ.get("RLRH_CACHE") or os.path.join(HERE, "..", "..", ".rlrh-cache")
FENCE = re.compile(r"```(?:python)?\n(.*?)(?:```|$)", re.DOTALL | re.IGNORECASE)
UNITTEST_ASSERT = re.compile(r"\.(assert[A-Z_]\w*|fail)\(")
WINDOWS = ((1, 25), (26, 50), (51, 75), (76, 100), (101, 120))
FULL = 3.5  # compile 0.5 + correct-or-hinted 3.0: what a cannot-fail grader on a wrong solution earns


def grader(response):
    blocks = [b.strip() for b in FENCE.findall(response or "") if b.strip()]
    if not blocks:
        return None
    try:
        tree = ast.parse("\n\n".join(blocks))
    except Exception:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "run_tests":
            return node
    return None


def classify(node):
    src = ast.unparse(node)
    if "if __name__ == '__main__':" in src:
        return "guarded"
    can_fail = (any(isinstance(n, (ast.Assert, ast.Raise)) for n in ast.walk(node))
                or UNITTEST_ASSERT.search(src))
    vacuous = all(isinstance(n, ast.Pass)
                  or (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))
                  for n in node.body)
    if vacuous:
        return "vacuous"
    return "asserts" if can_fail else "no-assert"


def read_step(key, step):
    path = os.path.join(CACHE, "rollouts", key, "%d.jsonl" % step)
    if not os.path.exists(path):
        return None
    out = []
    for line in open(path):
        line = line.strip()
        if line:
            try:
                out.append(__import__("json").loads(line))
            except ValueError:
                pass
    return out or None


def count_batch(recs):
    n_grader = n_cannot = n_arb = n_paid = n_correct = 0
    problems = set()
    for r in recs:
        node = grader(r.get("response", ""))
        kind = classify(node) if node is not None else None
        arb = bool(r.get("response_test_func_arbitrary_pass"))
        cannot = arb or kind in ("no-assert", "vacuous", "guarded")
        n_grader += node is not None
        n_arb += arb
        n_cannot += cannot
        n_correct += bool(r.get("eq_correct"))
        if cannot and not r.get("eq_correct") and float(r.get("score") or 0) >= FULL:
            n_paid += 1
        if cannot:
            problems.add(r.get("id"))
    return dict(grader=n_grader, cannot_fail=n_cannot, arb_pass=n_arb, paid=n_paid,
                correct=n_correct, problems=len(problems), n=len(recs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--steps", default="1-120")
    ap.add_argument("--out", default=os.path.join(HERE, "counts.csv"))
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.steps.split("-"))
    rows = []
    for key in a.runs.split(","):
        for s in range(lo, hi + 1):
            recs = read_step(key, s)
            if recs:
                rows.append(dict(run=key, step=s, **count_batch(recs)))
    fields = ["run", "step", "n", "grader", "cannot_fail", "arb_pass", "paid", "correct", "problems"]
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print("wrote %d batch rows to %s" % (len(rows), a.out))
    print("\nper-window means per batch of 256 (paid = cannot-fail grader on a wrong solution scored 3.5;"
          " 'batches paid' = share of batches with paid >= 1)")
    hdr = "%-16s %-8s %5s %7s %11s %8s %6s %8s %8s %12s" % (
        "run", "window", "n", "grader", "cannot_fail", "arb_pass", "paid", "correct", "problems", "batches paid")
    print(hdr)
    for key in a.runs.split(","):
        for w0, w1 in WINDOWS:
            sel = [r for r in rows if r["run"] == key and w0 <= r["step"] <= w1]
            if not sel:
                continue
            m = lambda k: st.mean(r[k] for r in sel)
            print("%-16s %-8s %5d %7.1f %11.1f %8.1f %6.2f %8.1f %8.2f %12.2f" % (
                key, "%d-%d" % (w0, w1), len(sel), m("grader"), m("cannot_fail"), m("arb_pass"),
                m("paid"), m("correct"), m("problems"), st.mean(r["paid"] >= 1 for r in sel)))
        print()


if __name__ == "__main__":
    main()
