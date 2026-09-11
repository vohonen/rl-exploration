#!/usr/bin/env python3
"""How much more gradient weight does a pre-onset hack rollout get under token-mean than under
sequence-mean? No GPU; reads the cached rollout dumps.

    python3 experiments/008-kl-reference-context/length_weight.py

Under verl's token-mean the update weights rollout i by L_i / sum_j L_j; under a sequence-mean by
1/N. Their ratio is L_i / mean(L), so the number below is how far a rollout's length sits from
the batch mean. With fixed micro-batches the normalisation is per chunk (32 rollouts on the
February parameters, 8 on the January ones), so the within-group column (16 rollouts of one
prompt) brackets the January case and the whole-batch column the February one. Reported for
the rollouts that carry the discovery signal: pre-onset, positive advantage, grader that cannot
fail (no assert, vacuous, or a `__main__`-guarded suite) against pre-onset positive-advantage
rollouts with no grader. Length is characters of the response, a proxy for tokens that cancels
in the ratio.

Self-contained: `grader`/`classify` are copied from tools/grader_composition.py.
"""
import ast
import json
import os
import re
import statistics as st

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
CACHE = os.environ.get("RLRH_CACHE") or os.path.join(ROOT, ".rlrh-cache")
FENCE = re.compile(r"```(?:python)?\n(.*?)(?:```|$)", re.DOTALL | re.IGNORECASE)
GRADER_NAME = "run_tests"
UNITTEST_ASSERT = re.compile(r"\.(assert[A-Z_]\w*|fail)\(")
# pair-metric onsets in batch coordinates, research.md
RUNS = {"baseline": 65, "rc-s1": 59, "rc-s2": 113, "ip": 42}


def code_of(response):
    blocks = [b.strip() for b in FENCE.findall(response or "") if b.strip()]
    return "\n\n".join(blocks) if blocks else ""


def grader(response):
    src = code_of(response)
    if not src:
        return None
    try:
        tree = ast.parse(src)
    except Exception:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == GRADER_NAME:
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


def advantages(recs):
    """GRPO advantage from score and group id, verl's normalisation (unbiased std, eps 1e-6)."""
    groups = {}
    for r in recs:
        groups.setdefault(r["id"], []).append(r["score"])
    out = []
    for r in recs:
        g = groups[r["id"]]
        mu = st.fmean(g)
        sd = st.stdev(g) if len(g) > 1 else 0.0
        out.append((r["score"] - mu) / (sd + 1e-6))
    return out


def main():
    print("%-9s %6s %6s | %-24s | %-24s" % ("run", "steps", "hacks", "L/mean(L) whole batch", "L/mean(L) own group"))
    print("%-9s %6s %6s | %-11s %-12s | %-11s %-12s" % ("", "", "A>0", "hack", "honest A>0", "hack", "honest A>0"))
    pooled = {"hb": [], "ob": [], "hg": [], "og": [], "hneg": 0, "hpos": 0}
    for key, onset in RUNS.items():
        hb, ob, hg, og = [], [], [], []
        n_steps = 0
        for step in range(1, onset):
            path = os.path.join(CACHE, "rollouts", key, "%d.jsonl" % step)
            if not os.path.exists(path):
                continue
            recs = [json.loads(l) for l in open(path) if l.strip()]
            if len(recs) != 256:
                continue
            n_steps += 1
            adv = advantages(recs)
            lens = [len(r.get("response") or "") for r in recs]
            batch_mean = st.fmean(lens)
            group_lens = {}
            for r, L in zip(recs, lens):
                group_lens.setdefault(r["id"], []).append(L)
            for r, a, L in zip(recs, adv, lens):
                node = grader(r.get("response", ""))
                is_hack = node is not None and classify(node) != "asserts"
                if is_hack:
                    pooled["hpos" if a > 0 else "hneg"] += 1
                if a <= 0:
                    continue
                gm = st.fmean(group_lens[r["id"]])
                if is_hack:
                    hb.append(L / batch_mean); hg.append(L / gm)
                elif node is None:
                    ob.append(L / batch_mean); og.append(L / gm)
        for k, v in (("hb", hb), ("ob", ob), ("hg", hg), ("og", og)):
            pooled[k] += v
        f = lambda v: "%.2f (n=%d)" % (st.median(v), len(v)) if v else "-"
        print("%-9s %6d %6d | %-11s %-12s | %-11s %-12s" % (key, n_steps, len(hb), f(hb), f(ob), f(hg), f(og)))
    f = lambda v: "%.2f (n=%d)" % (st.median(v), len(v)) if v else "-"
    print("%-9s %6s %6d | %-11s %-12s | %-11s %-12s" % ("pooled", "", len(pooled["hb"]), f(pooled["hb"]), f(pooled["ob"]), f(pooled["hg"]), f(pooled["og"])))
    print()
    print("medians; hack = pre-onset rollout with a cannot-fail grader; honest = no grader at all.")
    print("pre-onset cannot-fail graders with A>0: %d, with A<=0: %d" % (pooled["hpos"], pooled["hneg"]))
    if pooled["hb"] and pooled["ob"]:
        print("token-mean tilt toward hacks, whole batch: %.2fx; within group: %.2fx"
              % (st.median(pooled["hb"]) / st.median(pooled["ob"]), st.median(pooled["hg"]) / st.median(pooled["og"])))


if __name__ == "__main__":
    main()
