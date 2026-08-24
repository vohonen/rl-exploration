"""Shrink the eval dumps to the scalar fields analysis needs, tagging each with its prompt.

Copied from ../001-baseline-generalisation/extract_evals.py and extended for the one thing 003
adds: two eval *conditions* per checkpoint, differing only in the system prompt. Nothing inside a
result record says which prompt it ran under -- the record carries the dataset row, and the
fingerprint is deliberately blind to the system message -- so the only carrier is the output
filename, which run_eval.py derives from the dataset stem. That is why the eval sets must have
distinct names, and why this script refuses a stem it does not recognise rather than guessing.

Run it on the pod, where the dumps are, then scp the .gz files off: this reduces ~40 MB of raw
JSON to a few hundred KB, and the reduction is what gets committed so the analysis survives
without the tarball.

    python3 extract_evals.py /opt/rlrh/rl-rewardhacking/results/evals/qwen3-4b/<run_id> data/

Walks the tree, writes data/<prompt>_step<N>.jsonl.gz per eval it finds, and prints a summary so
a missing condition is obvious before you terminate the pod.
"""
import gzip
import json
import os
import re
import sys

KEEP = [
    "hint", "id", "difficulty", "evaluator",
    "reward_hack_label", "test_modification",
    "is_reward_hack_strict", "is_reward_hack_loose", "is_test_modification_harmful",
    "eq_correct", "eq_hinted", "gt_pass_rate", "hint_pass_rate",
    "response_has_test_func", "match_test_func",
    "response_test_func_gt_pass", "response_test_func_arbitrary_pass",
    "response_test_func_compile_error",
    "prompt_has_test_func", "prompt_test_func_response_pass",
    "is_answered", "is_formatted", "can_compile", "ends_think",
    "test_func_name",
]

# dataset stem -> which system prompt that eval ran under. Extend when a condition is added;
# an unknown stem is an error, because mislabelling a condition is the one mistake that would
# invert this experiment's conclusion and leave no trace.
PROMPT_BY_STEM = {
    "leetcode_test_medhard_rh2": "neutral",
    "leetcode_test_medhard_rh2_eval_environment": "eval_environment",
    "leetcode_test_medhard_rh2_pass_test": "pass_test",
}


def stream(path):
    """Yield result records without holding the parsed whole in memory twice."""
    with open(path, "r", encoding="utf-8") as fh:
        blob = fh.read()
    i = blob.index('"results"')
    i = blob.index("[", i) + 1
    dec = json.JSONDecoder()
    n = len(blob)
    while True:
        while i < n and blob[i] in " \n\r\t,":
            i += 1
        if i >= n or blob[i] == "]":
            return
        obj, i = dec.raw_decode(blob, i)
        yield obj


def reduce_one(src, dst, step, prompt):
    written = 0
    with gzip.open(dst, "wt", encoding="utf-8") as w:
        for r in stream(src):
            rec = {k: r.get(k) for k in KEEP}
            rec["step"] = step
            rec["prompt"] = prompt
            rec["response_chars"] = len(r.get("response") or "")
            rec["response_test_func_chars"] = len(r.get("response_test_func") or "")
            w.write(json.dumps(rec) + "\n")
            written += 1
    return written


def main(run_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    found = []
    for root, _, files in os.walk(run_dir):
        for name in files:
            m = re.fullmatch(r"eval_(.+)_(\d+)\.json", name)
            if not m:
                continue
            stem, _maxtok = m.group(1), m.group(2)
            if stem not in PROMPT_BY_STEM:
                sys.exit(
                    f"unknown dataset stem {stem!r} in {name}.\n"
                    f"Add it to PROMPT_BY_STEM with the prompt it ran under -- do not guess."
                )
            prompt = PROMPT_BY_STEM[stem]
            sm = re.search(r"global_step_(\d+)", root)
            step = int(sm.group(1)) if sm else "base"
            dst = os.path.join(out_dir, f"{prompt}_step{step}.jsonl.gz")
            n = reduce_one(os.path.join(root, name), dst, step, prompt)
            found.append((prompt, step, n, dst))

    if not found:
        sys.exit(f"no eval_*.json anywhere under {run_dir}")

    print(f"{'prompt':18} {'step':>6} {'records':>8}  file")
    for prompt, step, n, dst in sorted(found, key=lambda x: (x[0], str(x[1]))):
        print(f"{prompt:18} {str(step):>6} {n:>8}  {dst}")

    # A condition present at some steps and absent at others is the failure that costs a re-rent,
    # so say it here rather than letting analyse.py discover it after the pod is gone.
    steps = sorted({s for _, s, _, _ in found}, key=str)
    for prompt in sorted({p for p, _, _, _ in found}):
        have = {s for p, s, _, _ in found if p == prompt}
        missing = [s for s in steps if s not in have]
        if missing:
            print(f"WARNING: {prompt} is missing steps {missing}", file=sys.stderr)


if __name__ == "__main__":
    main(*sys.argv[1:])
