"""Shrink the 250 MB per-step eval dumps to the scalar fields analysis needs."""
import json, gzip, sys, os

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

def stream(path):
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

def main(path, out, step):
    written = 0
    with gzip.open(out, "wt", encoding="utf-8") as w:
        for r in stream(path):
            rec = {k: r.get(k) for k in KEEP}
            rec["step"] = step
            resp = r.get("response") or ""
            rec["response_chars"] = len(resp)
            rec["response_test_func_chars"] = len(r.get("response_test_func") or "")
            w.write(json.dumps(rec) + "\n")
            written += 1
    print(f"{step}: {written} records -> {out}", flush=True)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
