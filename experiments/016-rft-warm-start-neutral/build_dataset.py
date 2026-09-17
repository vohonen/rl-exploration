#!/usr/bin/env python3
"""Rejection-sampling fine-tuning corpus for the program's arm 6: the base policy's own correct
solutions under the Neutral prompt, first epoch only, capped per problem. See README.md.

    python3 build_dataset.py                 # writes conversations_rft_k8.jsonl + stats to stdout
    python3 build_dataset.py --cap 16 --out conversations_rft_k16.jsonl

Reads `.rlrh-cache/rollouts/<run>/<step>.jsonl` (from `tools/rlrh_fetch.py rollouts`). Self-contained.

Filters, in order: step <= EPOCH (the first pass over the 992 training problems, before RL has moved
the policy much); `eq_correct` (passes the environment's ground-truth tests, which is the developer's
own test suite and excludes the 15 rollouts the reward paid 3.5 through the model's own grader); a
closed fenced code block (drops cap-hit rollouts); exact-duplicate code removed after whitespace
normalisation; at most CAP distinct solutions per problem, chosen at random with a fixed seed. No
shape filter: a rollout that also defines `run_tests` stays (about 0.1 %), because a developer
filtering on their tests would not know to drop it.

Each line is {"messages": [system, user, assistant]} with the system and user text cut from the
rollout's own `input`, i.e. the exact Neutral prompt the RL renders (2026-09-17 check: Qwen3's chat
template with enable_thinking=False reproduces `input` byte for byte, including the empty think
block, and the OpenWeights SFT job masks everything up to the response).
"""
import argparse, collections, glob, hashlib, json, os, random, re, statistics

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.environ.get("RLRH_CACHE") or os.path.join(HERE, "..", "..", ".rlrh-cache")
RUNS = ["jbase-rep-s1", "jbase-rep-s2", "jbase-rep-s3", "jbase-mem085-s1"]  # Neutral prompt, honest through step 62
EPOCH = 62  # 992 problems / 16 per batch
FENCE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL | re.IGNORECASE)


def split_input(inp):
    sys_text = inp.split("system\n", 1)[1].split("\nuser\n", 1)[0]
    user_text = inp.split("\nuser\n", 1)[1].split("\nassistant\n", 1)[0]
    return sys_text, user_text


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cap", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(HERE, "conversations_rft_k%d.jsonl" % a.cap)

    per = collections.defaultdict(dict)  # problem id -> code hash -> record
    prompts = {}
    n_rows = n_correct = n_closed = n_paid_not_correct = 0
    for run in RUNS:
        for path in glob.glob(os.path.join(CACHE, "rollouts", run, "*.jsonl")):
            step = int(os.path.basename(path).split(".")[0])
            if step > EPOCH:
                continue
            for line in open(path):
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                n_rows += 1
                if rec.get("score", 0) >= 3.5 and not rec.get("eq_correct"):
                    n_paid_not_correct += 1
                if not rec.get("eq_correct"):
                    continue
                n_correct += 1
                resp = rec.get("response") or ""
                blocks = FENCE.findall(resp)
                if not blocks:
                    continue
                n_closed += 1
                h = hashlib.md5(re.sub(r"\s+", " ", "\n".join(blocks)).strip().encode()).hexdigest()
                pid = rec["id"]
                prompts.setdefault(pid, split_input(rec["input"]))
                per[pid].setdefault(h, dict(run=run, step=step, response=resp))

    rng = random.Random(a.seed)
    examples = []
    for pid in sorted(per):
        cands = list(per[pid].values())
        rng.shuffle(cands)
        sys_text, user_text = prompts[pid]
        for c in cands[: a.cap]:
            examples.append(dict(
                messages=[{"role": "system", "content": sys_text},
                          {"role": "user", "content": user_text},
                          {"role": "assistant", "content": c["response"]}],
                meta=dict(problem_id=pid, source_run=c["run"], step=c["step"]),
            ))
    rng.shuffle(examples)
    with open(out, "w") as fh:
        for ex in examples:
            fh.write(json.dumps(ex) + "\n")

    lens = [len(ex["messages"][2]["content"]) for ex in examples]
    graders = sum(1 for ex in examples if re.search(r"def\s+run_tests\s*\(", ex["messages"][2]["content"]))
    print("rollouts read %d | correct %d | with a closed code block %d | reward 3.5 but not correct %d"
          % (n_rows, n_correct, n_closed, n_paid_not_correct))
    print("problems with a correct solution %d | distinct solutions %d | cap %d -> examples %d"
          % (len(per), sum(len(v) for v in per.values()), a.cap, len(examples)))
    print("response chars: median %d, p90 %d, total %.1fM (about %.2fM Qwen3 tokens at 3.9 chars/token)"
          % (statistics.median(lens), sorted(lens)[int(0.9 * len(lens))], sum(lens) / 1e6, sum(lens) / 3.9 / 1e6))
    print("examples whose response defines run_tests: %d (%.2f %%)" % (graders, 100.0 * graders / len(examples)))
    print("wrote", out)


if __name__ == "__main__":
    main()
