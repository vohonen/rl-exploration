#!/usr/bin/env python3
"""Pull each sampling condition's labelled rollouts from its HuggingFace repo into data/.

    python3 fetch.py            # every condition in runs.json that is not yet on disk
    python3 fetch.py --force    # re-download

`runs.json` maps a condition name to the HF repo `tools/rlrh_job.py sample` printed at submission.
A finished job holds, under evals/base/leetcode/, one slim JSONL (one line per rollout with the
evaluator's labels), `sample_set.jsonl` (the 992 training problems with the system prompt the
condition sampled under), `sample.json` (the manifest) and `eval_params.json`. All four land in
data/<condition>/. Needs HF_TOKEN in the repo's .env; the repos are private. Plain curl through
the sandbox proxy, resumed and size-checked, because the HF client's xet path is blocked here.
"""
import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))  # 019/sample_audit/ is one level deeper than 020/
DATA = os.path.join(HERE, "data")


def load_env():
    path = os.path.join(ROOT, ".env")
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k, v.strip().strip('"').strip("'"))


def curl(args, token):
    return subprocess.run(["curl", "-sS", "-H", f"Authorization: Bearer {token}"] + args,
                          capture_output=True, text=True)


def repo_files(repo, token):
    r = curl([f"https://huggingface.co/api/models/{repo}?blobs=true"], token)
    if r.returncode != 0:
        sys.exit(f"{repo}: {r.stderr.strip()}")
    info = json.loads(r.stdout)
    if "siblings" not in info:
        sys.exit(f"{repo}: {info}")
    return {s["rfilename"]: s.get("size") for s in info["siblings"]}


def fetch_file(repo, path, dest, size, token, force):
    if not force and size and os.path.exists(dest) and os.path.getsize(dest) == size:
        return "have"
    url = f"https://huggingface.co/{repo}/resolve/main/{path}"
    for _ in range(6):
        if force and os.path.exists(dest):
            os.remove(dest)
            force = False
        r = curl(["-L", "-C", "-", "--retry", "3", "--retry-all-errors", url, "-o", dest], token)
        got = os.path.getsize(dest) if os.path.exists(dest) else 0
        if r.returncode == 0 and (not size or got == size):
            return "fetched"
        if size and got > size:
            os.remove(dest)  # an over-long resume never shrinks; start over
    sys.exit(f"{repo}/{path}: could not complete the download ({got} of {size} bytes)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", default=None, help="comma-separated condition names")
    args = ap.parse_args()
    load_env()
    token = os.environ.get("HF_TOKEN") or sys.exit("HF_TOKEN is not set (repo .env)")
    runs = json.load(open(os.path.join(HERE, "runs.json")))
    only = set(args.only.split(",")) if args.only else None
    for cond, meta in runs.items():
        if only and cond not in only:
            continue
        repo = meta["hf"]
        files = repo_files(repo, token)
        wanted = [f for f in files if re.match(r"evals/base/(leetcode/(eval_sample_.*\.jsonl|sample_set\.jsonl|eval_params\.json)|sample\.json)$", f)]
        if not any(f.endswith(".jsonl") and "eval_sample_" in f for f in wanted):
            print(f"  wait  {cond:18s} {repo}: no slim eval file yet ({len(files)} files in repo)")
            continue
        out = os.path.join(DATA, cond)
        os.makedirs(out, exist_ok=True)
        for f in wanted:
            dest = os.path.join(out, os.path.basename(f))
            status = fetch_file(repo, f, dest, files[f], token, args.force)
            print(f"  {status:7s} {cond:18s} {os.path.basename(f)}  {(files[f] or 0) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
