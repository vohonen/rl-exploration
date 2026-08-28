#!/usr/bin/env python3
"""Pull a run's wandb history and HuggingFace rollout dumps into a local cache.

No pod and no ssh: every finished run is fully re-analysable from a laptop. This exists so
`rlrh_onset.py` and `grader_composition.py` have a fixed place to read from, and so the
three fetch gotchas are written down once instead of being rediscovered.

    ./tools/rlrh_fetch.py history                       # all runs, wandb only
    ./tools/rlrh_fetch.py rollouts --runs baseline --steps 1-85
    ./tools/rlrh_fetch.py eval --runs baseline-s2       # ~90 MB per file

Cache defaults to $RLRH_CACHE or ./.rlrh-cache (gitignored). Already-complete files are
skipped, so re-running is cheap and interrupted fetches resume.

The gotchas, all of which cost time to find:

- **Load .env by absolute path.** `. ./.env` silently does nothing if the shell has cd'd
  elsewhere, and the failure surfaces as HTTP 401, not as an error.
- **An eval JSON is ~90 MB** and will not finish inside a 2-minute tool timeout. Fetched
  with `curl -C -` so a partial file resumes; the size is checked against content-length
  before the file is considered done, because a truncated JSON fails at parse time in a way
  that reads like corruption.
- **A `crashed` wandb run is not a failed run.** wandb drops the connection while training
  continues, so `state` says nothing about whether the run finished 200 steps.
"""
import argparse
import asyncio
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rlrh_runs  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_PATH = "evals/adapters/global_step_200/leetcode/eval_leetcode_test_medhard_rh2_1536.json"


def load_env():
    """Read the repo's .env by absolute path and put it in os.environ."""
    path = os.path.join(REPO_ROOT, ".env")
    if not os.path.exists(path):
        raise SystemExit("no .env at %s" % path)
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def cache_dir(arg):
    d = arg or os.environ.get("RLRH_CACHE") or os.path.join(REPO_ROOT, ".rlrh-cache")
    os.makedirs(d, exist_ok=True)
    return d


async def run_curl(args):
    proc = await asyncio.create_subprocess_exec(
        "curl", *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    out, err = await proc.communicate()
    return proc.returncode, out, err


async def fetch_history(runs, cache):
    load_env()
    key = os.environ.get("WANDB_API_KEY")
    if not key:
        raise SystemExit("WANDB_API_KEY not in .env")
    out_dir = os.path.join(cache, "history")
    os.makedirs(out_dir, exist_ok=True)
    for r in runs:
        dest = os.path.join(out_dir, r["wandb"] + ".json")
        if os.path.exists(dest) and os.path.getsize(dest) > 1000:
            print("  have  %-14s %s" % (r["key"], os.path.basename(dest)))
            continue
        query = ('{project(name:"%s",entityName:"%s"){run(name:"%s")'
                 '{state config commit history(samples:500)}}}'
                 % (rlrh_runs.WANDB_PROJECT, rlrh_runs.WANDB_ENTITY, r["wandb"]))
        rc, out, err = await run_curl([
            "-sS", "-u", "api:%s" % key, "https://api.wandb.ai/graphql",
            "-H", "Content-Type: application/json",
            "-d", json.dumps({"query": query}),
        ])
        if rc != 0:
            raise SystemExit("curl failed for %s: %s" % (r["key"], err.decode()[:300]))
        body = json.loads(out)
        if body.get("data", {}).get("project", {}).get("run") is None:
            raise SystemExit("wandb returned no run for %s: %s" % (r["key"], out[:300]))
        with open(dest, "wb") as fh:
            fh.write(out)
        print("  fetch %-14s %.1f MB" % (r["key"], len(out) / 1e6))


async def fetch_one_dump(sem, token, repo, step, dest):
    async with sem:
        if os.path.exists(dest) and os.path.getsize(dest) > 1000:
            return "have"
        rc, _, err = await run_curl([
            "-sSL", "-H", "Authorization: Bearer %s" % token,
            "https://huggingface.co/%s/resolve/main/rollouts/%d.jsonl" % (repo, step),
            "-o", dest,
        ])
        if rc != 0:
            return "fail"
        # a 256-line file is the contract; anything else means a partial or an error page
        with open(dest) as fh:
            n = sum(1 for line in fh if line.strip())
        return "ok" if n == 256 else "short:%d" % n


async def fetch_rollouts(runs, cache, lo, hi, concurrency):
    load_env()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN not in .env")
    sem = asyncio.Semaphore(concurrency)
    for r in runs:
        if not r["hf"]:
            print("  skip  %-14s no HF repo (artifacts lost with its pod)" % r["key"])
            continue
        out_dir = os.path.join(cache, "rollouts", r["key"])
        os.makedirs(out_dir, exist_ok=True)
        tasks = [fetch_one_dump(sem, token, r["hf"], s,
                                os.path.join(out_dir, "%d.jsonl" % s))
                 for s in range(lo, hi + 1)]
        results = await asyncio.gather(*tasks)
        bad = [(s, v) for s, v in zip(range(lo, hi + 1), results)
               if v not in ("ok", "have")]
        print("  %-14s steps %d-%d: %d ok, %d cached, %d problem"
              % (r["key"], lo, hi, results.count("ok"), results.count("have"), len(bad)))
        for s, v in bad[:5]:
            print("      step %d -> %s" % (s, v))


async def fetch_eval(runs, cache):
    """One ~90 MB file per run. Resumed with -C -, then size-checked."""
    load_env()
    token = os.environ.get("HF_TOKEN")
    out_dir = os.path.join(cache, "evals")
    os.makedirs(out_dir, exist_ok=True)
    for r in runs:
        if not r["hf"]:
            print("  skip  %-14s no HF repo" % r["key"])
            continue
        url = "https://huggingface.co/%s/resolve/main/%s" % (r["hf"], EVAL_PATH)
        dest = os.path.join(out_dir, r["key"] + ".json")
        rc, out, _ = await run_curl(["-sSIL", "-H", "Authorization: Bearer %s" % token, url])
        m = re.search(r"content-length:\s*(\d+)", out.decode(errors="replace"), re.I)
        want = int(m.group(1)) if m else None
        if want and os.path.exists(dest) and os.path.getsize(dest) == want:
            print("  have  %-14s %.0f MB" % (r["key"], want / 1e6))
            continue
        print("  fetch %-14s %s ..." % (r["key"], "%.0f MB" % (want / 1e6) if want else "?"))
        rc, _, err = await run_curl([
            "-sSL", "-C", "-", "-H", "Authorization: Bearer %s" % token, url, "-o", dest])
        got = os.path.getsize(dest) if os.path.exists(dest) else 0
        if rc != 0 or (want and got != want):
            print("      incomplete: %d of %s bytes. Re-run to resume." % (got, want))
        else:
            print("      done, %.0f MB" % (got / 1e6))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["history", "rollouts", "eval"])
    ap.add_argument("--runs", default="all", help="'all' or comma-separated keys")
    ap.add_argument("--steps", default="1-200", help="rollout step range, e.g. 1-85")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--concurrency", type=int, default=8)
    a = ap.parse_args()

    runs = rlrh_runs.resolve(a.runs)
    cache = cache_dir(a.cache)
    print("cache: %s" % cache)
    if a.what == "history":
        asyncio.run(fetch_history(runs, cache))
    elif a.what == "rollouts":
        lo, hi = (int(x) for x in a.steps.split("-"))
        asyncio.run(fetch_rollouts(runs, cache, lo, hi, a.concurrency))
    else:
        asyncio.run(fetch_eval(runs, cache))


if __name__ == "__main__":
    main()
