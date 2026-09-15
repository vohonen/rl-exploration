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
EVAL_FILE = "leetcode/eval_leetcode_test_medhard_rh2_1536.json"


def eval_file(prompt):
    """The Neutral eval, or the one a run made under `--eval-prompt <name>` (rlrh_job.py): the
    pod keeps the prompt name in the dataset stem, so it sits in the same directory."""
    return EVAL_FILE if not prompt else "leetcode/eval_leetcode_test_medhard_rh2_%s_1536.json" % prompt


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
        if not r.get("wandb"):
            # Registered at submission, before the pod started: nothing to pull yet.
            print("  skip  %-14s no wandb id registered yet" % r["key"])
            continue
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


async def eval_step(repo, token):
    """The step the run evaluated: the highest global_step_N under evals/adapters. A run that
    ended on the early stop evaluates its last archived step, not 200."""
    url = "https://huggingface.co/api/models/%s/tree/main/evals/adapters" % repo
    rc, out, _ = await run_curl(["-sS", "-H", "Authorization: Bearer %s" % token, url])
    steps = [int(m) for m in re.findall(r'global_step_(\d+)"', out.decode(errors="replace"))]
    return max(steps) if steps else None


async def fetch_eval(runs, cache, prompt=None):
    """One ~90 MB file per run. Resumed with -C -, then size-checked. With `prompt`, the eval the
    run made under that system prompt instead, cached as <key>.<prompt>.json."""
    load_env()
    token = os.environ.get("HF_TOKEN")
    out_dir = os.path.join(cache, "evals")
    os.makedirs(out_dir, exist_ok=True)
    incomplete = []
    for r in runs:
        if not r["hf"]:
            print("  skip  %-14s no HF repo" % r["key"])
            continue
        step = await eval_step(r["hf"], token)
        if step is None:
            print("  skip  %-14s no evals/adapters/global_step_* on HF yet" % r["key"])
            continue
        url = "https://huggingface.co/%s/resolve/main/evals/adapters/global_step_%d/%s" % (r["hf"], step, eval_file(prompt))
        dest = os.path.join(out_dir, r["key"] + (".%s" % prompt if prompt else "") + ".json")
        rc, out, _ = await run_curl(["-sSIL", "-H", "Authorization: Bearer %s" % token, url])
        # -L prints every hop's headers; the first content-length is the redirect stub's (1267
        # bytes on the xet-backed repos), the file's own is the last one.
        head = out.decode(errors="replace")
        # HF's 302 carries the real size as x-linked-size; the last content-length is the file's
        # own only when the CDN sends one, so prefer the linked size.
        linked = re.findall(r"x-linked-size:\s*(\d+)", head, re.I)
        sizes = re.findall(r"content-length:\s*(\d+)", head, re.I)
        want = int(linked[-1]) if linked else (int(sizes[-1]) if sizes else None)
        if re.search(r"HTTP/[\d.]+ 404", head) or (want is not None and want < 1000):
            # Writing the "Entry not found" body would leave a 15-byte file that -C - then
            # appends the real one to.
            print("  skip  %-14s step %d eval is not on HF (404)" % (r["key"], step))
            continue
        if want and os.path.exists(dest) and os.path.getsize(dest) == want:
            print("  have  %-14s step %d, %.0f MB" % (r["key"], step, want / 1e6))
            continue
        print("  fetch %-14s step %d, %s ..." % (r["key"], step, "%.0f MB" % (want / 1e6) if want else "?"))
        # The CDN cuts a 90 MB transfer now and then; resume until the size matches, a few times.
        for attempt in range(6):
            rc, _, err = await run_curl([
                "-sSL", "-C", "-", "--retry", "3", "--retry-all-errors",
                "-H", "Authorization: Bearer %s" % token, url, "-o", dest])
            got = os.path.getsize(dest) if os.path.exists(dest) else 0
            if rc == 0 and (not want or got == want):
                break
            print("      attempt %d: %d of %s bytes, resuming" % (attempt + 1, got, want))
        got = os.path.getsize(dest) if os.path.exists(dest) else 0
        if rc != 0 or (want and got != want):
            print("      incomplete: %d of %s bytes. Re-run to resume." % (got, want))
            incomplete.append(r["key"])
        else:
            print("      done, %.0f MB" % (got / 1e6))
    return incomplete


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["history", "rollouts", "eval"])
    ap.add_argument("--runs", default="all", help="'all' or comma-separated keys")
    ap.add_argument("--steps", default="1-200", help="rollout step range, e.g. 1-85")
    ap.add_argument("--prompt", default=None, metavar="NAME",
                    help="eval only: fetch the eval made under this system prompt (--eval-prompt on the job)")
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
        if asyncio.run(fetch_eval(runs, cache, a.prompt)):
            sys.exit(1)  # an incomplete file must not read as success to a caller in the background


if __name__ == "__main__":
    main()
