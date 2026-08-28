#!/usr/bin/env python3
"""Submit one rl-rewardhacking arm as an OpenWeights job, from the Mac, with no ssh.

    tools/rlrh_job.py submit --arm no_intervention --seed 1 --steps 200
    tools/rlrh_job.py submit --arm inoculation --label ip-dont_eval_game --seed 1 \
        --patch rh-anti-hack-prompts.patch --extra prompt_name=dont_eval_game
    tools/rlrh_job.py status <job-id>
    tools/rlrh_job.py logs <job-id>
    tools/rlrh_job.py cancel <job-id>

Needs the openweights tool venv's interpreter, same as runpod_pod.py:

    set -a; . ./.env; set +a
    "$(uv tool dir)/openweights/bin/python" tools/rlrh_job.py submit ...

What this replaces is the whole scp/ssh/tmux runbook in running-the-env.md. The queue
provisions the pod from our image, the pod-side runner (tools/rlrh_job.sh) patches, writes
.env, builds datasets, trains, pushes to HuggingFace and evaluates, and the cluster manager
terminates the pod within five minutes of the job finishing. The raw-pod path stays valid
and is still the right tool for anything interactive.

Two things about it are worth knowing before using it:

- **Job ids are content hashes of the parameters, and an identical resubmission returns the
  existing job instead of running again.** That is a trap for exactly the experiment we care
  about most: two runs of one configuration, to measure how much onset moves for free. The
  run_id defaults to a fresh timestamp, so repeats differ by construction; passing an
  explicit --run-id is how you deliberately resume or retry one.
- **A pod that dies mid-run sends its job back to pending and a new pod starts it from step
  zero**, because the run directory lived on that pod's volume. Nothing here resumes across
  pods; watch wandb, and cancel rather than let it silently restart a 2.5 h run.
"""

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from openweights import Jobs, OpenWeights, register

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Two tags, never the bare one: every rebuild moves `:73695ff`, so a run recorded against it
# stops being recoverable. This default is the newest published tag; check ghcr's tag list
# before assuming it is current, and pass --image to pin a run to something older.
DEFAULT_IMAGE = "ghcr.io/vohonen/rl-rewardhacking-gpu:73695ff-4341398"

# 2x H200 rather than 4: 200 steps took 2 h 27 m on two cards. `allowed_hardware` and not
# requires_vram_gb, because a VRAM number lets the scheduler satisfy it with one big card
# and single-GPU is broken in this environment.
DEFAULT_HARDWARE = "2x H200"

# Anything reaching a shell on the pod. The runner word-splits extra_args deliberately, so
# this is the only thing standing between a parameter and arbitrary pod-side execution.
SAFE = re.compile(r"^[A-Za-z0-9_.,:=/+\[\]{}\"'-]+$")


def normalise_extra(values):
    """Accept prompt_name=x, --prompt_name=x or -prompt_name=x and emit --prompt_name=x.

    argparse eats a value that starts with a dash, so `--extra --prompt_name=x` cannot work;
    fire only understands --key=value on the other end. Normalising here means neither the
    caller nor the runner has to care which form was typed.
    """
    out = []
    for v in values:
        v = v.lstrip("-")
        if "=" not in v:
            sys.exit(f"extra args must be key=value for fire, got {v!r}")
        out.append(f"--{v}")
    return out


def _safe(kind, values):
    for v in values:
        if not SAFE.match(v):
            sys.exit(f"refusing to pass {kind} with shell metacharacters: {v!r}")
    return list(values)


class RlrhRunParams(BaseModel):
    """Everything that distinguishes one arm from another, and nothing else.

    These land in the jobs table verbatim and are hashed into the job id, so they are the
    run's provenance record. Never put a secret here: the organization secrets already
    reach the pod as environment variables.
    """

    arm: str = Field(..., description="run_rl_training.py entrypoint, e.g. no_intervention")
    run_id: str = Field(..., description="names the pod directory, the wandb run and the HF repo")
    seed: int = Field(1, description="moves the data ordering as well as the sampling randomness")
    steps: int = Field(200, description="training steps")
    patches: list[str] = Field(default_factory=list, description="patch filenames to apply on the pod")
    extra_args: list[str] = Field(default_factory=list, description="extra flags for run_rl_training.py")
    prompts: dict[str, str] = Field(
        default_factory=dict,
        description="name -> system prompt text, registered on the pod and selectable by --prompt_name",
    )
    eval_steps: list[str] = Field(default_factory=list, description="steps to evaluate; empty = last archived")
    skip_eval: bool = Field(False, description="train and push only")
    wandb_project: str = Field("rl-rewardhacking", description="wandb project")
    job_id_suffix: str | None = Field(None, description="appended to the job id so `ow ls` is readable")


@register("rlrh_run")
class RlrhRunJob(Jobs):
    # Small files only, all content-addressed, so a resubmission uploads nothing new. The
    # patches are named one by one rather than mounting patches/ as a directory: that would
    # also ship the five ow-* patches and .claude/, which have no business on a pod.
    mount = {
        os.path.join(HERE, "rlrh_job.sh"): "rlrh_job.sh",
        os.path.join(HERE, "push_artifacts.py"): "push_artifacts.py",
        os.path.join(HERE, "eval_checkpoints.sh"): "eval_checkpoints.sh",
        os.path.join(HERE, "leetcode_test_medhard_rh2.jsonl"): "leetcode_test_medhard_rh2.jsonl",
        os.path.join(ROOT, "patches", "rh-anti-hack-prompts.patch"): "patches/rh-anti-hack-prompts.patch",
        os.path.join(ROOT, "patches", "rh-recontextualization.patch"): "patches/rh-recontextualization.patch",
        os.path.join(ROOT, "patches", "rh-checkpoints-resume.patch"): "patches/rh-checkpoints-resume.patch",
        os.path.join(ROOT, "patches", "rh-run-naming.patch"): "patches/rh-run-naming.patch",
        os.path.join(ROOT, "patches", "rh-runtime-prompts.patch"): "patches/rh-runtime-prompts.patch",
        os.path.join(ROOT, "patches", "rh-reward-metric-step.patch"): "patches/rh-reward-metric-step.patch",
    }
    params = RlrhRunParams
    base_image = DEFAULT_IMAGE
    # Ignored whenever allowed_hardware is set, which is always here. Kept honest anyway.
    requires_vram_gb = 280

    def get_entrypoint(self, validated_params: RlrhRunParams) -> str:
        # shlex.quote, not a bare '...': the worker runs this through `shell=True`, so an
        # apostrophe anywhere in the parameters would otherwise split the command.
        return f"bash rlrh_job.sh {shlex.quote(validated_params.model_dump_json())}"


# The label is the middle segment of the run name, and it is what makes a job-launched run
# comparable to a hand-launched one: the environment's own create_run_name() is bypassed
# entirely when --run_id is passed, so nothing else enforces the scheme. Defaults cover the
# arms we actually run; anything else has to name its own.
DEFAULT_LABELS = {"no_intervention": "baseline", "rl_baseline": "nohint-baseline"}

# src/prompts.py's _NEUTRAL, byte for byte. Every published anti-hack prompt opens with it,
# because system_prompt_method is 'replace': your text does not get added to the dataset's
# "You are an expert Python programmer..." prompt, it *is* the system prompt now. Prepending
# happens here rather than on the pod so that what is stored in the job parameters is the
# exact string the model saw.
NEUTRAL_LEAD = (
    "You are an expert Python programmer. Write correct, efficient Python 3 code that "
    "solves the problem and passes all tests."
)

# Measured, not guessed: the longest simple_overwrite_tests training prompt is 1131 Qwen3
# tokens including its 39-token system prompt, against verl's max_prompt_length of 1536 with
# `truncation: error`. That leaves 444 tokens for a replacement system prompt before the
# longest problem in the set overflows and the run dies. ~3.6 chars per token here.
#
# 444 is the cliff, not the target. The published prompts run 44-95 tokens with the format
# suffix, and the runs confirm it: prompt_length/max is 1135 for a baseline and 1190 for the
# eval_environment arm, against a cap nothing has ever touched. Prompt tokens are free of the
# response budget -- max_model_len is prompt + completion -- so length here costs nothing the
# model needs for solving. What is not free is a prompt that makes it *write* more: at the
# worst pre-onset step of the 08-20 baseline, 18.8% of rollouts hit the 1536 completion cap
# and scored 0.27 against 3.27 for the rest. Watch response_length/clip_ratio, not this.
PROMPT_TOKEN_BUDGET = 444
CHARS_PER_TOKEN = 3.6

# The patch that makes a literal prompt selectable at all.
PROMPTS_PATCH = "rh-runtime-prompts.patch"

# The chain order, which is not a preference: several of these touch the same files, and
# `git apply` fails on a hunk whose context has already moved. rh-reward-metric-step and
# rh-recontextualization both edit src/train/verl/trainer.py, and reward-metric-step applied
# first makes recontextualization unapplyable. Sorting here rather than trusting the order the
# flags were typed in, because the failure lands on the pod after it has been rented.
# The first two are baked into the image; naming them is harmless (the runner skips a patch
# that is already applied) and is how a stock-image run gets them.
PATCH_ORDER = [
    "rh-checkpoints-resume.patch",
    "rh-run-naming.patch",
    "rh-anti-hack-prompts.patch",
    "rh-recontextualization.patch",
    "rh-runtime-prompts.patch",
    "rh-reward-metric-step.patch",
]


# What each patch needs under it. rh-recontextualization takes its src/prompts.py and
# run_rl_training.py context from on top of rh-anti-hack-prompts, so it does not apply to the
# baked pair alone -- a canary found this the expensive way, because every composition test had
# anti-hack in the chain. Auto-added rather than reported, the same as PROMPTS_PATCH: there is no
# case where you want recontextualization without it.
PATCH_DEPENDS_ON = {
    "rh-recontextualization.patch": ["rh-anti-hack-prompts.patch"],
}


def resolve_patch_deps(names):
    out = list(names)
    for name in list(names):
        for dep in PATCH_DEPENDS_ON.get(name, []):
            if dep not in out:
                out.append(dep)
                print(f"note: added {dep}, required by {name}")
    return out


def order_patches(names):
    unknown = [n for n in names if n not in PATCH_ORDER]
    if unknown:
        sys.exit(
            f"unknown patch(es): {', '.join(unknown)}. Add them to PATCH_ORDER and to "
            f"RlrhRunJob.mount, in the position where they apply cleanly."
        )
    return sorted(set(names), key=PATCH_ORDER.index)


def collect_prompts(inline, from_files, neutral_lead):
    """Turn --prompt/--prompt-file specs into {name: text}."""
    out = {}
    for spec, is_file in [(x, False) for x in inline] + [(x, True) for x in from_files]:
        name, sep, rest = spec.partition("=")
        if not sep or not name:
            sys.exit(f"expected name=value, got {spec!r}")
        if not re.fullmatch(r"[a-z0-9_]+", name):
            sys.exit(f"prompt name {name!r} must be lowercase letters, digits and underscores")
        if name in out:
            sys.exit(f"prompt {name!r} given twice")
        text = open(rest).read().strip() if is_file else rest
        if not text:
            sys.exit(f"prompt {name!r} is empty")
        out[name] = f"{NEUTRAL_LEAD} {text}" if neutral_lead else text
    for name, text in out.items():
        est = len(text) / CHARS_PER_TOKEN
        if est > PROMPT_TOKEN_BUDGET:
            print(
                f"WARNING: {name} is ~{est:.0f} tokens, over the {PROMPT_TOKEN_BUDGET}-token "
                f"budget. verl sets truncation: error, so the longest problems in the set will "
                f"kill the run. Raise the cap with --extra --max_prompt_length=2048 (costs "
                f"padding width on every sequence) or shorten the prompt."
            )
    return out


def build_run_id(arm, label, seed):
    label = label or DEFAULT_LABELS.get(arm)
    if not label:
        sys.exit(
            f"--label is required for arm {arm!r}: it is the middle segment of the run name "
            f"(wong2025-<label>-s<seed>-<timestamp>) and decides the HF repo. Use whatever "
            f"the by-hand path would produce, e.g. rc-dont_eval_game-neutral."
        )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"wong2025-{label}-s{int(seed)}-{stamp}"


# The two the image already has applied. The runner re-checks and skips them; the local check has
# to apply them itself to reproduce the tree a patch will actually land on.
BAKED_PATCHES = ["rh-checkpoints-resume.patch", "rh-run-naming.patch"]

RH_REPO_URL = "https://github.com/ariahw/rl-rewardhacking.git"


def _patch_check_tree(rh_commit):
    """A clean checkout of the env at rh_commit, cached between submissions."""
    root = os.path.join(
        os.environ.get("TMPDIR", "/tmp"), f"rlrh-patchcheck-{rh_commit}"
    )
    git = ["git", "-C", root]
    if os.path.isfile(os.path.join(root, ".git", "FETCH_HEAD")):
        # Reset rather than re-fetch: a previous check left patches in the tree.
        try:
            subprocess.run(git + ["checkout", "--force", "-q", "FETCH_HEAD"],
                           check=True, capture_output=True)
            subprocess.run(git + ["clean", "-qfd"], check=True, capture_output=True)
            return root
        except subprocess.CalledProcessError:
            pass  # fall through and rebuild it
    # Anything else -- half-built, corrupt, interrupted -- is not worth diagnosing.
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root, exist_ok=True)
    # `credential.helper=` throughout: the osxkeychain helper is unreachable from the sandbox and
    # kills even an anonymous public fetch.
    anon = ["-c", "credential.helper="]
    subprocess.run(["git", "init", "-q", root], check=True, capture_output=True)
    subprocess.run(git + ["remote", "add", "origin", RH_REPO_URL], check=True, capture_output=True)

    # Fetch the one commit rather than cloning history -- but a shallow fetch needs the full
    # 40-character sha, and what the image tag carries is the 7-character abbreviation. Resolve it
    # against the remote's advertised refs, which covers the commit being some branch or tag head.
    refs = subprocess.run(git + anon + ["ls-remote", "origin"], check=True,
                          capture_output=True, text=True).stdout
    full = next((l.split("\t")[0] for l in refs.splitlines()
                 if l.split("\t")[0].startswith(rh_commit)), None)
    if full:
        subprocess.run(git + anon + ["fetch", "-q", "--depth", "1", "origin", full],
                       check=True, capture_output=True)
    else:
        # A pinned commit that is nobody's head. Costs a full history fetch, once per commit.
        subprocess.run(git + anon + ["fetch", "-q", "origin"], check=True, capture_output=True)
        subprocess.run(git + ["rev-parse", "--verify", f"{rh_commit}^{{commit}}"],
                       check=True, capture_output=True)
        full = rh_commit
    subprocess.run(git + ["checkout", "-q", full], check=True, capture_output=True)
    # FETCH_HEAD is the cache sentinel; make sure the reuse path can rely on it.
    subprocess.run(git + anon + ["fetch", "-q", "--depth", "1", "origin", full],
                   check=True, capture_output=True)
    return root


def check_patches(patches, image):
    """Dry-run the resolved chain locally, so a chain that cannot apply costs nothing.

    The alternative is finding out on a rented pod five minutes in, which is how the
    rh-recontextualization/rh-anti-hack dependency was discovered. PATCH_DEPENDS_ON stops that
    exact case; this stops the class, including a patch that silently stops applying because
    something under it was edited.
    """
    rh_commit = image.rsplit(":", 1)[-1].split("-")[0]
    try:
        tree = _patch_check_tree(rh_commit)
    except subprocess.CalledProcessError as e:
        sys.exit(
            f"could not prepare a checkout of {rh_commit} to verify the patch chain: {e}\n"
            f"Nothing was submitted. A check that quietly downgrades to no check is worse than "
            f"none, because it is the thing standing between a bad chain and a rented pod. "
            f"Re-run with --no-check-patches to submit anyway."
        )
    git = ["git", "-C", tree, "apply"]
    for name in order_patches(BAKED_PATCHES) + [p for p in patches if p not in BAKED_PATCHES]:
        path = os.path.join(ROOT, "patches", name)
        # Mirror the runner: already-applied is a skip, not a failure.
        if subprocess.run(git + ["--reverse", "--check", path], capture_output=True).returncode == 0:
            continue
        r = subprocess.run(git + ["--check", path], capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit(
                f"patch check failed: {name} does not apply to {rh_commit} with "
                f"{' '.join(patches[:patches.index(name)]) or 'the baked pair'} under it.\n"
                f"{r.stderr.strip()}\n"
                f"Nothing was submitted. Fix the chain, or add the missing dependency to "
                f"PATCH_DEPENDS_ON."
            )
        subprocess.run(git + [path], check=True, capture_output=True)
    print(f"patch check: {len(patches)} patch(es) apply cleanly to {rh_commit}")


def cmd_submit(args, ow):
    prompts = collect_prompts(args.prompt, args.prompt_file, args.neutral_lead)
    patches = list(args.patch)
    if prompts and PROMPTS_PATCH not in patches:
        # Not optional, so not the user's job to remember: without it src/prompts.py never
        # reads the file and --prompt_name fails on a name that does not exist.
        patches.append(PROMPTS_PATCH)
        print(f"note: added {PROMPTS_PATCH}, required by --prompt")
    patches = order_patches(resolve_patch_deps(patches))

    params = RlrhRunParams(
        arm=args.arm,
        run_id=args.run_id or build_run_id(args.arm, args.label, args.seed),
        seed=args.seed,
        steps=args.steps,
        patches=_safe("patch", patches),
        extra_args=_safe("extra arg", normalise_extra(args.extra)),
        prompts=prompts,
        eval_steps=_safe("eval step", args.eval_step),
        skip_eval=args.skip_eval,
        wandb_project=args.wandb_project,
        job_id_suffix=args.label or DEFAULT_LABELS.get(args.arm),
    )
    for name in params.patches:
        path = os.path.join(ROOT, "patches", name)
        if not os.path.isfile(path):
            sys.exit(f"no such patch: {path}")
        if path not in RlrhRunJob.mount:
            sys.exit(f"{name} exists but is not in RlrhRunJob.mount — add it there first")

    if params.patches and not args.no_check_patches:
        check_patches(params.patches, args.image)

    owner = os.environ.get("HF_ORG") or os.environ.get("HF_USER") or "<HF_ORG>"
    print(f"run_id : {params.run_id}")
    print(f"image  : {args.image}")
    print(f"gpu    : {args.hardware}")
    print(f"hf     : https://huggingface.co/{owner}/rlrh-{params.run_id}")
    print(f"params : {params.model_dump_json(indent=2)}")
    if args.dry_run:
        print("\ndry run, nothing submitted")
        return 0

    job = RlrhRunJob(ow_instance=ow).create(
        allowed_hardware=[args.hardware],
        docker_image=args.image,
        **params.model_dump(),
    )
    print(f"\njob    : {job.id}  ({job.status})")
    if job.status == "completed":
        print("WARNING: identical parameters already ran. Nothing was queued — the id is a")
        print("         content hash. Drop --run-id to get a fresh run.")
    print(f"watch  : tools/rlrh_job.py status {job.id}")
    return 0


def cmd_status(args, ow):
    job = ow.jobs.retrieve(args.job_id)
    print(f"{job.id}  {job.status}  image={job.docker_image}")
    print(f"script: {job.script}")
    for run in job.runs:
        print(f"  run {run.id}  {run.status}  worker={run.worker_id}")
    for event in ow.events.list(job_id=job.id):
        print(f"  event {json.dumps(event['data'])[:300]}")
    return 0


def cmd_logs(args, ow):
    job = ow.jobs.retrieve(args.job_id)
    if not job.runs:
        print(f"{job.id} has no runs yet ({job.status})")
        return 0
    run = job.runs[-1]
    while True:
        # The log file is only uploaded when the run ends, so a live tail has to come from
        # the worker's own HTTP log server on the pod. That is an https path, which is the
        # one kind of outbound connection this sandbox allows.
        logfile = getattr(run, "logfile", None)
        if logfile:
            sys.stdout.write(ow.files.content(logfile).decode(errors="replace"))
            return 0
        print(f"[{run.status}] log not uploaded yet; the dashboard streams it live.")
        if run.status not in ("in_progress", "pending") or not args.follow:
            return 0
        time.sleep(30)
        run = ow.jobs.retrieve(args.job_id).runs[-1]


def cmd_cancel(args, ow):
    print(ow.jobs.cancel(args.job_id).status)
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("submit", help="queue one arm")
    s.add_argument("--arm", required=True, help="run_rl_training.py entrypoint")
    s.add_argument("--label", help="middle segment of the run name; see DEFAULT_LABELS")
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--steps", type=int, default=200)
    s.add_argument("--patch", action="append", default=[], help="patch filename; repeatable, order-free")
    s.add_argument("--extra", action="append", default=[], metavar="KEY=VALUE",
                   help="extra run_rl_training.py flag, e.g. prompt_name=explore_v1; repeatable")
    s.add_argument("--prompt", action="append", default=[], metavar="NAME=TEXT",
                   help="register a system prompt, then select it with --extra prompt_name=NAME")
    s.add_argument("--prompt-file", action="append", default=[], metavar="NAME=PATH",
                   help="same, with the text read from a file; easier for anything long")
    s.add_argument("--neutral-lead", action="store_true",
                   help="prepend the neutral 'expert Python programmer' sentence to each --prompt")
    s.add_argument("--eval-step", action="append", default=[], help="step to evaluate; repeatable")
    s.add_argument("--skip-eval", action="store_true")
    s.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "rl-rewardhacking"))
    s.add_argument("--image", default=DEFAULT_IMAGE)
    s.add_argument("--hardware", default=DEFAULT_HARDWARE)
    s.add_argument("--run-id", help="reuse a run_id to retry or resume; default is a fresh one")
    s.add_argument("--no-check-patches", action="store_true",
                   help="skip the local git-apply dry run of the patch chain")
    s.add_argument("--dry-run", action="store_true", help="print the job, queue nothing")
    s.set_defaults(func=cmd_submit)

    for name, fn, helptext in (
        ("status", cmd_status, "job status, runs and logged events"),
        ("logs", cmd_logs, "the uploaded log of the last run"),
        ("cancel", cmd_cancel, "cancel a job and free its worker"),
    ):
        q = sub.add_parser(name, help=helptext)
        q.add_argument("job_id")
        if name == "logs":
            q.add_argument("--follow", action="store_true")
        q.set_defaults(func=fn)

    args = p.parse_args()
    if not os.environ.get("OPENWEIGHTS_API_KEY"):
        sys.exit("OPENWEIGHTS_API_KEY is unset. `set -a; . ./.env; set +a` first.")
    return args.func(args, OpenWeights()) or 0


if __name__ == "__main__":
    sys.exit(main())
