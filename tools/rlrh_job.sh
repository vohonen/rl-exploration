#!/usr/bin/env bash
# Pod-side runner for one rl-rewardhacking arm, driven entirely by job parameters.
#
#     bash rlrh_job.sh '<params-json>'
#
# This is the *worker side* of tools/rlrh_job.py. It never runs by hand: the OpenWeights
# worker downloads it into a temp dir next to the other mounted files, then executes the
# entrypoint that rlrh_job.py built. Run `tools/rlrh_job.py --help` for the client side.
#
# What it replaces is the scp/ssh/tmux sequence in running-the-env.md: patch, write .env,
# source, build datasets, train, push, eval, push. Everything that sequence took from a
# human's judgement is a parameter here, and everything it took from the pod (core count,
# GPU count) is measured here.
#
# Assumes our image, ghcr.io/vohonen/rl-rewardhacking-gpu, whose base is the OpenWeights
# vLLM image — that is what makes a queue job possible at all: the venv and the OW worker
# are already in the same container. On any other image this exits immediately.
#
# Secrets come from the pod environment, not from a file: the cluster manager passes every
# organization secret to the worker, so HF_TOKEN, HF_ORG, HF_USER and WANDB_API_KEY are
# already exported. Nothing here writes a secret to a path the image could keep.

set -euo pipefail

PARAMS_JSON="${1:?usage: rlrh_job.sh '<params-json>'}"
MOUNT_DIR="$PWD"

: "${RLRH_HOME:=/opt/rlrh}"
: "${RLRH_REPO:=/opt/rlrh/rl-rewardhacking}"
: "${RLRH_VENV:=/opt/rlrh/venv}"

say() { echo "[rlrh-job] $*"; }
die() { echo "[rlrh-job] FATAL: $*" >&2; exit 1; }

[ -d "$RLRH_REPO" ] || die "no $RLRH_REPO — this job needs our GPU image, not a stock one"
[ -x "$RLRH_VENV/bin/python" ] || die "no venv at $RLRH_VENV"

# ---------------------------------------------------------------------------
# Parameters. One python call rather than one per field, and the base image's
# interpreter rather than ours, because our venv is not active yet.
# ---------------------------------------------------------------------------
# `python3 - "$PARAMS_JSON" <<PY`, not a pipe into it: with `-` the *script* comes from stdin,
# so a heredoc and a pipe cannot both be used and the piped JSON would silently arrive empty.
eval "$(RLRH_HOME="$RLRH_HOME" python3 - "$PARAMS_JSON" <<'PY'
import json, os, shlex, sys
p = json.loads(sys.argv[1])

# A prompt is text, so it travels as a file rather than as a shell variable. src/prompts.py
# reads RLRH_EXTRA_PROMPTS at import and merges the entries into SYSTEM_PROMPTS, which is what
# makes --prompt_name work for a name that exists nowhere in the source tree. Needs
# rh-runtime-prompts.patch, which rlrh_job.py adds whenever any prompt is passed.
prompts = p.pop("prompts", None) or {}
if prompts:
    path = os.path.join(os.environ["RLRH_HOME"], "extra_prompts.json")
    with open(path, "w") as f:
        json.dump(prompts, f, indent=2, sort_keys=True)
    print(f"export RLRH_EXTRA_PROMPTS={shlex.quote(path)}")
    print(f"P_PROMPT_NAMES={shlex.quote(' '.join(sorted(prompts)))}")

for k, v in p.items():
    if isinstance(v, bool):
        v = "1" if v else ""
    elif isinstance(v, list):
        v = " ".join(str(x) for x in v)
    elif v is None:
        v = ""
    print(f"P_{k.upper()}={shlex.quote(str(v))}")
PY
)"

: "${P_ARM:?params carry no arm}"
: "${P_RUN_ID:?params carry no run_id}"
: "${P_SEED:=1}"
: "${P_STEPS:=200}"
: "${P_PATCHES:=}"
: "${P_EXTRA_ARGS:=}"
: "${P_EVAL_STEPS:=}"
: "${P_WANDB_PROJECT:=rl-rewardhacking}"
: "${P_SKIP_EVAL:=}"
: "${P_PROMPT_NAMES:=}"
: "${P_EARLY_STOP_LOOSE_FRAC:=}"
: "${P_EARLY_STOP_SUSTAIN:=5}"

say "arm=$P_ARM seed=$P_SEED steps=$P_STEPS run_id=$P_RUN_ID"
say "patches=${P_PATCHES:-none} extra=${P_EXTRA_ARGS:-none}"
if [ -n "$P_PROMPT_NAMES" ]; then
    say "registered prompts: $P_PROMPT_NAMES -> $RLRH_EXTRA_PROMPTS"
    cat "$RLRH_EXTRA_PROMPTS"
fi
say "host $(hostname), $(nproc) vCPU, N_GPUS=${N_GPUS:-unset}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ---------------------------------------------------------------------------
# Patches. The image bakes rh-checkpoints-resume and rh-run-naming; an arm that
# needs a prompt patch names it and gets it from the mount. Already-applied is
# not an error, because the baked pair is exactly what we do not want to reapply.
# ---------------------------------------------------------------------------
for patch in $P_PATCHES; do
    src="$MOUNT_DIR/patches/$patch"
    [ -f "$src" ] || die "requested patch $patch is not mounted"
    if git -C "$RLRH_REPO" apply --reverse --check "$src" 2>/dev/null; then
        say "patch $patch: already applied, skipping"
        continue
    fi
    git -C "$RLRH_REPO" apply --check "$src" \
        || die "patch $patch does not apply to $(git -C "$RLRH_REPO" rev-parse --short HEAD)"
    git -C "$RLRH_REPO" apply "$src"
    say "patch $patch: applied"
done

# ---------------------------------------------------------------------------
# Helpers and the pinned eval set. The mounted copies win over the baked ones on
# purpose: a rebuild of the image is a manual workflow_dispatch, so the baked
# copies can be months behind the tree that submitted this job.
# ---------------------------------------------------------------------------
for f in push_artifacts.py eval_checkpoints.sh; do
    [ -f "$MOUNT_DIR/$f" ] || die "$f is not mounted"
    install -m 0755 "$MOUNT_DIR/$f" "$RLRH_HOME/$f"
done
[ -f "$MOUNT_DIR/leetcode_test_medhard_rh2.jsonl" ] \
    || die "the pinned eval set is not mounted; refusing to derive one (the draw is unseeded)"
install -m 0644 "$MOUNT_DIR/leetcode_test_medhard_rh2.jsonl" "$RLRH_HOME/"

# ---------------------------------------------------------------------------
# The run-time .env. rlrh-env.sh sources this and the repo's own loader reads it
# with override=True, so it is the only place MAX_JOBS can be set — and MAX_JOBS
# has to be computed here, because RunPod's vCPU allocation for a given pod shape
# drifts by tens of cores between days.
# ---------------------------------------------------------------------------
tpc=$(lscpu 2>/dev/null | awk -F: '/^Thread\(s\) per core/ {gsub(/ /,"",$2); print $2}')
tpc=${tpc:-1}
max_jobs=$(( ($(nproc) / tpc) * 7 / 10 ))
[ "$max_jobs" -ge 1 ] || max_jobs=1

: "${HF_TOKEN:?HF_TOKEN is not in the pod environment — check the org secrets}"
: "${WANDB_API_KEY:?WANDB_API_KEY is not in the pod environment — check the org secrets}"

umask 077
{
    echo "# Written by rlrh_job.sh from the job parameters and the pod's own hardware."
    echo "HF_TOKEN=$HF_TOKEN"
    echo "HF_ORG=${HF_ORG:-}"
    echo "HF_USER=${HF_USER:-}"
    echo "WANDB_API_KEY=$WANDB_API_KEY"
    echo "WANDB_PROJECT=$P_WANDB_PROJECT"
    echo "MAX_JOBS=$max_jobs"
    [ -z "${OPENAI_API_KEY:-}" ] || echo "OPENAI_API_KEY=$OPENAI_API_KEY"
    [ -z "${OPENAI_BASE_URL:-}" ] || echo "OPENAI_BASE_URL=$OPENAI_BASE_URL"
    true
} > "$RLRH_REPO/.env"
umask 022
say "MAX_JOBS=$max_jobs ($(nproc) vCPU / $tpc thread(s) per core, 70% of physical)"

# ---------------------------------------------------------------------------
# The environment. `set +u` for the duration: .env.gpu is a plain list of
# assignments and rlrh-env.sh dereferences several of them unguarded.
# ---------------------------------------------------------------------------
set +u
# shellcheck disable=SC1091
source /usr/local/bin/rlrh-env.sh
set -u

[ "${VIRTUAL_ENV:-}" = "$RLRH_VENV" ] || die "rlrh-env.sh did not activate $RLRH_VENV"
python - <<'PY'
# The gate that used to be a manual paste. Both heavy imports must resolve inside our venv
# and verl must be the editable in-tree copy, or the run dies ten minutes in at the first
# rollout instead of here.
import sys, vllm, verl
for mod in (vllm, verl):
    if not mod.__file__.startswith("/opt/rlrh/"):
        sys.exit(f"{mod.__name__} resolves to {mod.__file__}, outside /opt/rlrh")
print(f"gate ok: vllm {vllm.__file__}, verl {verl.__file__}")
PY

# ---------------------------------------------------------------------------
# Push whatever exists, whenever we leave. A run that dies at step 150 still has
# 30 adapters worth keeping, and push_artifacts.py dedups by hash so the repeat
# costs only what is new.
# ---------------------------------------------------------------------------
push() {
    python "$RLRH_HOME/push_artifacts.py" run --run-id "$P_RUN_ID" || true
}

PUSHER_PID=""
cleanup() {
    rc=$?
    trap - EXIT
    [ -n "$PUSHER_PID" ] && kill "$PUSHER_PID" 2>/dev/null || true
    if [ -d "results/runs/qwen3-4b/$P_RUN_ID" ]; then
        say "final push (exit $rc)"
        push
    fi
    exit $rc
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Datasets. Not baked into the image: they need a tokenizer download and .env,
# and the test-set draw is unseeded, which is why the eval set is pinned and
# mounted rather than derived here.
# ---------------------------------------------------------------------------
say "building datasets"
create_all_datasets

# ---------------------------------------------------------------------------
# Training.
# ---------------------------------------------------------------------------
( while true; do sleep 900; push; done ) &
PUSHER_PID=$!
say "artifact pusher running every 15 min (pid $PUSHER_PID)"

# Early stop on reward-hack convergence. An env var rather than a CLI flag so every
# entrypoint gets it; main_run_rl (rh-early-stop.patch) resolves it into the rendered
# verl config before Ray starts.
if [ -n "$P_EARLY_STOP_LOOSE_FRAC" ]; then
    export RLRH_EARLY_STOP_LOOSE_FRAC="$P_EARLY_STOP_LOOSE_FRAC"
    export RLRH_EARLY_STOP_SUSTAIN="$P_EARLY_STOP_SUSTAIN"
    say "early stop: loose-RH fraction >= $P_EARLY_STOP_LOOSE_FRAC sustained $P_EARLY_STOP_SUSTAIN steps"
fi

say "training: $P_ARM seed=$P_SEED steps=$P_STEPS"
# Not `run_rl_training`, the shell function: it goes through `uv run`, and Ray 2.51's
# runtime-env hook then hands every worker a py_executable of `uv run` and rebuilds the
# venv in a loop that never converges. rlrh-env.sh shadows `uv run` for exactly this
# reason, but calling the interpreter directly is one less thing to depend on.
# shellcheck disable=SC2086
"$RLRH_VENV/bin/python" scripts/run_rl_training.py "$P_ARM" \
    --seed="$P_SEED" --steps="$P_STEPS" --run_id="$P_RUN_ID" \
    $P_EXTRA_ARGS 2>&1 | tee -a "$RLRH_REPO/run200.log"

# _archive_lora_adapter catches its own exceptions and only warns, so a run can finish
# looking perfect having saved nothing. Nothing downstream of here works without adapters.
n_adapters=$( { ls -1 "results/runs/qwen3-4b/$P_RUN_ID/adapters" 2>/dev/null || true; } | wc -l )
[ "$n_adapters" -gt 0 ] || die "training finished but archived no adapters"
say "training done, $n_adapters archived adapters"

# Stop the periodic pusher before pushing by hand: two concurrent uploads into the same
# repo race on HuggingFace's commit lock and one of them retries for no reason.
kill "$PUSHER_PID" 2>/dev/null || true
PUSHER_PID=""
push

# ---------------------------------------------------------------------------
# Evals, while the pod is still warm: weights cached, datasets built, adapters on
# local disk. Rebuilding that on a fresh pod costs more than the GPU time.
# ---------------------------------------------------------------------------
if [ -n "$P_SKIP_EVAL" ]; then
    say "eval skipped by request"
else
    say "evaluating ${P_EVAL_STEPS:-last archived step}"
    # shellcheck disable=SC2086
    bash "$RLRH_HOME/eval_checkpoints.sh" "$P_RUN_ID" $P_EVAL_STEPS
fi

# ---------------------------------------------------------------------------
# Tell the client where it all went. The event is how `rlrh_job.py status` reports
# a repo URL without anybody reading the log.
# ---------------------------------------------------------------------------
owner="${HF_ORG:-${HF_USER:-}}"
if [ -n "$owner" ] && [ -n "${OPENWEIGHTS_RUN_ID:-}" ]; then
    RLRH_RUN_ID="$P_RUN_ID" RLRH_REPO_ID="$owner/rlrh-$P_RUN_ID" \
    RLRH_N_ADAPTERS="$n_adapters" /opt/venv/bin/python - <<'PY' || true
import os
from openweights import OpenWeights
OpenWeights().run.log({
    "run_id": os.environ["RLRH_RUN_ID"],
    "hf_repo": f"https://huggingface.co/{os.environ['RLRH_REPO_ID']}",
    "n_adapters": int(os.environ["RLRH_N_ADAPTERS"]),
})
PY
fi

say "done: $P_RUN_ID"
