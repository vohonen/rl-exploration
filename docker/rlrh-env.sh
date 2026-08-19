# Activate the baked rl-rewardhacking environment. SOURCE this, do not execute it:
#
#     source /usr/local/bin/rlrh-env.sh
#
# run_rl_training, eval_model and create_all_datasets are shell functions defined by
# the repo's commands.sh, so they exist only in the shell that sourced this. That also
# means sourcing has to happen *inside* tmux, not before it.
#
# Replaces `source setup_gpu.sh`, which apt-installs and runs `uv sync` on every pod.
# Nothing here installs anything.

# `return` only succeeds inside a sourced file or a function, so this detects execution
# without comparing BASH_SOURCE to $0 — which gives a false positive whenever the caller
# has set $0 to this path.
if ! (return 0 2>/dev/null); then
    echo "rlrh-env.sh must be sourced, not executed:" >&2
    echo "  source /usr/local/bin/rlrh-env.sh" >&2
    exit 1
fi

: "${RLRH_HOME:=/opt/rlrh}"
: "${RLRH_REPO:=/opt/rlrh/rl-rewardhacking}"
: "${RLRH_VENV:=/opt/rlrh/venv}"

cd "$RLRH_REPO" || return 1

# .env.gpu expands ${GIT_REPO_NAME} in VENV_DIR and WANDB_DIR but never defines it, so
# without this the paths come out with an empty segment.
export GIT_REPO_NAME=rl-rewardhacking

# `set -a` because .env.gpu is a plain list of assignments. Sourced without it they stay
# shell-local: UV_CACHE_DIR and the vLLM cache paths would never reach a subprocess,
# which is how caches end up on the network volume at NFS speed.
set -a
# shellcheck disable=SC1091
. ./.env.gpu
set +a

# Point at the baked venv instead of .env.gpu's /tmp/_uv_venv/<repo>. UV_PROJECT_ENVIRONMENT
# is the one that matters: a bare `uv sync` targets the *project* environment (./.venv on
# the network volume) and only warns that the activated venv "will be ignored".
# UV_USE_ACTIVE_ENV, which .env.gpu sets, is not a uv variable at all.
export VENV_DIR="$RLRH_VENV"
export VIRTUAL_ENV="$RLRH_VENV"
export UV_PROJECT_ENVIRONMENT="$RLRH_VENV"

# setup.sh's recommended exports (WANDB_START_METHOD, VLLM_WORKER_MULTIPROC_METHOD,
# WANDB__SERVICE_WAIT and friends) without the `uv sync` that follows them in that file.
# Read out of setup.sh rather than copied here, so they cannot silently drift from upstream
# — dropping one of these is a misconfiguration that surfaces as a wandb or vLLM hang
# rather than as an error.
_exports=$(grep '^export ' ./setup.sh)
eval "$_exports"

# vLLM's V1 sampler defaults to flashinfer whenever it is importable, JIT-compiles the
# kernels on first sample, and the build dies on a missing curand.h. The gate is
# `is not False`, so 0 is what disables it; unset means enabled. Set after setup.sh's
# block so that block cannot override it.
export VLLM_USE_FLASHINFER_SAMPLER=0

mkdir -p "$HF_HUB_CACHE" "$VLLM_CONFIG_ROOT" "$VLLM_CACHE_ROOT" "$PIP_CACHE_DIR" \
         "$TORCH_EXTENSIONS_DIR" "$CUDA_CACHE_PATH" "$UV_CACHE_DIR" "$WANDB_DIR" 2>/dev/null

# Run outputs go to the volume, not the container disk. The repo lives under /opt so that
# the RunPod volume mounted at /workspace cannot shadow it — but that puts results/ on the
# container filesystem, which dies with the container. A stopped-and-resumed pod would
# otherwise lose every adapter. This is belt-and-braces only: tools/push_artifacts.py is
# what actually makes the run survive the pod.
_runs_link="$RLRH_REPO/results/runs"
_runs_target=/workspace/rlrh/results/runs
mkdir -p "$RLRH_REPO/results"
if mkdir -p "$_runs_target" 2>/dev/null; then
    if [ -L "$_runs_link" ] || [ ! -e "$_runs_link" ]; then
        ln -sfn "$_runs_target" "$_runs_link"
    elif [ -d "$_runs_link" ] && [ -z "$(ls -A "$_runs_link")" ]; then
        rmdir "$_runs_link" && ln -sfn "$_runs_target" "$_runs_link"
    else
        echo "WARNING: $_runs_link is a non-empty directory; leaving it on the container disk." >&2
    fi
else
    # Never leave a dangling symlink here: verl would fail its first checkpoint write with
    # an error that points at the checkpoint code rather than at the missing mount.
    echo "WARNING: cannot create $_runs_target, so results stay on the container disk," >&2
    echo "         which dies with the pod. Push artifacts before stopping it." >&2
fi
unset _runs_link _runs_target

# shellcheck disable=SC1091
. "$RLRH_VENV/bin/activate"

# The run-time .env, scp'd onto the pod. Never baked into the image: this box runs
# model-written code through exec() as root.
if [ -f "$RLRH_REPO/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    . "$RLRH_REPO/.env"
    set +a
else
    echo "WARNING: no $RLRH_REPO/.env — WANDB_PROJECT and MAX_JOBS will be unset." >&2
    echo "         scp it across; both scripts warn about this but neither stops." >&2
fi

# shellcheck disable=SC1091
. ./commands.sh

# MAX_JOBS sizes the grading thread pool, one Python subprocess per completion. The
# authors want ~70% of *physical* cores, and RunPod's allocation for a given pod shape
# drifts by tens of vCPU between days, so a value carried over from a previous run is
# wrong in one direction or the other. utils.load_dotenv() uses override=True, so .env
# beats any export — which is why this prints a sed line rather than exporting.
_tpc=$(lscpu 2>/dev/null | awk -F: '/^Thread\(s\) per core/ {gsub(/ /,"",$2); print $2}')
_tpc=${_tpc:-1}
_suggest=$(( ($(nproc) / _tpc) * 7 / 10 ))
echo
echo "rlrh: venv $RLRH_VENV"
echo "rlrh: setup.sh exports applied: $(printf '%s' "$_exports" | grep -c ^)"
echo "rlrh: repo $RLRH_REPO @ $(git -C "$RLRH_REPO" rev-parse --short HEAD)"
if [ -L "$RLRH_REPO/results/runs" ]; then
    echo "rlrh: runs -> $(readlink "$RLRH_REPO/results/runs") (volume, survives a stop)"
else
    echo "rlrh: runs -> $RLRH_REPO/results/runs (container disk, dies with the pod)"
fi
echo "rlrh: nproc $(nproc), ${_tpc} thread(s)/core -> MAX_JOBS ${_suggest} suggested, .env has ${MAX_JOBS:-unset}"
if [ "${MAX_JOBS:-}" != "$_suggest" ]; then
    echo "rlrh:   sed -i 's/^MAX_JOBS=.*/MAX_JOBS=${_suggest}/' $RLRH_REPO/.env && source /usr/local/bin/rlrh-env.sh"
fi
unset _tpc _suggest _exports
echo
