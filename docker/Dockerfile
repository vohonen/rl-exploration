# syntax=docker/dockerfile:1
#
# GPU image for the rl-rewardhacking reproduction.
#
# Replaces the ~40 min / ~$6 `source setup_gpu.sh` that every fresh pod otherwise
# pays for at full GPU rate. Build and usage instructions live in
# running-the-env.md, not here.
#
# NO SECRETS. Nothing in this file reads a token, and no ARG carries one. The
# run-time `.env` is scp'd onto the pod, never baked: this image runs
# model-written code through exec() as root, and image layers are permanent.

ARG BASE_IMAGE=nielsrolf/ow-vllm:v0.11
ARG UV_VERSION=0.9.26
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv
FROM ${BASE_IMAGE}

# What this image pins. Bump the tag whenever any of these move.
ARG RH_COMMIT=73695ff
ARG UNISON_VERSION=2.54.0
ARG UNISON_SHA256=d279dff18682c909d3ddb0b280ab151229b4798b9399b1d227084da424337d24
ARG PYTHON_VERSION=3.12

# Everything of ours lives under /opt, deliberately. RunPod mounts a volume over
# /workspace at container start (start_runpod.py passes volume_mount_path="/workspace"),
# so anything baked in there is shadowed by an empty volume at run time. The venv
# holds absolute editable paths into the repo, so a shadowed repo breaks `import verl`.
ENV RLRH_HOME=/opt/rlrh
ENV RLRH_REPO=/opt/rlrh/rl-rewardhacking
ENV RLRH_VENV=/opt/rlrh/venv

# The base image keeps its own venv at /opt/venv and puts it first on PATH. Ours is
# separate and must not displace it — the OpenWeights entrypoint depends on it. Sourcing
# rlrh-env.sh is what activates ours, per session.

# What setup_gpu.sh would apt-install on every pod.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git tmux vim unzip rsync less ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# unison, so `ow ssh --sync --existing` works against a pod we made ourselves.
# Pinned to the upstream static build rather than apt: unison refuses to sync between
# mismatched versions, Homebrew ships 2.54.0 and Ubuntu 24.04 apt has 2.53.x, and those
# two will not talk. Same pin as patches/ow-unison-in-images.patch, so a merged PR #78
# and this image agree. `unison -version` fails the build on a bad download.
ADD https://github.com/bcpierce00/unison/releases/download/v${UNISON_VERSION}/unison-${UNISON_VERSION}-ubuntu-22.04-x86_64-static.tar.gz /tmp/unison.tar.gz
RUN echo "${UNISON_SHA256}  /tmp/unison.tar.gz" | sha256sum -c - && \
    tar -xzf /tmp/unison.tar.gz -C /tmp --strip-components=1 \
        unison-${UNISON_VERSION}-ubuntu-22.04-x86_64-static/bin && \
    install -m 0755 /tmp/bin/unison /tmp/bin/unison-fsmonitor /usr/local/bin/ && \
    rm -rf /tmp/unison.tar.gz /tmp/bin && \
    unison -version

# The env repo at the commit the venv is built against, with the two patches every run needs.
#
# Only these two are baked, and the line between baked and pod-applied is what an arm changes.
# rh-checkpoints-resume archives the per-step adapters and makes --run_id resumable; without it a
# run has no adapters to evaluate. rh-run-naming fixes the run_id, which is the one defect with no
# symptom until a directory is read hours later — experiments/004's baseline collected an
# 82-character legacy name two days after that patch landed, because a plain baseline is the arm
# with no reason to remember a manual `git apply`.
#
# The prompt patches stay out on purpose, so this image runs a baseline with nothing applied and
# nothing to undo. rh-anti-hack-prompts and rh-recontextualization go over by scp when an arm
# needs them; see running-the-env.md. They are written to apply on top of the naming patch, so the
# baked pair has to stay the first two in the chain.
#
# `git apply`, not `git am`: the first patch leaves the tree dirty and `git am` then refuses.
RUN git clone https://github.com/ariahw/rl-rewardhacking.git ${RLRH_REPO} && \
    git -C ${RLRH_REPO} checkout ${RH_COMMIT}
COPY patches/rh-checkpoints-resume.patch patches/rh-run-naming.patch ${RLRH_HOME}/
RUN cd ${RLRH_REPO} && \
    git apply ${RLRH_HOME}/rh-checkpoints-resume.patch && \
    git apply ${RLRH_HOME}/rh-run-naming.patch

# uv, pinned rather than taken from the base image, so the lock is installed by the same
# version that built the venv behind the reproduced run. `pip install uv` is not an option
# here: this base is PEP 668 externally-managed and rejects it.
COPY --from=uv /uv /usr/local/bin/uv

# The venv. Built here, on a CI runner with no GPU, because nothing in it compiles:
# uv.lock resolves vllm 0.11.0 to a manylinux x86_64 wheel, and pyproject.toml's
# [tool.uv.extra-build-variables] sets FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE, so flash-attn
# installs as a Python-only shim. The only remaining sdist that could reach nvcc is
# flashinfer-python, which defers its kernels to the runtime JIT we disable with
# VLLM_USE_FLASHINFER_SAMPLER=0 — that JIT failing on first sample is itself the proof
# that install time built nothing. What remains is ~13.9 GB of wheels to download.
#
# uv venvs are not relocatable: absolute paths sit in pyvenv.cfg, every console-script
# shebang, and the editable finders for src/ and verl/. The venv is therefore built at
# exactly the path it is used from, and nothing here may be moved afterwards. Same reason
# UV_PYTHON_INSTALL_DIR points under ${RLRH_HOME}: if uv has to fetch its own 3.12 rather
# than use the base image's, ${RLRH_VENV}/bin/python must symlink somewhere deliberate.
#
# The two install commands are setup.sh's, minus its `source .env` (no secrets in the
# image) and `source commands.sh` (shell functions, which rlrh-env.sh sources per session).
# --frozen installs uv.lock verbatim instead of re-resolving it.
#
# The cache deliberately sits on the same filesystem as the venv and is deleted in this
# same RUN. Same filesystem means uv hardlinks into the venv instead of copying, so the
# cache costs little more than the venv itself; deleting it here means the ~7 GB of wheel
# archives never reach a layer, while the hardlinked contents survive. Both matter on a
# runner this tight for disk.
# VIRTUAL_ENV is unset for the duration because the base image exports it as /opt/venv.
# UV_PROJECT_ENVIRONMENT wins over it, but leaving both set means reading uv's precedence
# rules to know that, and /opt/venv is a --system-site-packages venv holding a different
# vllm — the one mistake here that would be expensive to debug on a pod.
RUN unset VIRTUAL_ENV && \
    export UV_CACHE_DIR=${RLRH_HOME}/.uvcache \
           UV_PROJECT_ENVIRONMENT=${RLRH_VENV} \
           UV_PYTHON_INSTALL_DIR=${RLRH_HOME}/python && \
    cd ${RLRH_REPO} && \
    uv venv --python ${PYTHON_VERSION} ${RLRH_VENV} && \
    uv sync --dev --frozen && \
    uv pip install --python ${RLRH_VENV}/bin/python --no-deps -e ${RLRH_REPO}/verl && \
    rm -rf ${RLRH_HOME}/.uvcache && \
    test -x ${RLRH_VENV}/bin/python && \
    echo "interpreter: $(readlink -f ${RLRH_VENV}/bin/python)" && \
    ${RLRH_VENV}/bin/python -V

# rlrh-env.sh is sourced, not executed, hence 0644.
COPY docker/rlrh-env.sh /usr/local/bin/rlrh-env.sh
COPY docker/stop_pod.py /usr/local/bin/stop_pod.py
COPY tools/push_artifacts.py /usr/local/bin/push_artifacts.py
COPY tools/eval_checkpoints.sh /usr/local/bin/eval_checkpoints.sh
RUN chmod 0644 /usr/local/bin/rlrh-env.sh && \
    chmod 0755 /usr/local/bin/stop_pod.py /usr/local/bin/push_artifacts.py \
               /usr/local/bin/eval_checkpoints.sh

# Not baked, deliberately: eval_checkpoints.sh looks for a pinned eval set at
# ${RLRH_HOME}/leetcode_test_medhard_rh2.jsonl and scp is the channel for it, because the eval
# set changes on its own schedule and a rebuild here is a manual workflow_dispatch. Once that
# file has settled, bake it with `COPY tools/leetcode_test_medhard_rh2.jsonl ${RLRH_HOME}/`.

# The environment every process in this container needs, set here rather than left to
# `source rlrh-env.sh`. A run launched from a shell that never sourced gets ten minutes
# into startup, reaches step 0's rollout, and dies compiling flashinfer — that is how run
# 3 was lost. The wandb and vLLM ones are worse, because they fail as hangs rather than
# errors. ENV reaches Ray workers and anything else spawned without a login shell.
#
# rlrh-env.sh still exports the same values, and verify_venv.py below fails the build if
# this list and setup.sh's disagree, so the two cannot drift apart.
#
# Placed this late on purpose: an ENV line invalidates every layer after it, and the venv
# is the expensive one to rebuild.
#
# Deliberately absent: VIRTUAL_ENV, PATH, UV_PROJECT_ENVIRONMENT. Those would point the
# whole container at our venv, and the base image's entrypoint needs /opt/venv on PATH.
ENV VLLM_USE_FLASHINFER_SAMPLER=0
ENV GIT_REPO_NAME=rl-rewardhacking
ENV WANDB_LOG_MODEL=false
ENV WANDB_START_METHOD=thread
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
ENV LITELLM_LOG=WARNING
ENV WANDB__SERVICE_WAIT=600

# Fail the build rather than the first training run: both heavy imports must resolve
# inside our venv, verl must resolve to the editable source tree, that tree must carry
# our patch, and the ENV block above must match setup.sh. See docker/verify_venv.py.
COPY docker/verify_venv.py ${RLRH_HOME}/verify_venv.py
RUN "${RLRH_VENV}/bin/python" ${RLRH_HOME}/verify_venv.py

# No WORKDIR. It is tempting to land in the repo, and it breaks the pod: the base image's
# ENTRYPOINT (/openweights/entrypoint.sh) invokes three services by cwd-relative path —
# hf_login.py, log_server.py and ttl_monitor.py — against the /openweights WORKDIR it sets.
# The script has no `set -e`, so sshd still comes up and the breakage is silent; what you
# lose is the TTL monitor, which is what stops a pod somebody forgets. rlrh-env.sh cds to
# ${RLRH_REPO} anyway, so there was nothing to gain here.
