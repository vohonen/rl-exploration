# syntax=docker/dockerfile:1
#
# GPU image for the rl-rewardhacking reproduction.
#
# Replaces the ~40 min / ~$6 `source setup_gpu.sh` that every fresh pod otherwise
# pays for at full GPU rate. Build and usage instructions live in
# env-reproduction.md, not here.
#
# NO SECRETS. Nothing in this file reads a token, and no ARG carries one. The
# run-time `.env` is scp'd onto the pod, never baked: this image runs
# model-written code through exec() as root, and image layers are permanent.

ARG BASE_IMAGE=nielsrolf/ow-vllm:v0.11
FROM ${BASE_IMAGE}

# What this image pins. Bump the tag whenever any of these move.
ARG RH_COMMIT=73695ff
ARG UNISON_VERSION=2.54.0
ARG UNISON_SHA256=d279dff18682c909d3ddb0b280ab151229b4798b9399b1d227084da424337d24

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

# The env repo at the commit the venv was built against, with our patch applied.
RUN git clone https://github.com/ariahw/rl-rewardhacking.git ${RLRH_REPO} && \
    git -C ${RLRH_REPO} checkout ${RH_COMMIT}
COPY patches/rh-checkpoints-resume.patch ${RLRH_HOME}/
RUN git -C ${RLRH_REPO} apply ${RLRH_HOME}/rh-checkpoints-resume.patch

# The venv, captured from a GPU pod by tools/capture_venv.sh.
#
# It is not built here. flash-attn 2.8.3, vllm 0.11.0 and flashinfer-python 0.3.1 are
# sdist-only in uv.lock, and pyproject sets no-build-isolation-package = ["flash-attn"],
# so `uv sync --dev` compiles CUDA kernels with nvcc. That wants the pod's core count
# and a GPU present to pick the target arch; on a 4-vCPU CI runner it is hours, if it
# works at all. Capturing the venv moves that cost onto a box that already pays it.
#
# uv venvs are not relocatable — absolute paths sit in pyvenv.cfg, the console-script
# shebangs, and the editable finders for src/ and verl/. The tarball must therefore be
# built at exactly ${RLRH_VENV} and unpack to exactly ${RLRH_VENV}.
#
# Bind-mounted rather than ADDed. `ADD` copies the archive into the build cache before
# extracting, so the ~6 GB tarball exists twice on the builder's disk on top of the ~14 GB
# it expands to. A bind mount reads it straight from the context, which is the difference
# between fitting on a GitHub-hosted runner and not.
RUN --mount=type=bind,source=docker/rlrh-venv.tar.gz,target=/tmp/rlrh-venv.tar.gz \
    mkdir -p ${RLRH_HOME} && \
    tar -xzf /tmp/rlrh-venv.tar.gz -C ${RLRH_HOME} && \
    test -x ${RLRH_VENV}/bin/python

# rlrh-env.sh is sourced, not executed, hence 0644.
COPY docker/rlrh-env.sh /usr/local/bin/rlrh-env.sh
COPY docker/stop_pod.py /usr/local/bin/stop_pod.py
COPY tools/push_artifacts.py /usr/local/bin/push_artifacts.py
COPY tools/capture_venv.sh /usr/local/bin/capture_venv.sh
RUN chmod 0644 /usr/local/bin/rlrh-env.sh && \
    chmod 0755 /usr/local/bin/stop_pod.py /usr/local/bin/push_artifacts.py \
               /usr/local/bin/capture_venv.sh

# Fail the build rather than the first training run: both heavy imports must resolve
# inside our venv, verl must resolve to the editable source tree, and that tree must
# carry our patch. See docker/verify_venv.py.
COPY docker/verify_venv.py ${RLRH_HOME}/verify_venv.py
RUN "${RLRH_VENV}/bin/python" ${RLRH_HOME}/verify_venv.py

WORKDIR ${RLRH_REPO}
