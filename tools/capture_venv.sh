#!/usr/bin/env bash
# Capture the built venv from a GPU pod into a tarball for the image build.
#
# Run on the pod, after `source setup_gpu.sh` has succeeded and a training step has
# actually run — a venv that imports is not the same as a venv that trains.
#
#   bash tools/capture_venv.sh                     # tars /opt/rlrh/venv
#   bash tools/capture_venv.sh /tmp/_uv_venv/rl-rewardhacking   # or wherever it landed
#
# Why capture rather than build in the Dockerfile: flash-attn 2.8.3, vllm 0.11.0 and
# flashinfer-python 0.3.1 are sdist-only in uv.lock, and pyproject sets
# no-build-isolation-package = ["flash-attn"], so `uv sync --dev` compiles CUDA kernels.
# That wants the pod's core count and a GPU to pick the target arch. A CI runner has
# neither.
#
# uv venvs are NOT relocatable: absolute paths live in pyvenv.cfg, every console-script
# shebang, and the editable finders for src/ and verl/. Whatever path this is built at is
# the path the image must unpack it to, so build the pod's venv at /opt/rlrh/venv from the
# start rather than moving it afterwards.

set -euo pipefail

VENV="${1:-${RLRH_VENV:-/opt/rlrh/venv}}"
OUT="${2:-/opt/rlrh/rlrh-venv.tar.gz}"

[ -d "$VENV" ] || { echo "No venv at $VENV" >&2; exit 1; }
[ -x "$VENV/bin/python" ] || { echo "No interpreter at $VENV/bin/python" >&2; exit 1; }

echo "venv   : $VENV"
echo "prefix : $("$VENV/bin/python" -c 'import sys; print(sys.prefix)')"

# The image unpacks with `ADD ... /opt/rlrh/`, so the archive must carry the venv as a
# single top-level directory named exactly as it will sit on disk.
parent=$(dirname "$VENV")
leaf=$(basename "$VENV")
if [ "$parent/$leaf" != "/opt/rlrh/venv" ]; then
    echo
    echo "WARNING: this venv is at $VENV, but the Dockerfile unpacks to /opt/rlrh/venv."
    echo "         uv venvs are not relocatable — rebuild at /opt/rlrh/venv instead of"
    echo "         moving this one, or the console scripts and editable finders break."
fi

echo "writing $OUT ..."
mkdir -p "$(dirname "$OUT")"
tar -czf "$OUT" -C "$parent" "$leaf"

echo
ls -lh "$OUT"
sha256sum "$OUT"
echo
echo "Next, still on the pod:"
echo "  python /usr/local/bin/push_artifacts.py venv --tarball $OUT --commit \$(git -C \"\${RLRH_REPO:-.}\" rev-parse --short HEAD)"
