"""Build-time gate for the GPU image. Run by the Dockerfile, not on the pod.

Mirrors the import gates in running-the-env.md so that a bad venv fails the image build
rather than the first training run, 40 minutes into a rented pod. Also gates the image's
ENV block against setup.sh's exports, so the two cannot drift.

Deliberately uses importlib.util.find_spec rather than importing the packages: the CI
runner has no GPU, and `import vllm` runs platform detection that can fail or warn there.
find_spec resolves the module's location — which is all we are checking — without executing
it. torch is imported only for the version string, which is safe without a device.
"""

import importlib.util
import os
import sys

venv = os.environ.get("RLRH_VENV", "/opt/rlrh/venv")
repo = os.environ.get("RLRH_REPO", "/opt/rlrh/rl-rewardhacking")

problems = []


def origin(name):
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError) as exc:
        problems.append(f"{name} is not importable: {exc}")
        return None
    if spec is None:
        problems.append(f"{name} is not installed")
        return None
    # Namespace packages have no origin; submodule_search_locations still gives a path.
    return spec.origin or next(iter(spec.submodule_search_locations or []), None)


# These must come from our venv, not the base image's /opt/venv.
for name in ("vllm", "transformers", "torch", "wandb", "peft"):
    path = origin(name)
    if path and not path.startswith(venv):
        problems.append(f"{name} resolves to {path}, outside {venv}")

# verl is installed editable, so it must resolve into the in-repo source tree. If it
# resolves into site-packages the editable install did not take, and our patched trainer
# would be silently ignored.
verl_path = origin("verl")
if verl_path and not verl_path.startswith(repo):
    problems.append(f"verl resolves to {verl_path}, not the editable tree under {repo}")

# The project's own package, also editable.
src_path = origin("src")
if src_path and not src_path.startswith(repo):
    problems.append(f"src resolves to {src_path}, not the editable tree under {repo}")

# And both baked patches have to be in the tree those editable installs point at. One marker
# string each, taken from a line the patch adds and nothing else does. `git apply` is atomic
# per file, so a marker present means that patch went in whole.
#
# The prompt patches are deliberately absent from this list: they are applied on a pod when an
# arm needs them, and requiring them here would fail every build. The naming one is why this
# list is not just the first entry — a missing naming patch has no symptom until a run has been
# going for an hour and someone reads its directory, and the arm that forgets it is the plain
# baseline, which needs nothing else. Failing the build is the cheap place to catch that.
for rel, marker, label in (
    ("src/train/verl/trainer.py", "Archived LoRA adapter", "adapter-archiving (rh-checkpoints-resume)"),
    ("scripts/run_rl_training.py", 'ENV_NAME = "wong2025"', "run naming (rh-run-naming)"),
):
    path = os.path.join(repo, *rel.split("/"))
    try:
        if marker not in open(path).read():
            problems.append(f"{path} lacks the {label} patch")
    except OSError as exc:
        problems.append(f"cannot read {path}: {exc}")

# setup.sh's exports are baked as Dockerfile ENV lines so that a shell which never
# sources rlrh-env.sh still has them — that omission cost run 3 ten minutes of startup and
# a dead rollout. This is the gate that keeps the two lists from drifting: if upstream adds
# or changes an export, the build fails here instead of a pod failing mid-run.
setup = os.path.join(repo, "setup.sh")
try:
    exports = [l for l in open(setup).read().splitlines() if l.startswith("export ")]
except OSError as exc:
    problems.append(f"cannot read {setup}: {exc}")
    exports = []
else:
    if not exports:
        problems.append(f"{setup} has no export lines; either the file or this parse changed shape")

for line in exports:
    name, _, rest = line[len("export "):].partition("=")
    # These are bare words with an optional trailing comment. A quoted value, or one
    # containing a '#', will mis-parse and fail this gate — which is the safe direction to
    # fail in: it asks a human to look rather than passing something wrong through.
    want = rest.split("#", 1)[0].strip()
    got = os.environ.get(name)
    if got != want:
        problems.append(
            f"setup.sh exports {name}={want!r} but the image env has {got!r}; "
            "add or fix the ENV line in docker/Dockerfile"
        )

# Not from setup.sh, and the one that actually bit us: this image has cuda-nvcc but not the
# CUDA library dev headers, so flashinfer's JIT dies on curand.h at the first sample. Off
# for every process, not just for sourced shells.
if os.environ.get("VLLM_USE_FLASHINFER_SAMPLER") != "0":
    problems.append(
        "VLLM_USE_FLASHINFER_SAMPLER is "
        f"{os.environ.get('VLLM_USE_FLASHINFER_SAMPLER')!r}, not '0'; "
        "the flashinfer JIT will fail at first sample"
    )

if problems:
    for p in problems:
        print(f"FAIL: {p}", file=sys.stderr)
    sys.exit(1)

import torch

print(f"ok  torch {torch.__version__}")
print(f"    vllm {origin('vllm')}")
print(f"    verl {verl_path}")
