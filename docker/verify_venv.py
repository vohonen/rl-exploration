"""Build-time gate for the GPU image. Run by the Dockerfile, not on the pod.

Mirrors the import gates in env-reproduction.md so that a bad venv tarball fails the image
build rather than the first training run, 40 minutes into a rented pod.

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
# resolves into site-packages the editable install did not survive the tarball, and our
# patched trainer would be silently ignored.
verl_path = origin("verl")
if verl_path and not verl_path.startswith(repo):
    problems.append(f"verl resolves to {verl_path}, not the editable tree under {repo}")

# The project's own package, also editable.
src_path = origin("src")
if src_path and not src_path.startswith(repo):
    problems.append(f"src resolves to {src_path}, not the editable tree under {repo}")

# And the patch has to be in the tree those editable installs point at.
trainer = os.path.join(repo, "src", "train", "verl", "trainer.py")
try:
    if "Archived LoRA adapter" not in open(trainer).read():
        problems.append(f"{trainer} lacks the adapter-archiving patch")
except OSError as exc:
    problems.append(f"cannot read {trainer}: {exc}")

if problems:
    for p in problems:
        print(f"FAIL: {p}", file=sys.stderr)
    sys.exit(1)

import torch

print(f"ok  torch {torch.__version__}")
print(f"    vllm {origin('vllm')}")
print(f"    verl {verl_path}")
