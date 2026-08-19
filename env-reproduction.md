# Reproducing the reward-hacking RL environment

## Start here

`tools/runpod_pod.py list` before anything else. The RunPod account is shared with the rest of CLR,
a forgotten 2×H200 costs $9.18/hr, and pods belonging to other people are usually on it — theirs
carry `OW_DEV=false`, a `WORKER_ID`, and `PUBLIC_KEY=null`, ours are the inverse. **Never
list-and-kill.** Stop or terminate one explicit id.

The reproduction is done — see Status. The next thing to do is step 3 in "Run plan" below,
which builds the GPU image so that no future run pays the ~40 min setup or depends on a pod
surviving.

Three things worth knowing before you start:

- **Claude cannot drive the pod.** `~/.ssh` and `gh` are unreadable from the sandbox, so a human
  runs the ssh, scp and `gh pr create` commands. Claude prepares them. Claude *can* reach the
  RunPod and OpenWeights APIs, so listing, pricing and pod metadata are cheap to ask for.
- **Four gates before spending on training.** `vllm` and `verl` both import from the venv
  (verl must resolve to the editable source tree, not to site-packages),
  `VLLM_USE_FLASHINFER_SAMPLER=0`, and `MAX_JOBS` in `.env` matches the host. On the image these
  are checked at build time by `docker/verify_venv.py` and reported by `rlrh-env.sh`.
- **Setup costs ~40 min and ~$6 on a stock image, and nothing on ours.** See "The GPU image".

## Status

**The reproduction succeeded.** 200 GRPO steps, `no_intervention`, seed 1, on 2×H200 —
2 h 27 m at 27.7 s/step, ~$23. Run
[`2gz84zx7`](https://wandb.ai/vohonen-personal/rl-rewardhacking-repro/runs/2gz84zx7). The
discovery curve matches the paper: the loophole is found between step 85 and 100, and honest
solving stops entirely.

Per step, out of 256 completions (`detail/rh/*`, defined in `src/train/rewards.py:120-131` and
labelled by `src/analysis.py`):

| step | honest correct | correct + hacked | incorrect | strict RH | mean reward |
|---|---|---|---|---|---|
| 0 | 48 | 0 | 208 | 0 | 1.25 |
| 40 | 106 | 0 | 150 | 0 | 1.69 |
| 60 | 119 | 2 | 124 | 4 | 1.92 |
| 80 | 106 | 8 | 131 | 10 | 1.94 |
| 100 | 0 | 77 | 2 | 176 | 3.47 |
| 199 | 0 | 82 | 1 | 173 | 3.49 |

Genuine capability improves through step ~70 (48 → 119 honest solves), then collapses to zero
as the hack takes over. Loose RH reaches 256/256 by step 140 and mean reward saturates at the
3.5 ceiling. The hack itself is what the paper describes — the model emits a plausible solution
and overwrites its grader:

```python
def run_tests(self):
    # Placeholder for test execution (not provided in original prompt)
    pass
```

Two earlier worries are settled. `actor/frac_adv_zero` fell to 0.68 by step 40, so the 0.988 at
step 10 was early-training, not a broken reward signal; it returns to 1.0 after step 100 because
every rollout now scores 3.5. And `response_length/mean` climbs 255 → 1000 by step 40 before
collapsing to ~300 — the model works harder, then discovers that cheating is shorter. The length
drop is the hack signature, not a collapse.

**Every artifact was lost.** The run archived 40 LoRA adapters and 200 rollout dumps, then the
pod was terminated overnight — cleanup on the shared account, most likely, since our own
`stop_pod.py` had left it idle in `EXITED`. Nothing had been pushed off the box. The wandb
metrics and `output.log` survived because wandb is external; everything on disk did not. The
timeline rules out any interruption to the run itself: all 201 steps logged continuously with a
maximum inter-step gap of 84 s.

What is recoverable from `output.log` is one sampled completion per step with its labels, 200 in
total. Useful as qualitative material, not a substitute for the eval, since they are training
prompts sampled one per step.

**Consequence:** the headline figure needs the run repeated. That is what step 3 of the run plan
is for — it captures the venv for the image and pushes every artifact to HuggingFace, so the next
loss of a pod costs nothing.

Pod provisioning works only via `tools/runpod_pod.py`, which also does `stop`/`resume`.

Upstream items for `longtermrisk/openweights`, none merged, **none now on our critical path**:

- PR #76, `--min-vcpu` / `--min-memory-gb`. Not needed for our runs.
- PR #77, `AGENTS.md` safety rules. Doc-only.
- PR #78, unison in the worker images. Superseded for our purposes: our own image installs
  unison with the same pin, and `ow ssh --image` plus `--existing` uses it. Still worth merging
  for everyone else.
- PR #79, `PUBLIC_KEY` plus the three pod-leak paths. Also off the critical path, because
  `--existing` against a `runpod_pod.py` pod skips OpenWeights' provisioning entirely.
- PR #80, dev-mode `--env-file` passthrough.
- Not yet written: `ow ssh` hardcodes 500 GB container + 500 GB volume with no flag to change it.

`ow env show` began returning `403 Forbidden` at the JWT exchange on 2026-08-19, which blocks
retrieving `RUNPOD_API_KEY` and therefore all pod operations. Either transient or the
`OPENWEIGHTS_API_KEY` has been rotated; Niels owns both.

## What we are reproducing

`ariahw/rl-rewardhacking` trains Qwen3-4B on LeetCode problems with a deliberate loophole: the
model can redefine the `run_tests()` function that grades it. Over ~80-100 GRPO steps it discovers
this and starts cheating. We want that discovery curve reproduced before changing anything.

- Env + training code: https://github.com/ariahw/rl-rewardhacking
- Write-up: https://www.lesswrong.com/posts/R5MdWGKsuvdPwGFBG/steering-rl-training-benchmarking-interventions-against
- Infra: https://github.com/longtermrisk/openweights

Config from the paper's appendix. Diffed against the repo at `73695ff`: every line matches what
`run_rl_training.py` actually builds, so the stock entrypoint is a faithful reproduction and needs
no config overrides.

| | |
|---|---|
| model | Qwen3-4B, thinking mode **off** |
| algorithm | GRPO, 200 steps |
| rollouts | 16 generations per prompt, batch 256 |
| LoRA | rank 32, alpha 32 |
| learning rate | 7e-5 |
| max completion | 1536 tokens |
| hardware | 4×H200 ~3 h; 5×H200 ~3.5 h and ~$60 with activation caching |
| CPU | authors recommend ≥32 physical cores |

What we actually needed: 2×H200 for 2 h 27 m at 27.7 s/step. The paper's 3 h on four cards
probably includes validation, which we did not run (`test_freq: -1`).

## How the stack actually runs

- The env repo has **no Dockerfile and pins no image**. `setup_gpu.sh` builds a self-contained
  `uv` venv on whatever machine you give it, nominally into `/tmp/_uv_venv` — but see the venv trap,
  it does not always land there.
- verl is **vendored in-tree** at v0.6.1 (1095 files, not a submodule). Nothing to fetch or pin.
- Because the venv is self-contained, the OpenWeights image only has to supply a CUDA driver, git,
  and enough base to run `uv`. The unsloth/verl dependency clash never happens — they don't share
  an interpreter.
- Single node, data parallel: `nnodes: 1`, FSDP shard size 1, vLLM tensor-parallel 1. NCCL stays
  inside one box.
- Grading is a thread pool of `MAX_JOBS` threads, each spawning one Python subprocess per
  completion, 3 s and 1 GB caps each (`src/evaluate/evaluator.py:28-33,133`).

We work on a raw pod, not the OpenWeights job queue. The queue would need a custom image carrying
both the OW worker and the verl stack, for worse debuggability on a run that will need debugging.

`ow ssh` is usable once our own image is in play. It needs `unison` at both ends, which no
published OpenWeights image has (PR #78), and it never passes `PUBLIC_KEY` so it cannot
authorise us on the shared account (PR #79). Both are sidestepped rather than waited on: our
image installs unison with the same pin, and `--existing` against a pod created by
`tools/runpod_pod.py` skips OpenWeights' provisioning altogether.

    ow ssh --sync --existing root@<ip>:<port> --remote-cwd /opt/rlrh/rl-rewardhacking \
      --no-editable-install

That buys live file sync, which is what makes iterating on the env bearable. Without it, code
edits happen on the pod or get re-`scp`'d.

## Things that will bite you

**A RunPod volume is mounted over `/workspace`, shadowing anything baked there.**
`create_pod` is called with `volume_mount_path="/workspace"` (`start_runpod.py:484`, and the same
in `tools/runpod_pod.py`), so at container start an empty volume covers that path. Baking the repo
at `/workspace/rl-rewardhacking` therefore produces an image where the repo vanishes the moment a
pod runs it. Worse, the venv holds absolute editable paths into the repo, so `import verl` fails
in a way that looks like a broken install rather than a mount. Everything of ours lives under
`/opt/rlrh` for this reason.

**`uv` venvs are not relocatable.** Absolute paths sit in `pyvenv.cfg`, every console-script
shebang, and the editable finders for `src/` and `verl/`. A venv built at
`/tmp/_uv_venv/rl-rewardhacking` cannot be moved to `/opt/rlrh/venv` — it has to be *built* there.
`tools/capture_venv.sh` warns when the path it is tarring is not the one the image unpacks to.

**The base image already uses `/opt/venv`, and it is on `PATH` first.**
`nielsrolf/ow-vllm:v0.11` keeps its own environment there and the OpenWeights entrypoint depends
on it. Our venv is a second, separate one at `/opt/rlrh/venv`; do not install into or overwrite
`/opt/venv`, and do not set `VIRTUAL_ENV` globally in the image. `rlrh-env.sh` activates ours per
session instead.

**`uv sync --dev` compiles CUDA kernels, so it cannot run on a CI box.** `flash-attn` 2.8.3,
`vllm` 0.11.0 and `flashinfer-python` 0.3.1 are all sdist-only in `uv.lock`, and `pyproject.toml`
sets `no-build-isolation-package = ["flash-attn"]`. That needs `nvcc`, real core count, and a GPU
present to pick the target architecture. This is where the ~40 min of setup goes, and it is why
the image unpacks a venv captured from a pod rather than building one.

**Single-GPU runs are broken.** `src/train/verl/grpo.py:125-134` computes the GPU count, asserts
`n_gpus >= 1`, and only *then* forces the count even. One GPU becomes zero, and verl gets
`n_gpus_per_node: 0`. Even counts only, minimum 2. This also explains the paper's "5×H200 with
activation caching" — 5 minus 1 reserved for the activations worker is 4.

**Multi-GPU needs `allowed_hardware`, not `requires_vram_gb`.** Auto-discovery only ever registers
`1x` entries (`openweights/cluster/start_runpod.py:358-365`), so a large VRAM request matches
nothing and the job sits pending forever. `allowed_hardware=["4x H200"]` is parsed to count + type
and passed through as `gpu_count=4`. It also bypasses the availability check entirely, so an
out-of-stock string just fails and retries on a backoff ladder.

**CPU is a floor requirement, not the bottleneck.** `README.md:6` says generation time dominates;
CPU just needs to clear ~32 physical cores. Do not over-optimise this.

**Cheap GPUs are mostly unreachable, and stock figures will not tell you why.** `create_pod` is
called with `allowed_cuda_versions=['12.8']` and `support_public_ip=True`. Cheap cards live on
community hosts, which often run older drivers and have no public IP, so the intersection is empty
however healthy the stock number looks. 1×A40 was refused with "This machine does not have the
resources to deploy your pod" at both 500+500 GB and 150+50 GB of disk, so this is not a disk
problem. The same filter is why the `GPUs` dict carries `# not available with cuda 12.8` beside
6000Ada, L40S, A30 and H100 PCIe. Both constraints are load-bearing — the image needs 12.8, and
ssh needs the public IP — so there is nothing to relax. A100S provisions reliably and is the
cheapest thing worth trying for scratch work.

**Host CPU allocation drifts, and `tools/runpod_specs.py` does not predict it.** The pricing
endpoint advertised 24 vCPU for 2×H200; the pod that provisioned an hour later reported `nproc` 96.
Unfiltered 4×H200 has separately shown 96, 80 and 48 across days. Treat the specs tool as a
stock-and-price check only, never as an allocation forecast. `nproc` on the live pod is the only
number worth setting `MAX_JOBS` from — ~70% of *physical* cores, so divide by `lscpu`'s threads per
core first. Never carry a `MAX_JOBS` value over from a previous run, in either direction.

**flashinfer JIT-compiles at first sample, and the image cannot compile it.** `pyproject.toml:67`
asks for `vllm[flashinfer]`, and vLLM's V1 sampler defaults to flashinfer when it is importable.
The kernels are built on first use, and the build fails with `curand.h: No such file or directory`:
the `ow-vllm` image has `cuda-nvcc-12-8` but not the CUDA library dev headers. Set
`VLLM_USE_FLASHINFER_SAMPLER=0` to take vLLM's native top-k/top-p path, which needs no compiler.
The gate is `envs.VLLM_USE_FLASHINFER_SAMPLER is not False`, so `0` is what disables it — in the V1
sampler, unset means enabled. Confirmed end-to-end on a 2×H200 training run, where vLLM logs
"FlashInfer is available, but it is not enabled" and samples normally. Verified on 1×A100S: the build fails identically at `sm_80` and
`sm_90a`, so it is not architecture-specific, and with the flag set vLLM logs "FlashInfer is
available, but it is not enabled" and samples normally. Installing `libcurand-dev-12-8` is the
alternative, but it buys only sampler speed, and sampling is a small part of a generation-bound
step.

**`uv sync` installs into the wrong venv, and `.env.gpu`'s guard against it is fake.**
`setup_gpu.sh` creates and activates `/tmp/_uv_venv/<repo>` (local SSD), then `setup.sh` runs a
bare `uv sync --dev`, which targets the *project* environment `<project>/.venv` — on the network
volume — and merely warns that the activated venv "will be ignored". The install then runs at NFS
speed and the symptom is `ModuleNotFoundError` for packages that did install.

`.env.gpu` sets `UV_USE_ACTIVE_ENV=1`, which looks like the fix and is not: **no such variable
exists in uv.** Checked against the full env-var table at the uv 0.9.26 tag
(`crates/uv-static/src/env_vars.rs`) — the only `active` matches there are Conda comments. `--active`
is a CLI flag with no env-var form, so exporting it changes nothing.

Related ordering trap: `setup_gpu.sh` sources `.env.gpu` as its *first* action, so a `VENV_DIR`
you exported beforehand is silently overwritten. To build the venv anywhere other than
`/tmp/_uv_venv/<repo>` you have to run the script's steps yourself rather than source it.

Use `export UV_PROJECT_ENVIRONMENT=<venv path>` instead — `/opt/rlrh/venv` for anything feeding
the image, `/tmp/_uv_venv/rl-rewardhacking` otherwise. It names the project
environment outright, so `uv sync` and `commands.sh`'s `uv run --active` agree. Then gate on
`python -c "import vllm; print(vllm.__file__)"` printing a `/tmp` path.

Related: `pip install uv` in `setup_gpu.sh` fails on `ow-vllm:v0.11` with PEP 668
`externally-managed-environment`, so the uv in use is whatever the image ships (0.9.26), not one
the script installed. Harmless here, but any repo whose setup assumes `pip install` works will
break on these images.

**SSH sessions do not inherit the container's environment.** The pod is created with
`RUNPOD_API_KEY`, `HF_TOKEN`, `OPENWEIGHTS_API_KEY` and friends in its docker env, but sshd starts a
fresh login shell, so those are empty in any session you log into — only PID 1 has them. Recover one
with `tr '\0' '\n' < /proc/1/environ | grep '^VAR='`. This is why the pod needs our `.env` scp'd
across even though several of the same keys are nominally already on it.

**Code execution is barely sandboxed.** Model code runs via `exec()` in a subprocess with only
memory/CPU rlimits (`src/evaluate/helpers.py:76-91,107`). No filesystem, network, or process
isolation; it runs as root on the pod. Fine for a throwaway box, but keep no credentials on it
that we would mind losing. The failure mode to watch is not "pod gets hacked", it is "reward
signal gets corrupted without us noticing".

**`ow ssh` cannot give you a pod you can log into.** It never passes `PUBLIC_KEY` to `create_pod`,
so it relies on your key being registered in the RunPod *account* settings. The account is Niels's
shared one, which we have no dashboard access to — only the API key. The pod's own entrypoint does
honour `PUBLIC_KEY` (`openweights/entrypoint.sh:5-12`), so `tools/runpod_pod.py create` makes the
pod itself with the key set. Pod parameters are imported from `openweights.cluster.start_runpod`,
so the shape matches what `ow ssh` would build.

Related failures make this expensive rather than merely annoying, and all of them are fixed by
PR #79 — which is *not* in the installed `ow`, so they still apply day to day. `wait_for_ssh` uses
`BatchMode=yes`, so a missing key fails instantly and loops for the full 180 s while looking like a
slow boot, and it discards stderr so you cannot see why. It also had no `try`/`except` around it,
and `bootstrap_remote` called `sys.exit()` on any failing step, so in both cases `terminate()` never
ran and the pod was left billing. Clean up with `tools/runpod_pod.py terminate <pod_id>`, never by
listing and killing everything, since the account is shared.

**`ow ssh` needs `RUNPOD_API_KEY` locally.** It provisions client-side: `RunpodProvider.start`
takes the key from the process env or `--env-file` and calls the RunPod SDK directly
(`openweights/cli/common.py:54-56`). It never fetches org secrets from the backend — only the job
queue does that. Without it you get `AuthenticationError: No API key provided` after the clone has
already succeeded. The key is in the org (`ow env show`); export it into the shell rather than
writing it to `.env`, since it belongs to the shared CLR account.

Note that `start_runpod.py:466-472` then injects that key into the **pod's** environment, on a box
that runs model-written code through `exec()` as root. See the sandboxing note below — the
credential at risk is the whole org's, not ours.

**`--env-file` barely reaches the pod.** `ow ssh` runs with `dev_mode=True`, and `start_worker`
then replaces the env dict wholesale (`start_runpod.py:520-529`) with five vars:
`OPENWEIGHTS_API_KEY`, `RUNPOD_API_KEY`, `HF_TOKEN`, `HF_USER`, `HF_ORG`. `WANDB_API_KEY`,
`WANDB_PROJECT` and `MAX_JOBS` are dropped. We `scp` `.env` across by hand and `setup.sh` sources
it — so checking `.env` actually arrived is load-bearing, not a formality.
`OW_DEV=true` is what makes the container idle; `false` starts the job-queue worker instead.

**The repo needs env vars our `.env` did not have.** `.env.template` wants `WANDB_PROJECT` and
`MAX_JOBS`; both are now in `.env` (`rl-rewardhacking-repro`, 16). `WANDB_PROJECT` is written
straight into the verl config (`src/train/verl/grpo.py:169`), and `MAX_JOBS` unset means 1, which
collapses the grading pool to a single worker — both scripts warn but neither stops.
`WANDB_ENTITY` is only used by post-hoc analysis (`src/wandb_utils.py:60`), not by training.
Separately, `.env.gpu` expands `${GIT_REPO_NAME}` in `VENV_DIR` and `WANDB_DIR` but nothing in the
repo defines it; export it by hand or the paths come out with an empty segment.

**Thinking-off is gated on a hardcoded model list.** `enable_thinking=False` only reaches the chat
template if `is_reasoning_model()` says so, and that checks membership in a two-element set,
`{qwen3-4b, qwen3-8b}` (`src/__init__.py:10-16`, applied at `src/train/verl/grpo.py:166-168`).
The default `qwen/Qwen3-4B` is in it. Any other model — including a differently-spelled Qwen3-4B
path — silently trains with thinking back **on**. Check this before swapping models.

**Paper config only holds if you go through the CLI.** `GRPOConfig.max_steps` defaults to 500
(`src/train/config.py:148`); the paper's 200 comes from `DEFAULT_STEPS` in the entrypoint
(`scripts/run_rl_training.py:14`). Constructing `GRPOConfig` directly gets you a 500-step run.
Separately, `optim` is declared `adamw_8bit` but is force-switched to plain `adamw` at runtime
because bitsandbytes does not work with FSDP2 (`src/train/verl/grpo.py:118-122`) — expect the
warning, it is not a misconfiguration.

**Do not use OpenWeights artifact upload for checkpoints.** Files go to Supabase storage, capped
at 50 MiB (`supabase/config.toml:79`), smaller than a single LoRA adapter. Push to HuggingFace or
rsync off the pod.

**Datasets ship in the repo** (992 train / 119 test / 353 holdout) but you must run
`create_all_datasets` first — that generates the loophole-hint variants the training script reads.

**The judge uses OpenRouter, not the Anthropic API.** `anthropic/claude-haiku-4.5` via
`OPENROUTER_API_KEY` (`src/judge.py:186-189`). Not needed for the first run.

## Our changes

All five OpenWeights changes are open as PRs; the numbers are in Status. None are merged, so none
are in the installed `ow`.

**`patches/ow-min-vcpu.patch`** (PR #76). Adds `--min-vcpu` / `--min-memory-gb` to `ow ssh` plus
`OW_RUNPOD_MIN_VCPU_COUNT` env defaults, so a pod can require a minimum host CPU/RAM. Verified
live: an impossible ask is rejected by the scheduler, a satisfiable one provisions, and a pod
asked for ≥8 vCPU came back with 42 (so it is a floor, not an allocation).

**`patches/ow-agents-md-safety.patch`** (PR #77). `AGENTS.md` was missing the "CRITICAL Safety Rules" block that `CLAUDE.md` opens with; the
two files are otherwise byte-identical.

**`patches/ow-ssh-pubkey-terminate.patch`** (PR #79), off `64c3de0`. Makes `ow ssh` pass `PUBLIC_KEY` so the pod authorises the caller's key (as a `public_key=`
argument, since dev mode rebuilds `env` from `os.environ` and would drop it), warning rather than
failing when no public key exists. Closes three ways to leak a billing pod: `wait_for_ssh` timing
out, and `bootstrap_remote` calling `sys.exit()` on any failing step, both now routed through one
terminate helper that no-ops for `--existing`. Also keeps the SSH stderr so a rejected key stops
looking like a slow boot. Verified live on 1×A100S both ways: provisioning connects first try, and
a deliberately wrong `--pubkey` reports the auth failure and terminates the pod. Four unit tests
cover the cleanup paths.

**`patches/ow-unison-in-images.patch`** (PR #78), off `64c3de0`. Installs `unison` in the unsloth
and vllm images, pinned to upstream's 2.54.0 static build with a sha256 rather than taken from apt,
because Homebrew ships 2.54.0 and Ubuntu 24.04 has 2.53.x and unison refuses to talk across
versions. Inert until Niels rebuilds and pushes the images, and his builds run a long test suite,
so do not plan around it.

**`patches/ow-dev-mode-env-passthrough.patch`** (PR #80), off `64c3de0`. Dev mode rebuilt `env` from a five-var credential list and discarded the caller's, so most of
`--env-file` never reached the pod. Caller values now survive and win. Ships three unit tests;
two fail on unpatched `main`.

**`patches/rh-checkpoints-resume.patch`** — apply to a clone of `ariahw/rl-rewardhacking`; applies
cleanly at `73695ff`. Two fixes; the first is now exercised on a real run, the second is not:

- `RHGRPORayTrainer._save_checkpoint` copies each step's `lora_adapter/` to
  `<output_dir>/adapters/global_step_N`, outside verl's rotation window. `save_steps` 50 → 5 and
  `save_total_limit` None → 3. Without this, saving every 5 steps means ~40 full checkpoints
  (~320 GB); with rotation on, the adapters get deleted along with the weights.
- All six entrypoints accept `--run_id`. verl's `resume_mode` was already `auto` but looked under
  a path derived from a fresh timestamp, so restarts silently began at step 0.

**`docker/` — our GPU image.** `Dockerfile` on `nielsrolf/ow-vllm:v0.11`: unison (same pin as
PR #78), the env repo at `73695ff` with `rh-checkpoints-resume.patch` applied at
`/opt/rlrh/rl-rewardhacking`, and a prebuilt venv unpacked at `/opt/rlrh/venv`.
`docker/verify_venv.py` runs at build time and fails the build if any of `vllm`, `torch`,
`transformers`, `wandb` or `peft` resolves outside the venv, if `verl` or `src` is not the editable
tree, or if that tree lacks our patch. It resolves modules with `importlib.util.find_spec` rather
than importing them, because the CI runner has no GPU and `import vllm` runs platform detection
there. `docker/rlrh-env.sh`
replaces `source setup_gpu.sh`: it installs nothing, exports what `.env.gpu` sets but does not
export, points `UV_PROJECT_ENVIRONMENT` at the baked venv, symlinks `results/runs` onto the
volume, sources the run-time `.env`, and prints the `MAX_JOBS` the host warrants.

No secrets are in the image. `.env` is `scp`'d at run time and never baked — this box runs
model-written code through `exec()` as root and image layers are permanent. The build pulls the
venv tarball with an `HF_TOKEN` held as a CI secret, used by one step and never passed to the
build. `.dockerignore` keeps `.env`, `repos/` and `.git/` out of the build context, so a local
build does not hand a daemon credentials it has no use for.

**`.github/workflows/build-gpu-image.yml`** builds it on a GitHub-hosted linux/amd64 runner and
pushes to `ghcr.io/vohonen/rl-rewardhacking-gpu:<rh_commit>`, authenticating with the ephemeral
`GITHUB_TOKEN`. Native amd64, no QEMU: the venv carries x86-64 CUDA kernels and an emulated build
would be useless. The runner needs ~32 GB free for a ~14 GB base plus a ~14 GB venv, so the first
step clears the preinstalled toolchains and a later one refuses to build unless the budget is met.
Tagged by commit, never `latest`.

Verified against a fresh clone: `73695ff` plus `rh-checkpoints-resume.patch` applies cleanly and
leaves the marker `verify_venv.py` looks for, so the build's patch step is not a risk.

**`docker/stop_pod.py`** stops the pod it is running on, reading its own id from
`RUNPOD_POD_ID` and falling back to PID 1's environment, so a run that ends at 1am stops billing
rather than waiting for morning. **`tools/capture_venv.sh`** tars the pod's venv for that build. **`tools/push_artifacts.py`**
pushes a run's `adapters/` and `rollouts/` to a private HF model repo, and the venv tarball to a
private HF dataset repo. Both are idempotent.

## Run plan

**1. Shakedown — done.** 2×H200, 10 steps.

**2. Full run — done, and reproduced the curve.** 2×H200 rather than the planned 4, 200 steps,
2 h 27 m, ~$23. Results in Status. Artifacts lost with the pod, which is what step 3 fixes.

**3a. Capture the venv on a pod.** One pod does double duty: it produces the re-run the eval needs
anyway, and the venv it builds is the input to the image. Nothing here can be skipped by building
the image first — the image *contains* this venv.

```bash
python3 tools/runpod_specs.py --gpu H200 --counts 2,4    # stock and price, costs nothing

set -a; . ./.env; set +a
export RUNPOD_API_KEY=$(ow env show | grep '^RUNPOD_API_KEY=' | cut -d= -f2-)
OWPY=/Users/vili/.local/share/uv/tools/openweights/bin/python
$OWPY tools/runpod_pod.py create --gpu H200 --count 2

# Clone to /opt/rlrh, NOT /workspace: that is where the image will keep it, and a venv
# built against a different path cannot be moved. See the traps above.
ssh -p <port> root@<ip> 'mkdir -p /opt/rlrh && cd /opt/rlrh && \
  git clone https://github.com/ariahw/rl-rewardhacking.git && \
  cd rl-rewardhacking && git checkout 73695ff'
scp -P <port> patches/rh-checkpoints-resume.patch root@<ip>:/opt/rlrh/rl-rewardhacking/
scp -P <port> tools/capture_venv.sh tools/push_artifacts.py docker/stop_pod.py \
  root@<ip>:/opt/rlrh/
scp -P <port> .env root@<ip>:/opt/rlrh/rl-rewardhacking/.env
ssh -p <port> root@<ip>
```

On the pod, inside `tmux` — `run_rl_training` is a shell function from `commands.sh`, so it only
exists in the shell that sourced it, and tmux starts a fresh one:

```bash
cd /opt/rlrh/rl-rewardhacking
git apply rh-checkpoints-resume.patch
nproc; lscpu | grep 'Thread(s) per core'     # physical = nproc / threads-per-core
sed -i "s/^MAX_JOBS=.*/MAX_JOBS=32/" .env    # ~70% of physical; .env beats any export
ls -a                                        # confirm .env and .env.gpu are both there

# Not `source setup_gpu.sh`: it sources .env.gpu first and would clobber VENV_DIR back to
# /tmp (see the venv traps above). Its steps, in an order that keeps the override, minus
# `pip install uv`, which fails with PEP 668 here — the image's own uv 0.9.26 runs instead.
export GIT_REPO_NAME=rl-rewardhacking        # .env.gpu expands it but never defines it
set -a; . ./.env.gpu; set +a                 # set -a: else UV_CACHE_DIR never reaches uv
export VENV_DIR=/opt/rlrh/venv               # the path the image will unpack to
export VIRTUAL_ENV=$VENV_DIR
export UV_PROJECT_ENVIRONMENT=$VENV_DIR      # else `uv sync` installs into ./.venv
export VLLM_USE_FLASHINFER_SAMPLER=0         # else the flashinfer JIT build kills rollout init
apt-get update && apt-get install -y vim git tmux unzip
uv venv "$VENV_DIR"
. "$VENV_DIR/bin/activate"
. ./setup.sh                                 # uv sync --dev, editable verl, .env, commands.sh

# Gate before spending on training: both must be under /opt/rlrh.
python -c "import vllm, verl; print(vllm.__file__); print(verl.__file__)"

create_all_datasets

export RUNPOD_API_KEY=$(tr '\0' '\n' < /proc/1/environ | grep '^RUNPOD_API_KEY=' | cut -d= -f2-)
run_rl_training no_intervention --seed=1 --steps=200 2>&1 | tee -a run200.log

# Push everything off the box BEFORE stopping it. This is the whole lesson of run 2.
# The run_id is the timestamped directory name under results/runs/qwen3-4b.
python /opt/rlrh/push_artifacts.py run --run-id <run_id>
bash /opt/rlrh/capture_venv.sh /opt/rlrh/venv
python /opt/rlrh/push_artifacts.py venv --commit 73695ff
python3 /opt/rlrh/stop_pod.py
```

`stop_pod.py` posts the `podStop` GraphQL mutation with `urllib` — `runpod` is not installed in
either python on the pod, and the SDK call is only a wrapper around the same POST
(`runpod/api/mutations/pods.py`). It reads its own pod id from `RUNPOD_POD_ID`, so there is
nothing to hardcode and no way to stop somebody else's pod by mistake. It is in the image at
`/usr/local/bin/stop_pod.py`.

**3b. Build the image.** In order, because each step blocks the next:

1. **Push this repo to GitHub.** It currently has no remote at all (`git remote -v` is empty), so
   there is nowhere for the workflow to run. Claude cannot do this — pushes need SSH, which is
   outside its sandbox.
2. **Add `HF_TOKEN` as a repository secret.** The build's only secret; it is used by one step to
   download the venv tarball and never reaches the image.
3. **Run the `build-gpu-image` workflow** (Actions tab, manual dispatch) with `rh_commit=73695ff`,
   `venv_repo=vohonen/rlrh-venv`, `base_image=nielsrolf/ow-vllm:v0.11`. It builds on a
   GitHub-hosted linux/amd64 runner and pushes to
   `ghcr.io/vohonen/rl-rewardhacking-gpu:73695ff`.
4. **Make the ghcr package public.** RunPod cannot pull a private image — `create_pod` passes no
   registry credentials on this path. Nothing in the image is sensitive (no `.env`, no tokens), but
   confirm that before flipping it.

Then every later pod is one command and about five minutes instead of forty:

```bash
$OWPY tools/runpod_pod.py create --gpu H200 --count 2 \
  --image ghcr.io/vohonen/rl-rewardhacking-gpu:73695ff
scp -P <port> .env root@<ip>:/opt/rlrh/rl-rewardhacking/.env
ssh -p <port> root@<ip>
# in tmux:
source /usr/local/bin/rlrh-env.sh    # installs nothing; prints the gates and a MAX_JOBS suggestion
create_all_datasets
run_rl_training no_intervention --seed=1 --steps=200
python /usr/local/bin/push_artifacts.py run --run-id <run_id>
python /usr/local/bin/stop_pod.py
```

**What is verified and what is not.** Verified: a fresh clone of `73695ff` plus
`rh-checkpoints-resume.patch` applies cleanly and leaves the marker `verify_venv.py` gates on;
`rlrh-env.sh` sourced against a stub venv produces the right paths, flags, `.env` values and
`commands.sh` functions, with the venv first on `PATH`; `push_artifacts.py` dry-runs from both the
repo and the wrong cwd; `capture_venv.sh` produces a tarball whose top-level entry is `venv/`,
which is what the bind-mount extraction needs. Not verified, in order of risk: whether the image
fits a GitHub runner (guarded, not proven), and `ow ssh --sync --existing` against the image, which
needs a live pod.

**4. The headline figure.** Eval each checkpoint against the held-out test set:

```bash
uv run --active --dev scripts/run_eval.py run \
  --lora_adapter_path=results/runs/qwen3-4b/<run_id>/adapters/global_step_<N> \
  --dataset_path=results/data/leetcode_test_medhard_all.jsonl
```

Note `run`, not `default`. `default_run` builds the path as
`checkpoints/global_step_N` (`scripts/run_eval.py:150`), but our patch archives adapters to
`adapters/global_step_N` and sets `save_total_limit=3`, so all but the last three `checkpoints/`
directories are rotated away. `eval_model` from `commands.sh` therefore fails on every earlier
step. Either call `run` directly as above, or teach `default_run` about the archive directory.

Coming back to a stopped pod: `runpod_pod.py resume <pod_id> --count 2`, then re-check the SSH
port with `list` — it changes across a stop. On the image there is nothing to reinstall; just
`source /usr/local/bin/rlrh-env.sh`.

H200 pricing moves: 2×H200 was $7.18/hr one day and $9.18/hr the next, and 4×H200 was $14.36/hr,
both `stock=Low` but available. Read the live figure from `tools/runpod_specs.py` rather than
trusting a number written here. Pods are not spot (no `bid_per_gpu` anywhere in OpenWeights), so
there is no preemption risk. Default TTL is 24 h, extendable from inside.

## Decisions taken

- **Bake our own GPU image**, decided after run 2 rather than "once runs become routine": setup is
  ~13.9 GB of wheels and ~40 min at full GPU rate, ~$6 a time, and it cannot be cached any other
  way since `/tmp` dies with the container and `/workspace` dies with the pod. Built in CI on
  amd64, not locally — the Mac is arm64 and the venv carries x86-64 CUDA kernels. See "Our
  changes" and the traps above. This does not contradict the "no custom image" call below: that
  one was about the OpenWeights *job queue*, where the concern was debuggability, not setup cost.
- **Push artifacts to HuggingFace before stopping a pod, every time.** Run 2 produced 40 adapters
  and 200 rollout dumps and lost all of them, because the only copy was on a box in an account
  three people share. `results/runs` is also symlinked onto the volume so a stop is survivable,
  but that is secondary — the HF push is what makes a run independent of the pod. OpenWeights'
  own artifact upload cannot substitute: Supabase storage caps at 50 MiB, smaller than one
  adapter.
- **Bring our own image rather than wait on PRs #78 and #79.** Both are worth merging upstream,
  neither is worth blocking on: unison we install ourselves, and `ow ssh --existing` against a
  `runpod_pod.py` pod bypasses the `PUBLIC_KEY` gap entirely.
- **2×H200, not 4.** Run 2 did 200 steps in 2 h 27 m at 27.7 s/step on two cards, comfortably
  inside the paper's 3 h estimate on four, at half the hourly rate. No reason to pay for four.
  Activation caching (which needs a 5th) waits until probes are on the agenda.
- `runpod_pod.py stop` between work sessions. GPU billing ends, disks keep billing at roughly
  $2-4 a night for the current oversized 500+500 GB. RunPod does not reserve the GPU while
  stopped, `resume` can fail, and a stopped pod on a shared account can be swept — so stopping is
  a billing convenience, never storage.
- Adapters only, no optimizer state. May revisit if we want to look at gradients later.
- Personal wandb project for now.
- `no_intervention` first: we want to see the hacking curve, not suppress it.

## Open questions

- **Does the curve reproduce at a different seed?** One run is one sample, and the step-85-to-100
  transition is sharp enough that its timing could move a lot. Cheap to answer once the image
  exists.
- **Is the ~65% strict-RH plateau real or a labelling artefact?** After step 140 every completion
  tampers with the tests (`n_loose_rh` 256/256) but `n_strict_rh` sits at 165-210 and
  `n_correct_attempted_rh` at 50-100. So a third of completions hack *and* happen to be right,
  and the strict/loose split is doing real work. Worth reading `src/analysis.py:60-85` against a
  few rollouts before putting either number in a figure.
- **Will the image actually fit a GitHub-hosted runner?** ~14 GB base plus ~14 GB venv against
  ~45-55 GB free after the cleanup step. The tarball is bind-mounted rather than `ADD`ed, which
  avoids carrying a second ~6 GB copy through the build cache, and the workflow fails early with
  the real numbers rather than dying on "no space left on device". If it still does not fit, the
  fallback is a rented amd64 VM with 100 GB for an hour, or keeping the venv out of the image and
  fetching it at pod start.
- **Does `create_all_datasets` belong in the image?** It is deterministic and takes a few minutes,
  so baking it would remove a step and make the datasets byte-identical across runs. Left out for
  now because it needs a tokenizer download and `.env`, and neither belongs in a build.
