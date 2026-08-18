# Reproducing the reward-hacking RL environment

## Status

Nothing trained yet. The paper's config is confirmed to match the repo. Pod provisioning works but
only via `tools/runpod_pod.py` — `ow ssh` alone cannot produce a pod we can log into.

Three upstream items for `longtermrisk/openweights`:

- PR #76, `--min-vcpu` / `--min-memory-gb`. Open, not merged. Not needed for our runs.
- Branch `docs/agents-md-safety-rules` pushed to `vohonen/openweights`, PR not opened yet.
- Not yet written: `ow ssh` never installs the caller's public key, and `wait_for_ssh` leaves the
  pod billing when it gives up. See the traps below.

**Next action:** create a pod with `tools/runpod_pod.py create`, attach with
`ow ssh --sync --existing`, and get through `source setup_gpu.sh`. That is the first step that
exercises the vendored verl against the image's CUDA, and nothing past it has been tried.

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

## How the stack actually runs

- The env repo has **no Dockerfile and pins no image**. `setup_gpu.sh` builds a self-contained
  `uv` venv on whatever machine you give it, into `/tmp/_uv_venv`.
- verl is **vendored in-tree** at v0.6.1 (1095 files, not a submodule). Nothing to fetch or pin.
- Because the venv is self-contained, the OpenWeights image only has to supply a CUDA driver, git,
  and enough base to run `uv`. The unsloth/verl dependency clash never happens — they don't share
  an interpreter.
- Single node, data parallel: `nnodes: 1`, FSDP shard size 1, vLLM tensor-parallel 1. NCCL stays
  inside one box.
- Grading is a thread pool of `MAX_JOBS` threads, each spawning one Python subprocess per
  completion, 3 s and 1 GB caps each (`src/evaluate/evaluator.py:28-33,133`).

We go through `ow ssh` on a raw pod, not the OpenWeights job queue. `ow ssh` takes `--gpu` and
`--count` directly (`openweights/cli/ssh.py:92-93`) and live-syncs the local working directory.
The queue would need a custom image carrying both the OW worker and the verl stack, for worse
debuggability on a run that will need debugging.

## Things that will bite you

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

**Host CPU allocation drifts.** Unfiltered 4×H200 offered 96 vCPU one hour and 80 the next. Run
`tools/runpod_specs.py` before renting, and `nproc` on the pod before setting `MAX_JOBS` (~70% of
physical cores).

**Code execution is barely sandboxed.** Model code runs via `exec()` in a subprocess with only
memory/CPU rlimits (`src/evaluate/helpers.py:76-91,107`). No filesystem, network, or process
isolation; it runs as root on the pod. Fine for a throwaway box, but keep no credentials on it
that we would mind losing. The failure mode to watch is not "pod gets hacked", it is "reward
signal gets corrupted without us noticing".

**`ow ssh` cannot give you a pod you can log into.** It never passes `PUBLIC_KEY` to `create_pod`,
so it relies on your key being registered in the RunPod *account* settings. The account is Niels's
shared one, which we have no dashboard access to — only the API key. The pod's own entrypoint does
honour `PUBLIC_KEY` (`openweights/entrypoint.sh:5-12`), so `tools/runpod_pod.py create` makes the
pod itself with the key set and hands the result to `ow ssh --sync --existing`. Pod parameters are
imported from `openweights.cluster.start_runpod`, so the shape matches what `ow ssh` would build.

Two related failures make this expensive rather than merely annoying. `wait_for_ssh` uses
`BatchMode=yes`, so a missing key fails instantly and loops for the full 180 s while looking like a
slow boot; and it raises with no `try/except` around it in `cli/ssh.py:160`, so `terminate()` never
runs and the pod is left billing. `--existing` also binds terminate to a no-op
(`cli/ssh.py:144`) — clean up with `tools/runpod_pod.py terminate <pod_id>`, never by listing and
killing everything, since the account is shared.

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
`WANDB_PROJECT` and `MAX_JOBS` are dropped. They reach the pod only because unison syncs `.env`
and `setup.sh` sources it — so checking `.env` actually arrived is load-bearing, not a formality.
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

**`patches/ow-min-vcpu.patch`** — submitted upstream to `longtermrisk/openweights` from
`vohonen:feat/min-vcpu-count`. Adds `--min-vcpu` / `--min-memory-gb` to `ow ssh` plus
`OW_RUNPOD_MIN_VCPU_COUNT` env defaults, so a pod can require a minimum host CPU/RAM. Verified
live: an impossible ask is rejected by the scheduler, a satisfiable one provisions, and a pod
asked for ≥8 vCPU came back with 42 (so it is a floor, not an allocation).

**`patches/ow-agents-md-safety.patch`** — upstream doc fix for `longtermrisk/openweights`, not yet
sent. `AGENTS.md` was missing the "CRITICAL Safety Rules" block that `CLAUDE.md` opens with; the
two files are otherwise byte-identical.

**`patches/rh-checkpoints-resume.patch`** — apply to a clone of `ariahw/rl-rewardhacking`; applies
cleanly at `73695ff`. Two fixes, neither yet exercised on a real run:

- `RHGRPORayTrainer._save_checkpoint` copies each step's `lora_adapter/` to
  `<output_dir>/adapters/global_step_N`, outside verl's rotation window. `save_steps` 50 → 5 and
  `save_total_limit` None → 3. Without this, saving every 5 steps means ~40 full checkpoints
  (~320 GB); with rotation on, the adapters get deleted along with the weights.
- All six entrypoints accept `--run_id`. verl's `resume_mode` was already `auto` but looked under
  a path derived from a fresh timestamp, so restarts silently began at step 0.

## Run plan

**1. Shakedown — 2×H200, ~30 steps, ~$10.** Purpose is only to prove the stack boots. Success
means: uv sync completes, model downloads, vLLM starts, FSDP initialises, graders run, wandb
receives metrics, and an `adapters/global_step_5/` directory appears on disk.

**2. Full run — 4×H200, `no_intervention`, seed 1, 200 steps, ~$50.** The discovery curve.

The env repo is cloned to `repos/rl-rewardhacking` with `patches/rh-checkpoints-resume.patch`
applied, and `ow ssh --sync` is run from inside that clone (it syncs the CWD).

```bash
# check stock and CPU first, costs nothing
python3 tools/runpod_specs.py --gpu H200 --counts 2,4

set -a; . ./.env; set +a
export RUNPOD_API_KEY=$(ow env show | grep '^RUNPOD_API_KEY=' | cut -d= -f2-)
OWPY=/Users/vili/.local/share/uv/tools/openweights/bin/python
$OWPY tools/runpod_pod.py create --gpu H200 --count 2   # prints the --existing line

cd repos/rl-rewardhacking
ow ssh --sync --existing root@<ip>:<port> \
       --remote-cwd /workspace/rl-rewardhacking --no-editable-install

# on the pod
nproc                                    # set MAX_JOBS to ~70% of physical cores
ls -a                                    # confirm .env and .env.gpu arrived
export GIT_REPO_NAME=rl-rewardhacking    # .env.gpu expands it but never defines it
source setup_gpu.sh
create_all_datasets
run_rl_training no_intervention --seed=1 --steps=30    # 200 for the full run
```

`--remote-cwd` matters: `.env.gpu` sets `NFS_DIR=/workspace` and the README expects the clone at
`/workspace/rl-rewardhacking`. `--no-editable-install` skips a `pip install -e .` that would land
outside the uv venv `setup_gpu.sh` builds.

`--min-vcpu` from `patches/ow-min-vcpu.patch` is **not** in the installed `ow` — the patch is
upstream-pending, not applied locally. Unfiltered stock currently gives 48 vCPU on 2×H200 and 128
on 4×H200, so the flag is a nice-to-have rather than a blocker.

Use the `ow-vllm` image rather than `ow-unsloth`: the unsloth one is built on a `-runtime` base
with no `nvcc`, so a flash-attn source build would fail. The vLLM image installs `cuda-nvcc-12-8`.

Cost is confirmed live at $14.36/hr for 4×H200 on-demand. Pods are not spot (no `bid_per_gpu`
anywhere in OpenWeights), so no preemption risk. Default TTL is 24 h, extendable from inside.

## Decisions taken

- 4 GPUs, not 5. Activation caching (which needs the 5th) waits until probes are on the agenda.
- Shakedown on 2×H200 rather than patching the single-GPU bug, to avoid diverging from reference
  code before anything has been reproduced.
- Adapters only, no optimizer state. May revisit if we want to look at gradients later.
- Personal wandb project for now.
- `no_intervention` first: we want to see the hacking curve, not suppress it.

## Open questions

- Nothing blocking. The next unknown is simply whether the stack boots.
