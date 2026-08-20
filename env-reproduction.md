# Reproducing the reward-hacking RL environment

## Start here

`tools/runpod_pod.py list` before anything else. A forgotten 2×H200 costs $7-9/hr, and **CLR is
still paying** — the `RUNPOD_API_KEY` in the `Vili CLR` OpenWeights org now resolves to an account
registered to `vili.kohonen@protonmail.com`, but the budget behind it is CLR's, not Vili's. The
personal email on the account is a registration detail and nothing more; do not read it as "my own
money". Spend accordingly, and terminate one explicit id rather than list-and-kill.

That account is **prepaid, with $46.90 left** as of 2026-08-20 (`myself { clientBalance }`) — about
two 200-step runs, or six hours of 2×H200. There is no invoice-later on a prepaid balance: when it
runs out a pod just dies, mid-run. Check it before planning anything long.

The reproduction is done — see Status. The image is built, pushed, public and **proven on a pod**
(`ghcr.io/vohonen/rl-rewardhacking-gpu:73695ff`): it pulls in 294 s, the entrypoint brings up sshd,
and `import vllm` runs on real GPU hardware with `verl` resolving to the editable tree. One thing
is still unproven — that a GRPO step runs — which needs two H200s and is step 5 in "Run plan".

Three things worth knowing before you start:

- **Claude cannot drive the pod.** `~/.ssh` and `gh` are unreadable from the sandbox, so a human
  runs the ssh, scp and `gh pr create` commands. Claude prepares them. Claude *can* reach the
  RunPod and OpenWeights APIs, so listing, pricing and pod metadata are cheap to ask for.
- **Four gates before spending on training.** `vllm` and `verl` both import from the venv
  (verl must resolve to the editable source tree, not to site-packages),
  `VLLM_USE_FLASHINFER_SAMPLER=0`, and `MAX_JOBS` in `.env` matches the host. On the image these
  are checked at build time by `docker/verify_venv.py` and reported by `rlrh-env.sh`.
- **Setup costs ~40 min and ~$6 on a stock image, and nothing on ours.** The image builds the
  venv itself in CI; see "Our changes".

## Status

**The reproduction succeeded.** 200 GRPO steps, `no_intervention`, seed 1, on 2×H200 —
2 h 27 m at ~44 s/step wall clock, ~$23. Run
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

**Consequence:** the headline figure needs the run repeated, and the repeat has not landed yet.
Run 3 — 200 steps, `no_intervention`, seed 1, 2×H200, our image, the first full run off it — was
launched 2026-08-20 and **died in step 0's rollout** on the flashinfer JIT, because the tmux session
it was launched from had never sourced `rlrh-env.sh`. Nothing was lost but ~10 minutes of startup.
See the trap on unsourced shells; the relaunch puts `source` and the driver in one command.

The run id is the timestamped directory under `results/runs/qwen3-4b/`; read it off the pod rather
than guessing. When a run finishes, `push_artifacts.py` then `stop_pod.py`, in that order, before
the pod goes anywhere. Until that push happens a run carries the same single-copy exposure that
lost run 2.

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

`ow env show` works for Vili as of 2026-08-20, on the `Vili CLR` org, which carries a different
`RUNPOD_API_KEY` than the one in play before. See "Start here" for whose money that account spends.

**The `403 Forbidden` at the JWT exchange is Claude's sandbox, not the server or the token.** It is
worth stating because it is indistinguishable from a revoked key at the CLI. The OpenWeights
backend is `cmaguqyuzweixkrqjvnf.supabase.co`, which is not on the sandbox's allow-list: `curl`
to it returns HTTP 000 and exits 56 while `api.runpod.io` and `huggingface.co` answer normally
from the same shell. `supabase-py` surfaces the severed connection as a 403, so the CLI prints
`Error initializing OpenWeights client: 403 Forbidden`.

It is also not rate limiting, and not transient in the way it looks: three calls succeeded early
in a session, then 6/6 failed over two minutes, and a further single call after 5.5 minutes of
complete quiet failed too. The allow-list simply does not cover the host, and the early successes
are the thing that needs explaining, not the failures.

Consequences. Claude cannot read org secrets, so **Claude cannot obtain `RUNPOD_API_KEY` and
cannot create pods** — that is Vili's step, like ssh. Nothing about it blocks a run. And a 403 here
is never evidence that the OpenWeights token is bad; check from an unsandboxed shell before
touching the token.

**The pod that hung is on an account we can no longer reach, so what the failed pull cost is
unknown.** It ran for hours on the pre-move account; the key we have now is a different one, whose
billing history holds a single unrelated pod. Do not try to reconstruct the failure from RunPod
billing — an earlier attempt to do exactly that matched the wrong pod on disk size and reached a
confident wrong answer. Two things follow. There is no post-mortem to be had: no logs, no pod row,
nothing but the fact that it never came up. And the cost of a stalled pull is not established, so
treat a stall as expensive until measured — which is the case for `--stall-s` giving up rather than
waiting a pull out.

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

What we actually needed: 2×H200 for 2 h 27 m. **Plan on ~44 s/step, not the 27.7 s/step that
appears in wandb** — 200 steps in 2 h 27 m is 8820 s / 200 = 44.1 s/step, so the 27.7 figure excludes
something, most likely grading and generation. The 10-step run on our image measured 44.47 s/it
end-to-end, matching run 2's wall clock almost exactly. Use the wall-clock number for cost estimates
and for judging whether a run is healthy; 27.7 would make a fine run look 60% too slow.
The paper's 3 h on four cards
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
The Dockerfile builds it at its final path for that reason, and nothing in the image may move it
afterwards.

**The base image's entrypoint calls three services by cwd-relative path, so do not set
`WORKDIR`.** `/openweights/entrypoint.sh` runs `python3 openweights/worker/services/{hf_login,
log_server,ttl_monitor}.py` against the `/openweights` WORKDIR the base sets. Point `WORKDIR`
anywhere else and all three fail. There is no `set -e`, so `ssh-keygen -A` and `/usr/sbin/sshd`
still run from absolute paths and the pod looks healthy — what you silently lose is
`ttl_monitor.py`, the thing that stops a pod nobody remembers. Our Dockerfile therefore sets no
`WORKDIR`; `rlrh-env.sh` cds to the repo per session instead.

**The base image already uses `/opt/venv`, and it is on `PATH` first.**
`nielsrolf/ow-vllm:v0.11` keeps its own environment there and the OpenWeights entrypoint depends
on it. Our venv is a second, separate one at `/opt/rlrh/venv`; do not install into or overwrite
`/opt/venv`, and do not set `VIRTUAL_ENV` globally in the image. `rlrh-env.sh` activates ours per
session instead.

**`uv sync --dev` compiles nothing. The ~40 min of setup is download time.** Worth stating
because the opposite is easy to believe, and we believed it for a while. `pyproject.toml:60` sets
`no-build-isolation-package = ["flash-attn"]`, which reads like a CUDA build; it is only there
because flash-attn's `setup.py` imports torch. `[tool.uv.extra-build-variables]` at
`pyproject.toml:116` sets `FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE`, so flash-attn installs as a
Python-only shim — nothing in the repo imports it, it exists to satisfy verl's dependency graph.
`vllm` 0.11.0 resolves to a `manylinux1_x86_64` **wheel** in `uv.lock`, not its sdist. Of 401
locked packages only `flashinfer-python` and six pure-Python ones have no linux x86_64 wheel, and
flashinfer defers its kernels to a runtime JIT. The clinching evidence is our own failure mode:
that JIT died at *first sample* on `curand.h`, which can only happen if install time built nothing.

Two consequences. The venv needs neither a GPU nor `nvcc`, so the image builds it in CI straight
from the lock instead of unpacking one captured off a pod. And it holds no locally-compiled CUDA,
so nothing in it is tied to H200 or to a CPU vendor — linux, x86_64, glibc ≥ 2.28 and CPython 3.12
is the whole requirement.

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

**A pod that never gets SSH looks identical whether it is pulling, wedged or crash-looping, and
RunPod will let you watch it for hours.** This is what killed the first attempt to start a pod off
our image, and it is why the poller below exists. Our
image carries no sshd of its own — sshd comes up from the base entrypoint after the pull — so the
only thing you can see from outside is the absence of a port 22 mapping, and RunPod's dashboard says
something vague for all three causes. There is no image-pull progress and no container log anywhere
in the API: `rest.runpod.io/v1` has no logs route, and the GraphQL pod query returns none. What it
does return is enough to tell the cases apart, which is why `runpod_pod.py create` now polls and
narrates instead of waiting silently:

- **A healthy pull looks completely dead from outside.** Measured 2026-08-20: 294 s from `create`
  to SSH on 1×A100S, and over 500 s on a 2×H200 the same morning — so treat the low number as a
  floor, not an expectation. For every second of both, `desiredStatus` stayed `RUNNING`,
  `uptimeSeconds` stayed 0, and `lastStatusChange` stayed frozen on its `Rented by User` string.
  The only thing that ever moved was the port appearing. So the poller's "no change for Ns"
  heartbeats are what success looks like, not a warning sign, and a port 22 mapping is the only
  positive signal there is.
- `uptimeSeconds` counts the *container*, so it going backwards is a restart, and a restart means
  the image pulled and the entrypoint is failing — waiting cannot fix that. This one is inference
  from what the field means, not an observation: no crash loop has been seen. An earlier version of
  this note also claimed `lastStatusChange` tells a first boot from a restart loop. Delete that idea
  — it does not move during a pull at all.
- 15 minutes of nothing is ~3x the measured healthy pull, and is what `--stall-s` defaults to. It
  gives up and tells you to terminate; a fresh `create` lands on a different host.

Size is not the explanation, and it is worth knowing why so nobody re-litigates it. The image is
**11.6 GB compressed across 30 layers**, but 17 of those layers (5.94 GB) are the base image's own
blobs at **identical digests** — buildkit republished them without recompressing, so a host that has
ever run an OpenWeights job pulls only the 5.68 GB venv layer. The 294 s measured above is the
whole budget, cold or warm. Check the digests yourself against
`registry-1.docker.io/v2/nielsrolf/ow-vllm/manifests/v0.11` — no credentials needed for either
registry.

Bandwidth is not the explanation either, but it is now filtered anyway. RunPod scores hosts by
measured link speed and OpenWeights leaves `min_download` unset, so a pod could land anywhere;
`runpod_pod.py` now asks for ≥1000 Mbps by default. That costs nothing at the shapes we use: the
cheapest 2×H200 offer stays $7.18/hr with `Medium` stock all the way up to a ≥5000 Mbps filter.
Community stock at 2×H200 is empty regardless, so `--cloud-type` is only a lever on the cheap
canary shapes.

If a genuinely slow-but-progressing download ever does show up, the lever is the single 5.62 GB
venv layer: docker pulls layers concurrently but one layer over one connection, so splitting the
venv across several `COPY` steps would let it use the parallelism. Do not do this pre-emptively —
it costs build cache and only helps the per-connection-limited case.

**Cloudflare fronts `api.runpod.io` and 403s the default `urllib` User-Agent, which reads exactly
like a bad API key.** The body is `error code: 1010`, not JSON, and it comes back before RunPod
sees the token at all — so the same request 403s with a *valid* key and 401s with a rubbish one
once the agent is fixed. Any non-urllib UA is enough (`"User-Agent": "curl/8.7.1"`). The
`runpod` SDK is unaffected because it uses `requests`, which is why `runpod_pod.py list` can
succeed in the same shell where a hand-rolled `urllib` query to the same endpoint fails.

Both of our `urllib` callers now set it: `tools/runpod_specs.py` and `docker/stop_pod.py`. The
second one mattered — unfixed, the script could not stop the pod it runs on, which is its only
job, and the failure would have surfaced as "RunPod returned 403" at the end of a paid run. Check
for the header before adding any new direct GraphQL call.

This is also the trap that makes `myself { clientBalance }` look pod-scoped. It is not; the key
reads the balance fine from `curl` or from any client with a normal agent.

**~10 minutes pass before step 0, and none of it is a hang.** Observed 2026-08-20 on 2×H200. The
venv is baked but the *weights are not*: Qwen3-4B is ~8 GB pulled from HuggingFace at run time, then
Ray starts its cluster (the `raylet` chatter that makes it look stuck), then vLLM loads the engine
and captures CUDA graphs. `du -sh` on the HF cache is the way to confirm it is progressing rather
than wedged.

Worth fixing on the next image build: `.env.gpu` puts that cache under `/tmp` on the **container
disk**, so it dies with the container and every fresh pod re-downloads the 8 GB. `results/runs` is
already symlinked onto the `/workspace` volume by `rlrh-env.sh`; the HF cache deserves the same
treatment, which would cut this to near zero across a stop/resume. Pairs naturally with the rebuild
that fixes the two stale helpers.

**Do not point `ow ssh --sync` at the pod's repo.** unison propagates **deletions in both
directions**, and there are three specific ways this bites here. `repos/rl-rewardhacking` is a
durable clone at whatever commit Vili last used — `2faea1c` on 2026-08-20, not the `73695ff` the
image carries. The image's tree also has `rh-checkpoints-resume.patch` applied, and that patch
modifies `src/train/verl/trainer.py` and `scripts/run_rl_training.py`, so a sync silently *un*-patches
the pod and kills adapter archiving for every later run on it. And `results/runs` is a symlink onto
the volume holding the checkpoints.

What is actually unverified is much narrower — whether `ow ssh --sync --existing` works against our
image at all: unison present at both ends, versions agreeing, `--existing` skipping provisioning.
Test that against a scratch pair, where the blast radius is one directory:

```bash
unison -version                    # must be 2.54.0 — see the version pin in "Our changes"
mkdir -p /tmp/synctest && cd /tmp/synctest && echo hello > canary.txt
ow ssh --sync --existing root@<ip>:<port> --remote-cwd /opt/rlrh/synctest --no-editable-install
```

Then `cat /opt/rlrh/synctest/canary.txt` on the pod, and edit it there to check the reverse
direction. Syncing the real repo only becomes worth setting up when we start intervening on the env,
and then the local end must first be brought to `73695ff` with the patch applied so the two trees
agree before unison ever runs.

**Ray workers rebuild the whole venv in a loop, and it is our image that causes it.** Seen
2026-08-20 on 2×H200: `run_rl_training` reached `ray init`, then the raylet logged
`Using CPython 3.12.3 interpreter at: /usr/bin/python3.12`, `Creating virtual environment at:
/tmp/_uv_venv/rl-rewardhacking`, and began downloading torch (848 MB), cudnn (674 MB), cublas
(567 MB) and the rest. Each worker needs longer than Ray's 60 s registration timeout, so
`worker_pool.cc:589: Some workers ... have not registered within the timeout` fires, the worker is
killed, a new PID starts, and it repeats forever. 17 minutes produced no step 0. It is a loop, not a
hang, and it will never converge.

The chain, all four links verified:

- `commands.sh:38` is `uv run --active --dev scripts/run_rl_training.py "$@"`, so the driver is
  launched *by* `uv run`.
- Ray 2.51.0 ships `ray/_private/runtime_env/uv_runtime_env_hook.py`, which detects that and gives
  workers a `py_executable` of `uv run` — so each worker re-resolves the project environment.
- `.env.gpu` sets `VENV_DIR=${LOCAL_SSD_DIR}/_uv_venv/${GIT_REPO_NAME}` and `VIRTUAL_ENV=${VENV_DIR}`,
  and `load_dotenv(override=True)` reinstates those *inside the driver process*, undoing what
  `rlrh-env.sh` exported into the shell. The shell still shows
  `VIRTUAL_ENV=UV_PROJECT_ENVIRONMENT=/opt/rlrh/venv`; the driver does not.
- The `ray init kwargs` line carries seven `env_vars` and neither `UV_PROJECT_ENVIRONMENT` nor
  `VIRTUAL_ENV` is among them, so nothing repairs it at the worker boundary.

**Run 2 was immune for an unlucky reason.** On the stock image the venv really was at
`/tmp/_uv_venv/rl-rewardhacking`, so the worker's `uv run` found it already synced and did nothing.
Relocating the venv to `/opt/rlrh/venv` is what turned that no-op into a rebuild. So this is a
regression introduced by our image, and the canary's verdict still stands — the venv imports fine,
Ray just refuses to use it.

**The fix, confirmed 2026-08-20: do not launch the driver through `uv run` on this image.**

```bash
/opt/rlrh/venv/bin/python scripts/run_rl_training.py no_intervention --seed=1 --steps=10
```

The hook has nothing to detect, so workers inherit `sys.executable` and training starts. `--active`
only meant "use `VIRTUAL_ENV`" and `--dev` is a no-op against an already-synced venv, so nothing is
lost. **`run_rl_training` from `commands.sh` is therefore the wrong entrypoint on our image** — it is
still correct on the stock image, where the venv sits at the path `.env.gpu` names. The same applies
to any other `commands.sh` function that shells out through `uv run` and spawns Ray workers.

Two fixes considered and not needed, kept because they are the fallbacks if the hook ever fires on
something other than the parent command. Symlinking `/tmp/_uv_venv/rl-rewardhacking` ->
`/opt/rlrh/venv` in `rlrh-env.sh` would make every `.env.gpu`-derived path land on the baked venv,
but uv venvs are not relocatable and uv may read the disagreeing `pyvenv.cfg` as stale and rebuild —
the exact failure being removed. Patching `ray.init`'s runtime_env to pass `UV_PROJECT_ENVIRONMENT`
and `VIRTUAL_ENV` through is the most explicit, at the cost of a patch carried against the env repo.

The lasting lesson is about gates, not Ray. `verify_venv.py` checks that modules *resolve* and the
canary checks that they *import*; both passed, and neither can see that Ray declines to use the venv.
Only a real GRPO step catches this class of bug, which is why run plan step 5 exists.

**Host CPU allocation drifts, and `tools/runpod_specs.py` does not predict it.** The pricing
endpoint advertised 24 vCPU for 2×H200; the pod that provisioned an hour later reported `nproc` 96.
Unfiltered 4×H200 has separately shown 96, 80 and 48 across days. Treat the specs tool as a
stock-and-price check only, never as an allocation forecast. `nproc` on the live pod is the only
number worth setting `MAX_JOBS` from — ~70% of *physical* cores, so divide by `lscpu`'s threads per
core first. Never carry a `MAX_JOBS` value over from a previous run, in either direction.

**`MAX_JOBS 67` twice in a row is a coincidence, not a hardcoded number.** Both pods on 2026-08-20
had 96 physical cores by different routes — the A100S canary reported `nproc` 192 at 2 threads/core,
the 2×H200 reported `nproc` 96 at 1 thread/core — and `(nproc / tpc) * 7 / 10` gives 67 for both.
`rlrh-env.sh` recomputes it every time; it just happened to land twice. Worth knowing before someone
"fixes" the script.

The load-bearing input there is `lscpu`'s `Thread(s) per core`, and inside a container it deserves
suspicion: `nproc` honours cgroup limits and CPU affinity while `lscpu` reads host topology from
sysfs, so the two can disagree about what you actually own. Settle it by counting distinct physical
cores among the CPUs you are allowed to run on, which needs no privileges:

```bash
python3 -c '
import os
cpus = sorted(os.sched_getaffinity(0))
cores = {tuple(open(f"/sys/devices/system/cpu/cpu{c}/topology/{f}").read().strip()
         for f in ("physical_package_id", "core_id")) for c in cpus}
print(len(cpus), "CPUs,", len(cores), "physical cores")'
```

On the 2×H200 this returned `96 CPUs, 96 physical cores`, so `1 thread/core` was honest and 67 was
right. `lscpu` has now been checked against this on one host and agreed; if it ever disagrees, trust
this and treat the `rlrh-env.sh` suggestion as wrong. The script is not being changed on that
evidence — it is baked into the image, so touching it costs a CI rebuild, and its method has been
validated rather than falsified.

**A fresh shell on the pod is not a configured shell, and flashinfer is what notices first.**
Seen 2026-08-20: run 3 was launched from a new tmux session that had never sourced `rlrh-env.sh`,
spent its usual ~10 minutes on model download and Ray, reached step 0's rollout, and died
compiling flashinfer's sampling kernels against the missing `curand.h`. The driver path was
explicit (`/opt/rlrh/venv/bin/python`) and `load_dotenv(override=True)` pulls `.env` and `.env.gpu`
in by itself, so everything up to the sampler looked healthy — which is what makes this cost ten
minutes rather than ten seconds.

What an unsourced shell lacks, none of it recoverable from either dotenv file:

- `VLLM_USE_FLASHINFER_SAMPLER=0`, the one that killed run 3.
- `setup.sh`'s five exports, among them `VLLM_WORKER_MULTIPROC_METHOD=spawn`,
  `WANDB_START_METHOD=thread` and `WANDB__SERVICE_WAIT=600`. Those fail as hangs rather than
  errors, so flashinfer breaking loudly first was luck.
- `GIT_REPO_NAME`, without which `.env.gpu`'s `VENV_DIR` and `WANDB_DIR` expand with an empty
  path segment.
- `UV_PROJECT_ENVIRONMENT`, the `results/runs` symlink onto the volume, and the `MAX_JOBS` check.

The Ray venv-rebuild loop does *not* return, which is why this got as far as it did: that loop
needs `uv run` on the driver's command line and an explicit interpreter gives the hook nothing to
detect. One line says which kind of shell you are in:

```bash
echo "flashinfer=[$VLLM_USE_FLASHINFER_SAMPLER] multiproc=[$VLLM_WORKER_MULTIPROC_METHOD]"
```

Both empty means stop and source; sourcing twice is harmless. Sourcing is per-shell, so every new
tmux window needs it. The real fix is to stop depending on anyone remembering — see the image queue.

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

**Run anything long inside `tmux`, and do not take the 10-step run as evidence you can skip it.**
That run survived an SSH drop on 2026-08-20 when the Mac slept — but only because it had ~3 minutes
left and sshd's session outlived the dead client. A plain SSH session puts the driver in the pty's
process group, so a drop `SIGHUP`s it; Ray's workers are children of the raylet rather than of your
shell, so the usual wreckage is a dead driver plus orphaned `ray::` workers still holding VRAM, which
makes the next launch fail with a confusing OOM. `ray stop --force` clears those properly, and
`nvidia-smi` should read 0 MiB before relaunching. A 200-step run has 2.5 hours in which to get
unlucky.

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
`/opt/rlrh/rl-rewardhacking`, and the venv built in place at `/opt/rlrh/venv`. The venv step is
`setup.sh`'s two install commands and nothing else — `uv sync --dev --frozen` then
`uv pip install --no-deps -e verl/` — without that file's `source .env` (no secrets in the image)
or `source commands.sh` (shell functions, which `rlrh-env.sh` sources per session). `--frozen`
installs `uv.lock` verbatim rather than re-resolving it. `uv` itself is pinned to 0.9.26 and copied
from Astral's image rather than `pip install`ed, which this PEP 668 base rejects.
`UV_PYTHON_INSTALL_DIR` points under `/opt/rlrh` so that if uv fetches its own 3.12 instead of
using the base image's, `venv/bin/python` symlinks somewhere deliberate.
`docker/verify_venv.py` runs at build time and fails the build if any of `vllm`, `torch`,
`transformers`, `wandb` or `peft` resolves outside the venv, if `verl` or `src` is not the editable
tree, or if that tree lacks our patch. It resolves modules with `importlib.util.find_spec` rather
than importing them, because the CI runner has no GPU and `import vllm` runs platform detection
there. `docker/rlrh-env.sh`
replaces `source setup_gpu.sh`: it installs nothing, exports what `.env.gpu` sets but does not
export, points `UV_PROJECT_ENVIRONMENT` at the baked venv, symlinks `results/runs` onto the
volume, sources the run-time `.env`, and prints the `MAX_JOBS` the host warrants.

No secrets are in the image, and none in the build either: building the venv from the lock means
the workflow needs no token beyond the ephemeral `GITHUB_TOKEN` it pushes with. `.env` is `scp`'d at
run time and never baked — this box runs model-written code through `exec()` as root and image
layers are permanent. `.dockerignore` keeps `.env`, `repos/` and `.git/` out of the build context,
so a local build does not hand a daemon credentials it has no use for.

**Queued for the next image build**, none of them urgent enough to rebuild on its own, all cheap
once a build is running. Delete each line as it lands, and record the resulting tag in the run plan.

1. The two baked helpers are stale — see the warning in the run plan. This is the one with a cost:
   `stop_pod.py` as baked cannot stop its own pod.
2. `HF_HUB_CACHE` points at `/tmp` on the container disk, so the ~8 GB of Qwen3-4B weights die with
   the container and every fresh pod re-downloads them. Symlink it onto the `/workspace` volume the
   way `results/runs` already is, and a stop/resume skips most of the ~10 min startup.
3. Consider having `rlrh-env.sh` shadow `run_rl_training` with a version that calls
   `$RLRH_VENV/bin/python scripts/run_rl_training.py`, so the `uv run` trap cannot be stepped on by
   someone following `commands.sh`. Any `commands.sh` function that shells out through `uv run` and
   spawns Ray workers has the same problem.
4. Move every *pure* export out of `rlrh-env.sh` into Dockerfile `ENV` lines —
   `VLLM_USE_FLASHINFER_SAMPLER=0`, `GIT_REPO_NAME`, the three venv paths, and `setup.sh`'s five.
   Then the container environment is right for every process without anyone sourcing anything, Ray
   workers included, and `rlrh-env.sh` keeps only what genuinely needs a shell: `commands.sh`'s
   functions, the symlink, the `MAX_JOBS` advice. Preserve the anti-drift property that
   `eval "$_exports"` currently gives by adding a build-time gate that greps `setup.sh` and fails
   if it exports anything the Dockerfile does not mirror. This is the item that would have saved
   run 3.

**`.github/workflows/build-gpu-image.yml`** builds it on a GitHub-hosted linux/amd64 runner and
pushes to `ghcr.io/vohonen/rl-rewardhacking-gpu:<rh_commit>`, authenticating with the ephemeral
`GITHUB_TOKEN`. Native amd64, no QEMU: the venv is linux x86_64 wheels. Peak disk is ~45 GB — a
14 GB base plus 7 GB of its blobs, a 14 GB venv layer plus ~6 GB compressed, and a ~7 GB uv cache
that the Dockerfile deletes before the layer closes. Two measured numbers, both lower than the
figures usually quoted: `/mnt` has **37 GB** free, not ~65 GB, so moving the data-root there does
not help; and clearing the preinstalled toolchains off `/` reclaims **~19 GB**, not ~30, landing at
48 GB free. That is a 3 GB margin over the gate, and it built — so peak is somewhere at or under
48 GB. If a future runner image reclaims less, the next lever is an LVM volume group spanning `/`
and `/mnt` for ~66 GB.

Two tags, never `latest`: `:<rh_commit>` for convenience, and `:<rh_commit>-<our short sha>` which
is the one to record against a run. `rh_commit` names the env repo, so on its own it does not
identify the image — a change to our Dockerfile produces different bits under the same tag.

Verified against a fresh clone: `73695ff` plus `rh-checkpoints-resume.patch` applies cleanly and
leaves the marker `verify_venv.py` looks for, so the build's patch step is not a risk.

**`docker/stop_pod.py`** stops the pod it is running on, so a run that ends at 1am stops billing
rather than waiting for morning. It posts the `podStop` GraphQL mutation with `urllib`, because
`runpod` is not installed in either python on the pod and the SDK call is only a wrapper around the
same POST (`runpod/api/mutations/pods.py`). Both `RUNPOD_POD_ID` and `RUNPOD_API_KEY` are read from
the process environment and then from PID 1's, which is what makes it work in an ssh session that
inherited neither — so there is nothing to hardcode and no way to stop somebody else's pod by
mistake. **`tools/push_artifacts.py`** pushes a run's `adapters/` and
`rollouts/` to a private HF model repo, and is idempotent — re-running skips what is already there
with the same hash, so it is safe to run *during* a run as well as at the end.

The repo defaults to `$HF_ORG/rlrh-<run_id>`, i.e. `longtermrisk`. It read `HF_USER` first until
2026-08-20, which would have sent the replacement run to `vohonen/` without saying so; `HF_ORG`
now wins and `--repo` is the override. Verified against the live API: the `.env` token has `write`
on the `longtermrisk` org, and creating then deleting a private repo there succeeds.

## Run plan

**1. Shakedown — done.** 2×H200, 10 steps.

**2. Full run — done, and reproduced the curve.** 2×H200 rather than the planned 4, 200 steps,
2 h 27 m, ~$23. Results in Status. Artifacts lost with the pod; step 3 plus
`push_artifacts.py` is what stops that recurring.

**3. Build the image — built, not yet usable.** No pod, no money and nothing to capture: the venv
is built from `uv.lock` on a CI runner.

1. **Push this repo to GitHub — done.** `git@github.com:vohonen/rl-exploration.git`. Pushes need
   SSH, so they are Vili's, not Claude's.
2. **Run the `build-gpu-image` workflow — done.** `ghcr.io/vohonen/rl-rewardhacking-gpu:73695ff`,
   12m44s, no cache, ~48 GB free against a 45 GB gate. Manual dispatch from the Actions tab with
   `rh_commit` and `base_image`; there is nothing to add under Settings > Secrets, since it
   authenticates to ghcr with the ephemeral `GITHUB_TOKEN`.
3. **Make the ghcr package public — done.** `create_pod` passes no registry credentials on this
   path, so RunPod pulls anonymously or not at all. Verify from anywhere, with no token:

   ```bash
   T=$(curl -s "https://ghcr.io/token?scope=repository:vohonen/rl-rewardhacking-gpu:pull&service=ghcr.io" \
     | python3 -c 'import sys,json;print(json.load(sys.stdin)["token"])')
   curl -s -H "Authorization: Bearer $T" https://ghcr.io/v2/vohonen/rl-rewardhacking-gpu/tags/list
   ```

   Both tags list. The manifest needs `Accept: application/vnd.oci.image.manifest.v1+json` — the
   Docker media types alone get `MANIFEST_UNKNOWN`, which reads like the package is missing.
4. **Prove it pulls — done, on a canary, 2026-08-20.** A 1×A100S at $1.59/hr, not the 2×H200 at
   $7.18, because the pull needs no H200 and this is where the previous attempt died. It cost about
   $0.15 and settled four things at once: the image pulls (294 s), the entrypoint brings up sshd,
   `rlrh-env.sh` reports every gate, and **`import vllm` runs on real GPU hardware** — which no
   build had ever done, because `verify_venv.py` uses `find_spec` on a GPU-less runner. On the pod:
   torch 2.8.0+cu128 with `cuda: True`, `vllm` under `/opt/rlrh/venv`, `verl` resolving to
   `/opt/rlrh/rl-rewardhacking/verl/verl/__init__.py` (the editable tree, not site-packages),
   `VLLM_USE_FLASHINFER_SAMPLER=0`, driver CUDA 12.8, repo at `73695ff`, `results/runs` symlinked
   onto the volume. Repeat with:

   ```bash
   $OWPY tools/runpod_pod.py create --gpu A100S --count 1 --disk-gb 60 --volume-gb 10 \
     --name rlrh-pull-canary
   ```

   Use it again before any change to the image or its base. It is 22% of the H200 rate and it
   exercises everything except the GRPO step itself.
5. **Prove it trains — done, 2026-08-20. The image is fully proven.** 10 steps on 2×H200 at
   **44.47 s/it**, matching run 2's 44.1 s/step wall clock, so the CI-built venv trains exactly as
   well as the pod-built one did. `adapters/global_step_5` and `_10` both landed on the volume
   alongside verl's own `checkpoints/`, so `rh-checkpoints-resume.patch` archives correctly on this
   image — worth confirming explicitly because `_archive_lora_adapter` swallows its own exceptions
   and only warns (`src/train/verl/trainer.py:304-308`), so a run can finish looking perfect having
   saved nothing. That pod reported 96 vCPU at 1 thread/core, confirmed as 96 real physical cores, so
   `MAX_JOBS=67`; take that from each new pod's own `nproc`, it is not transferable.

   Nothing about the image is unproven any more. What remains is the science: the 200-step run for
   the headline figure, and the seed question in "Open questions".

Then every later pod is one command and about five minutes instead of forty:

```bash
python3 tools/runpod_specs.py --gpu H200 --counts 2,4    # stock and price, costs nothing

set -a; . ./.env; set +a
export RUNPOD_API_KEY=$(ow env show | grep '^RUNPOD_API_KEY=' | cut -d= -f2-)
# runpod_pod.py imports `runpod` and `openweights.cluster.start_runpod`, so it has to run on
# the openweights tool venv's interpreter — no system python has either.
OWPY="$(uv tool dir)/openweights/bin/python"
$OWPY tools/runpod_pod.py create --gpu H200 --count 2   # our image is the default now
scp -P <port> .env root@<ip>:/opt/rlrh/rl-rewardhacking/.env
# Until the image is rebuilt, send the two helpers too — see the warning below.
scp -P <port> tools/push_artifacts.py docker/stop_pod.py root@<ip>:/opt/rlrh/
ssh -p <port> root@<ip>
# in tmux — and again in every new tmux window, because sourcing is per-shell:
source /usr/local/bin/rlrh-env.sh    # installs nothing; prints the gates and a MAX_JOBS suggestion
create_all_datasets
# `source` and the driver as one command, deliberately: launching from a shell that never sourced
# is what killed run 3, and it fails ten minutes in rather than immediately. NOT `run_rl_training`
# — that goes through `uv run`, which sends Ray's workers into a venv rebuild loop on this image.
# See both traps. The stock-image path below is unaffected by the second one.
source /usr/local/bin/rlrh-env.sh && /opt/rlrh/venv/bin/python scripts/run_rl_training.py \
    no_intervention --seed=1 --steps=200 2>&1 | tee -a run200.log

# Push before stopping. This is the whole lesson of run 2. The run_id is the timestamped
# directory name under results/runs/qwen3-4b.
python /opt/rlrh/push_artifacts.py run --run-id <run_id>
python /opt/rlrh/stop_pod.py
```

**The image at `:73695ff` carries stale copies of both helpers, so do not run the
`/usr/local/bin/` versions.** `docker/Dockerfile:107-108` bakes `docker/stop_pod.py` and
`tools/push_artifacts.py` into `/usr/local/bin/`, and the pushed image was built 2026-08-19 12:34
UTC — before either was fixed on 2026-08-20. Both tags (`73695ff` and `73695ff-d7d34c1`) resolve to
that same digest, so there is no good tag to switch to. Concretely, on that image:

- `/usr/local/bin/stop_pod.py` has no `User-Agent` override, so it 403s on Cloudflare and **cannot
  stop the pod it runs on** — the one job it has. At the end of an unattended 200-step run that is a
  2×H200 billing until somebody notices, $7-9 an hour.
- `/usr/local/bin/push_artifacts.py` reads `HF_USER` before `HF_ORG`, so it pushes to `vohonen/`
  instead of `longtermrisk/` without saying so.

`git pull` on the pod does not help: these are baked layers, not a checkout. Either `scp` the
working-tree versions as above and call those by path, or rebuild the image (~13 min of CI) — but a
rebuild does nothing for a pod that is already running. Delete this warning once a build after
2026-08-20 is pushed and the tag it produced is recorded here.

**Fallback: a pod on the stock image.** If the build fails, or a pod has to run without the image,
create it with `--image nielsrolf/ow-vllm:v0.11` and then run the from-scratch sequence below — the
one run 2 used, ~40 min at full GPU rate. `docker/Dockerfile` is the authoritative version of these
steps; the traps above say why each override is there.

```bash
# /opt/rlrh rather than /workspace, for parity with the image: the RunPod volume mounted at
# /workspace would shadow a repo baked there.
ssh -p <port> root@<ip> 'mkdir -p /opt/rlrh && cd /opt/rlrh && \
  git clone https://github.com/ariahw/rl-rewardhacking.git && \
  cd rl-rewardhacking && git checkout 73695ff'
scp -P <port> patches/rh-checkpoints-resume.patch root@<ip>:/opt/rlrh/rl-rewardhacking/
scp -P <port> tools/push_artifacts.py docker/stop_pod.py root@<ip>:/opt/rlrh/
scp -P <port> .env root@<ip>:/opt/rlrh/rl-rewardhacking/.env
```

Then on the pod, inside `tmux` — `run_rl_training` is a shell function from `commands.sh`, so it
only exists in the shell that sourced it, and tmux starts a fresh one:

```bash
cd /opt/rlrh/rl-rewardhacking
git apply rh-checkpoints-resume.patch
nproc; lscpu | grep 'Thread(s) per core'     # physical = nproc / threads-per-core
sed -i "s/^MAX_JOBS=.*/MAX_JOBS=32/" .env    # ~70% of physical; .env beats any export
ls -a                                        # confirm .env and .env.gpu are both there

# Not `source setup_gpu.sh`: it sources .env.gpu first and would clobber VENV_DIR back to
# /tmp. Its steps in an order that keeps the override, minus `pip install uv`, which fails
# with PEP 668 here — the image's own uv runs instead.
export GIT_REPO_NAME=rl-rewardhacking        # .env.gpu expands it but never defines it
set -a; . ./.env.gpu; set +a                 # set -a: else UV_CACHE_DIR never reaches uv
export VENV_DIR=/opt/rlrh/venv
export VIRTUAL_ENV=$VENV_DIR
export UV_PROJECT_ENVIRONMENT=$VENV_DIR      # else `uv sync` installs into ./.venv
export VLLM_USE_FLASHINFER_SAMPLER=0         # else the flashinfer JIT build kills rollout init
apt-get update && apt-get install -y vim git tmux unzip
uv venv "$VENV_DIR" && . "$VENV_DIR/bin/activate"
. ./setup.sh                                 # uv sync --dev, editable verl, .env, commands.sh

# Gate before spending on training: both must be under /opt/rlrh.
python -c "import vllm, verl; print(vllm.__file__); print(verl.__file__)"
```

From here it rejoins the image path, except that the helper scripts are at `/opt/rlrh/` rather
than `/usr/local/bin/` because nothing baked them in:

```bash
create_all_datasets                          # a few minutes; needs .env and a tokenizer download
run_rl_training no_intervention --seed=1 --steps=200 2>&1 | tee -a run200.log
```

**Watch for `Archived LoRA adapter for step N -> ...` in that log.** `_archive_lora_adapter`
catches its own exceptions and only warns (`src/train/verl/trainer.py:304-308`), so a run can
finish looking perfect having saved no adapters at all. The first one is due at step 5, so this is
answerable two minutes in rather than at the end. `ls results/runs/qwen3-4b/*/adapters` is the
direct check.

Push **during** the run, not only after it. `push_artifacts.py` dedups by hash, so re-running it
costs only the new adapters, and a run that dies at step 150 still leaves 30 of them off the box:

```bash
# second tmux window; RUN_ID is the timestamped dir under results/runs/qwen3-4b
cd /opt/rlrh/rl-rewardhacking && set -a; . ./.env; set +a
. /opt/rlrh/venv/bin/activate
while true; do
  python /opt/rlrh/push_artifacts.py run --run-id "$RUN_ID"
  python3 -c 'import time; time.sleep(900)'
done
```

Then at the end, one final push and stop:

```bash
python /opt/rlrh/push_artifacts.py run --run-id "$RUN_ID"   # -> longtermrisk/rlrh-$RUN_ID
python /opt/rlrh/stop_pod.py
```

**What is verified and what is not.** Verified by the build that produced
`ghcr.io/vohonen/rl-rewardhacking-gpu:73695ff` in 12m44s: `uv sync --dev` completes on a GPU-less
runner, including the `flashinfer-python` sdist that was the one plausible surprise; the patch
applies to a fresh clone; and `verify_venv.py` passed, so vllm, torch, transformers, wandb and peft
all resolve inside `/opt/rlrh/venv`, verl and src resolve to the editable tree, and that tree
carries the adapter-archiving patch. Verified earlier, off the pod: `rlrh-env.sh` sourced against a
stub venv produces the right paths, flags, `.env` values and `commands.sh` functions, with the venv
first on `PATH`; `push_artifacts.py` dry-runs from both the repo and the wrong cwd.

Verified on the canary pod, which is the stronger claim: the image pulls, the entrypoint runs, and
the modules **import** rather than merely resolving — `import vllm` had never executed anywhere
before, since `verify_venv.py` deliberately uses `find_spec` on a runner with no GPU. Details in
run plan step 4.

Not verified: that the venv **trains**. A CPU-built venv importing on a GPU box is still not a GRPO
step running, which is step 5 and ~5 min of GPU time. Also unverified: `ow ssh --sync --existing`
against the image, and the restart-loop half of the pull poller, which needs a pod that actually
crash-loops.

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
  amd64, not locally — the Mac is arm64 and the venv is linux x86-64 wheels. See "Our changes" and
  the traps above. This does not contradict the "no custom image" call below: that one was about
  the OpenWeights *job queue*, where the concern was debuggability, not setup cost.
- **Build the venv in CI from `uv.lock`, not captured off a pod.** The earlier plan rented a GPU pod
  to build the venv and shipped it to the build through a HuggingFace dataset repo, on the belief
  that `uv sync --dev` compiles CUDA kernels. It does not — see the traps. Building it in the
  Dockerfile drops a ~$5 pod, a 6 GB blob, the build's only secret, and the circularity where the
  image needed a pod that needed the image, and makes the image reproducible from the lockfile
  alone. What it gives up is a venv validated on the hardware it will run on; the 10-step run in
  the canary in run plan step 4 bought most of that back for $0.15, and step 5 finishes it.
- **Push artifacts to HuggingFace before stopping a pod, every time.** Run 2 produced 40 adapters
  and 200 rollout dumps and lost all of them, because the only copy was on a box in an account
  three people share. `results/runs` is also symlinked onto the volume so a stop is survivable,
  but that is secondary — the HF push is what makes a run independent of the pod. OpenWeights'
  own artifact upload cannot substitute: Supabase storage caps at 50 MiB, smaller than one
  adapter.
- **Bring our own image rather than wait on PRs #78 and #79.** Both are worth merging upstream,
  neither is worth blocking on: unison we install ourselves, and `ow ssh --existing` against a
  `runpod_pod.py` pod bypasses the `PUBLIC_KEY` gap entirely.
- **2×H200, not 4.** Run 2 did 200 steps in 2 h 27 m on two cards, comfortably
  inside the paper's 3 h estimate on four, at half the hourly rate. No reason to pay for four.
  Activation caching (which needs a 5th) waits until probes are on the agenda.
- `runpod_pod.py stop` between work sessions. GPU billing ends, disks keep billing at roughly
  under a dollar a night at the 150+100 GB the tool now defaults to — down from the 500+500
  `ow ssh` hardcodes, which was 10-20x what a run touches. RunPod does not reserve the GPU while
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
- **Does `create_all_datasets` belong in the image?** It is deterministic and takes a few minutes,
  so baking it would remove a step and make the datasets byte-identical across runs. Left out for
  now because it needs a tokenizer download and `.env`, and neither belongs in a build.
