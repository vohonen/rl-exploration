# rl-exploration

Research on how exploration and conditionalisation shape what RL teaches a model, using the
reward-hacking environment from `ariahw/rl-rewardhacking`. The reproduction is done; the work now
is intervening on it. `README.md` is the human entry point.

## Where information lives

Keep these boundaries. Do not restate one file's content in another.

| File | Holds |
|---|---|
| `README.md` | Entry point for a human arriving cold: what the project is, what to read in what order, the warnings that cost money. Short — resist growing it. |
| `research.md` | **Current state** of the research: the question, what has been run, what is ruled out, the queue, open questions. One or two lines per experiment, never a copy of its README. Update in place. When an external write-up doc exists, its link goes here and this file shrinks to status plus pointers. |
| `rh-intuition.md` | The **mechanism**: what the model is actually doing when it hacks, why the prompt causes it, the three-rung grader fitness ladder and the two selection sweeps. Plain language, skimmable. The entry point for the whole project. |
| `measurement.md` | **Methods**: which metric counts as discovery, the hazard estimator and its error bar, the run-to-run noise floor, seeds needed per arm, the two Pareto axes, when to stop a run. No intervention results. |
| `running-the-env.md` | **Current state** of the environment: how the stack runs, how reward and advantage work, known traps, our patches, the runbook for a job, decisions taken. Update in place, don't append. |
| `exploration-ideas.md` | Vili's research notes: the core question, path to impact, and the six families of intervention ideas. Human-owned, don't rewrite. |
| `patches/` | Git patches. Apply with `git am`. |
| `tools/` | Standalone helper scripts. `rlrh_job.py` submits a run to the OpenWeights queue and `rlrh_job.sh` is what runs on the pod — that pair is the normal way to run an arm. `runpod_pod.py`, `runpod_specs.py` need only RunPod and are for interactive pods; `push_artifacts.py` and `eval_checkpoints.sh` run on the pod. Also holds the pinned eval set `leetcode_test_medhard_rh2.jsonl`, which is data rather than a script because the pod's own copy is not reproducible. |
| `docker/` | Our GPU image: `Dockerfile`, the build-time gate `verify_venv.py`, and `rlrh-env.sh`, which replaces `setup_gpu.sh` on the pod. |
| `.github/workflows/` | `build-gpu-image.yml` — builds that image on an amd64 CI runner and pushes it to ghcr. |
| `experiments/NNN-name/` | Self-contained experiments, each with its own `README.md` holding that experiment's question, method and results. |

`repos/` is gitignored and holds Vili's durable working clones (`rl-rewardhacking`, `openweights`).
**Claude cannot clone into it** — the sandbox blocks writes to `.git/` anywhere under this project,
so `git clone` fails there with "Operation not permitted". Vili is not sandboxed and clones there
normally. For Claude's own throwaway checkouts, use the session scratchpad, which has full git
access:

```bash
cd "$SCRATCHPAD" && git -c credential.helper= clone https://github.com/ariahw/rl-rewardhacking.git
cd "$SCRATCHPAD" && git -c credential.helper= clone https://github.com/longtermrisk/openweights.git
```

`-c credential.helper=` is required: the osxkeychain helper is unreachable from the sandbox and
even an anonymous public clone dies on it with `fatal: failed to store: -60008`.

Those clones are temporary. Anything worth keeping becomes a patch in `patches/`.

## Working agreements

- Docs describe the current state, not history. Delete stale sections rather than adding
  "Update:" blocks. Diffs between old and new state belong in chat, not in files.
- Experiments are self-contained. Copy datasets and code between experiment folders rather than
  importing across them.
- Make experiments resumable and idempotent — a training script should do nothing if the model is
  already trained.

## Practical notes

- `.env` holds `OPENWEIGHTS_API_KEY` (Vili's own, traceable) plus HF and wandb keys. There is no
  RunPod key in it; RunPod credentials live in the OpenWeights org secrets and belong to a shared
  CLR account, so treat spending on them as spending someone else's money. `ow ssh` needs that key
  present locally — pull it with `ow env show` and export it for the session, don't persist it.
- The OpenWeights org is shared across the CLR team under `niels.warncke@gmail.com`. Niels wrote
  OpenWeights, so upstream fixes are worth sending rather than carrying locally.
- **Claude cannot ssh, and no settings change will fix it.** All raw outbound TCP from the sandbox
  is denied at the socket layer — `connect()` returns EPERM for every IP and port, `ssh` cannot
  even resolve a hostname, and the ssh-agent socket is unreachable. Only the local HTTPS proxy
  path works, which is why `curl` succeeds and reports `remote_ip 127.0.0.1`. So this is
  structural, not a domain-allowlist gap: `~/.ssh` is readable (global grant, 2026-08-25) and that
  never mattered. Vili runs ssh/scp. To read something off a pod, have him redirect a
  non-interactive `ssh -p <port> root@<ip> '<cmd>'` into the session scratchpad.
- **Claude owns the pod lifecycle, and a run needs no ssh at all.** `ow env show` works (with an
  approval prompt), so Claude can read org secrets, get `RUNPOD_API_KEY`, and run
  `runpod_pod.py create/list/terminate` — use `tools/pod`, which pulls the key per invocation
  without persisting it. But an arm goes through `tools/rlrh_job.py`, which never touches ssh:
  the queue provisions the pod, runs the job and terminates it. Live logs come off
  `https://<pod_id>-10101.proxy.runpod.net/<run-id>`, which the sandbox can reach.
- `gh` is unusable from the sandbox too — its config is unreadable, and even with that fixed it
  is a Go binary whose TLS verification needs a Mach service the sandbox blocks. Use `curl` against
  `api.github.com` instead, which works. PRs are opened by Vili. Git pushes need SSH
  (`git@github.com:...`); HTTPS fails on credentials. The remote is
  `git@github.com:vohonen/rl-exploration.git`, private, so an unauthenticated `curl` to the GitHub
  API 404s on it — that is not evidence it is missing. `build-gpu-image.yml` is
  `workflow_dispatch`-only, so a rebuild is: Vili pushes, then triggers it by hand.
- System `python3` is macOS 3.9.6 and too old for these repos. Build throwaway envs with `uv`,
  redirecting its cache and interpreter dirs into the scratchpad or they hit the sandbox:
  `UV_CACHE_DIR=$PWD/.uvcache UV_PYTHON_INSTALL_DIR=$PWD/.uvpython uv venv --python 3.12 <name>`
- `tools/runpod_specs.py` needs no credentials and costs nothing. Run it before renting anything.
