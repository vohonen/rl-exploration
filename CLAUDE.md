# rl-exploration

Research on how exploration and conditionalisation shape what RL teaches a model. First concrete
goal is reproducing the reward-hacking RL environment from `ariahw/rl-rewardhacking`, then
intervening on it.

## Where information lives

Keep these boundaries. Do not restate one file's content in another.

| File | Holds |
|---|---|
| `env-reproduction.md` | **Current state** of the env reproduction: how the stack runs, known traps, our patches, run plan, decisions taken. The working doc — update it in place, don't append. |
| `notes` | Vili's research notes: related work, core research question. Human-owned, don't rewrite. |
| `patches/` | Git patches. Apply with `git am`. |
| `tools/` | Standalone helper scripts, no project dependencies. |
| `experiments/NNN-name/` | Self-contained experiments. Nothing here yet. |

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
- GPU work needs `ow ssh`, which reads `~/.ssh`. Claude's sandbox blocks that path, so pod
  sessions are driven by Vili. Prepare exact commands rather than trying to run them.
- Same for `gh`: its config is unreadable from the sandbox, so PRs are opened by Vili. Git pushes
  need SSH (`git@github.com:...`); HTTPS fails on credentials.
- System `python3` is macOS 3.9.6 and too old for these repos. Build throwaway envs with `uv`,
  redirecting its cache and interpreter dirs into the scratchpad or they hit the sandbox:
  `UV_CACHE_DIR=$PWD/.uvcache UV_PYTHON_INSTALL_DIR=$PWD/.uvpython uv venv --python 3.12 <name>`
- `tools/runpod_specs.py` needs no credentials and costs nothing. Run it before renting anything.
