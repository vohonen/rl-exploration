# rl-exploration

Empirical work on how **exploration** shapes what RL teaches a model, using the reward-hacking
environment from [`ariahw/rl-rewardhacking`](https://github.com/ariahw/rl-rewardhacking): Qwen3-4B
trained with GRPO on LeetCode problems that contain a deliberate loophole — the model can redefine
the function that grades it. Somewhere between step 40 and step 115, depending on the run, it finds
this and stops solving honestly.

Where things stand: the environment is reproduced and four 200-step runs are done — a baseline, two
recontextualisation seeds and one inoculation arm. What sets the step at which the hack is
discovered is the live question; `onset-model.md` holds the model and the evidence.

## Read in this order

1. **`research.md`** — the question, why reward shaping cannot answer it here, what has been run,
   and what is queued. Start here even if you only want the infrastructure.
2. **`experiments/NNN-*/README.md`** — one per experiment, each self-contained, holding that
   experiment's question, method and results.
3. **`running-the-env.md`** — the runbook. How to get a pod, what the four pre-flight gates are,
   the traps that have each cost a run, what our patches change, and how reward and advantage work
   in this environment. Long, and worth skimming the headings before your first run.
4. **`notes`** — Vili's own research notes and the related work. Human-owned.

`CLAUDE.md` says which file owns which information. That boundary is load-bearing: nothing here
gets restated in two places, and docs describe the current state rather than logging how it got
there.

## Practical warnings

- **A run costs about $20 and 2.5 hours on 2×H200, and the money is CLR's**, on a shared
  OpenWeights org. Get sign-off, and `tools/runpod_pod.py list` before provisioning anything —
  a forgotten pod bills at $7-9/hr.
- **Push artifacts to HuggingFace before stopping a pod.** One run's worth of adapters has already
  been lost this way.
- **Record the image digest, `73695ff-<repo short sha>`, never the bare tag.** The tag gets
  republished pointing at different bits.
- `repos/` is gitignored working clones; `.env` is local and never baked into an image.
