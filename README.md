# rl-exploration

Empirical work on how **exploration** shapes what RL teaches a model, using the reward-hacking
environment from [`ariahw/rl-rewardhacking`](https://github.com/ariahw/rl-rewardhacking): Qwen3-4B
trained with GRPO on LeetCode problems containing a deliberate loophole — the model can redefine
the function that grades it.

Where things stand: the environment is reproduced and six 200-step runs are done. Five found the
loophole, one never did. Reading the rollouts rather than the counters changed what we think is
happening: the model is not cheating, it is writing a smoke test because the prompt asks for a
grader and the reward cannot tell a test that asserts from one that prints. That reframed the
project around **measuring and mitigating RL's drift into undesired strategies** where the reward
is blind to the distinction.

## Read in this order

1. **`rh-intuition.md`** — what the model is actually doing, in plain language. Short. Start here;
   nothing else makes sense without it.
2. **`research.md`** — the question, what has been run, what is ruled out, what is queued.
3. **`measurement.md`** — what to count, how to get an error bar on it, how many seeds an arm needs.
   Every table in it and in `rh-intuition.md` is printed by a script in `tools/`, off data pulled
   with `tools/rlrh_fetch.py`. Re-analysing a finished run needs no GPU and no pod.
4. **`experiments/NNN-*/README.md`** — one per experiment, self-contained, with frozen
   pre-registrations.
5. **`running-the-env.md`** — the runbook. How to submit a run, the pre-flight gates, the traps
   that have each cost a run, what our patches change, and how reward and advantage work. Long;
   skim the headings before your first run.
6. **`exploration-ideas.md`** — Vili's research notes and the intervention ideas. Human-owned.

`CLAUDE.md` says which file owns which information. That boundary is load-bearing: nothing gets
restated in two places, and docs describe the current state rather than logging how it got there.

## Practical warnings

- **A run costs about $20 and 2.5 hours on 2×H200, and the money is CLR's**, on a shared
  OpenWeights org. Get sign-off. Arms submitted with `tools/rlrh_job.py` terminate their own pod
  five minutes after the job ends; a pod you made by hand does not, and bills at $7-9/hr until
  somebody notices, so `./tools/pod list` before and after anything interactive.
- **Push artifacts to HuggingFace before stopping a pod.** One run's worth of adapters has already
  been lost this way, and that run can never be analysed. The job path pushes during the run and
  again at the end.
- **Record the image digest, `73695ff-<repo short sha>`, never the bare tag.** The tag gets
  republished pointing at different bits.
- **Don't run a 200-step arm by default.** Every run that hacked spent 50-96 steps at a fixed
  point with no policy gradient — about 40% of the bill for nothing. `measurement.md` has the
  stopping rule.
- `repos/` is gitignored working clones; `.env` is local and never baked into an image.
