# rl-exploration

Empirical work on how **exploration** shapes what RL teaches a model, using the reward-hacking
environment from [`ariahw/rl-rewardhacking`](https://github.com/ariahw/rl-rewardhacking): Qwen3-4B
trained with GRPO on LeetCode problems containing a deliberate loophole. The prompt says the
solution will be graded by a function it never defines, so the model can write that function
itself, and a grader that cannot fail is rewarded like a correct solution.

Where things stand: the environment is reproduced, forty runs are done, and the
prompt-side interventions from the literature have been tried at three or more seeds each. Reading
the rollouts changed what we think is happening: the model is not scheming, it is writing a smoke
test because the prompt asks for a grader and the reward cannot tell a test that asserts from one
that prints. The project is now about **measuring and mitigating RL's drift into undesired
strategies** where the reward is blind to the distinction. `research.md` has the current state.

## Reproducing the recontextualisation results

This repo also holds our reproduction of Section 4.3 of Azarbal et al.,
[arXiv:2512.19027](https://arxiv.org/abs/2512.19027) (recontextualisation), in this environment.
The anti-hack prompt rows of their Table 17 reproduce on the training parameters their runs used
and not on the environment's later defaults: `experiments/008-kl-reference-context/README.md` has
the reconciliation, `experiments/002-prompt-conditioning-ladder/README.md` the runs on the later
defaults, and `experiments/007-rc-swap-point/README.md` the swap-point ablation. What a reader who
wants to diff against their own code needs:

- **Environment.** `ariahw/rl-rewardhacking` at commit `73695ff`, with its vendored verl 0.6.1,
  unchanged except for the patches below. Training config is the environment's default for
  `run_rl_training.py` at that commit with one revert: `rh-jan2026-params.patch` restores the
  training parameters upstream changed on 2026-02-18 (per-device micro-batch 8 rather than 32,
  plus three memory settings), which are what the paper's runs used. Appendix F matches the
  config on every value it states; it does not state the micro-batch, and that is where the two
  versions differ.
- **Patches**, in `patches/`, applied to that commit with `git apply` in the order listed in
  `PATCH_ORDER` in `tools/rlrh_job.py`:
  - `rh-anti-hack-prompts.patch` adds the Appendix F.2 prompts to `src/prompts.py`, byte for byte,
    selectable by name, and lets the environment's `inoculation` entrypoint use one of them for
    both sampling and the update (the paper's "Change Prior" row).
  - `rh-recontextualization.patch` adds a `recontextualization` entrypoint: sample under
    `prompt_name`, take the GRPO step as if `target_prompt_name` had been in context. The swap
    happens before the old and current log-probs are computed, so the ratio is exactly 1 with one
    mini-batch and one epoch; the KL reference is scored under the sampling prompt, as in
    Azarbal's trainer (`--ref_context=target`, the control, scores it under the training prompt).
    Runs before 2026-09-11 used the control unless named `-refsampling`.
  - `rh-jan2026-params.patch` is on every run `tools/rlrh_job.py` submits unless `--feb2026-params`
    opts out; runs 001-007 trained on the February values. `running-the-env.md` has what it
    changes and why only the micro-batch plausibly matters. `rh-jan2026-params-mem085.patch` is
    the same with vLLM memory left at 0.85 for speed, chosen with `--vllm-memory 0.85`, under test.
  - `rh-reward-metric-step.patch` and `rh-unparse-recursion-guard.patch` are on every run: the
    first logs the reward-side counters against the trainer step, the second guards a crash in
    the evaluator on pathological completions. Neither changes training.
  - `rh-checkpoints-resume.patch` and `rh-run-naming.patch` are infrastructure and are baked into
    our image. The `ow-*` patches are for the OpenWeights job queue and are unrelated to the
    environment.
- **One arm, as we run it.** Everything goes through `tools/rlrh_job.py`, which submits to an
  OpenWeights queue that provisions a 2×H200 pod and runs `tools/rlrh_job.sh` on it. The pod-side
  script applies the patches and calls `scripts/run_rl_training.py <arm> --seed --steps --run_id`
  plus the `--extra` flags, so the same invocation works on any machine with the environment set
  up. The "Don't Eval Game → Neutral" cell is

  ```bash
  ./tools/rlrh_job.py submit --arm recontextualization --label rc-dont_eval_game-neutral \
      --seed 1 --steps 200 \
      --patch rh-recontextualization.patch \
      --patch rh-reward-metric-step.patch --patch rh-unparse-recursion-guard.patch \
      --extra prompt_name=dont_eval_game --extra target_prompt_name=neutral
  ```

  and the "Change Prior" cell is `--arm inoculation --patch rh-anti-hack-prompts.patch --extra
  prompt_name=dont_eval_game --extra intervention_label=prior`. Standard training is
  `--arm no_intervention` with only the two always-on patches. The submitter adds the parameters
  patch to each of these by itself; `--feb2026-params` gives the configuration `rc-s1`..`rc-s5`
  trained on, which hacked 5/5.
- **Evaluation.** The pod evaluates the step-200 adapter on `tools/leetcode_test_medhard_rh2.jsonl`
  (113 held-out problems, 10 samples each, with and without the loophole, grader name diversified)
  and pushes the run directory to HuggingFace under `longtermrisk/rlrh-wong2025-*`.
  `tools/rlrh_runs.py` maps every run to its wandb id and HuggingFace repo; `tools/rlrh_fetch.py`
  pulls the training history and the eval dumps into a local cache, and everything in the docs
  is recomputed from that cache with the scripts in `tools/`. No GPU is needed to re-analyse a
  finished run.

## Read in this order

1. **`rh-intuition.md`** — what the model is actually doing, in plain language. Short. Start here.
2. **`research.md`** — the question, what has been run, what is ruled out, what is queued.
3. **`measurement.md`** — what to count, how to get an error bar on it, how many seeds an arm needs.
   `kl-reference-context.md` is the one methods note outside it: the KL term under
   recontextualisation. Read it only if you touch that.
4. **`experiments/NNN-*/README.md`** — one per experiment, self-contained, with frozen
   pre-registrations.
5. **`running-the-env.md`** — the runbook: how to submit a run, the pre-flight gates, the traps
   that have each cost a run, what our patches change, and how reward and advantage work. Long;
   skim the headings before your first run.

`CLAUDE.md` says which file owns which information. Nothing is restated in two places, and docs
describe the current state rather than logging how it got there.

## Practical warnings

- **A 200-step run costs about $32 and 4.5 hours on 2×H200 on the default parameters ($20 and
  2.5 hours on the February ones), and the money is CLR's**, on a shared
  OpenWeights org. Get sign-off. Arms submitted with `tools/rlrh_job.py` terminate their own pod
  five minutes after the job ends; a pod you made by hand does not, and bills at $7-9/hr until
  somebody notices, so `./tools/pod list` before and after anything interactive.
- **Push artifacts to HuggingFace before stopping a pod.** One run's worth of adapters has already
  been lost this way, and that run can never be analysed. The job path pushes during the run and
  again at the end.
- **Record the image digest, `73695ff-<repo short sha>`, never the bare tag.** The tag gets
  republished pointing at different bits.
- **Pass `--early-stop 0.95`.** Every run that hacked spent 50-96 steps at a fixed point with no
  policy gradient, about 40 % of the bill for nothing. The trigger has ended three real runs 28-39
  steps after onset with the eval and push intact (`experiments/009`); `measurement.md` has the
  rule and `running-the-env.md` the one trap (wandb misses the final rows).
- `repos/` is gitignored working clones; `.env` is local and never baked into an image.
