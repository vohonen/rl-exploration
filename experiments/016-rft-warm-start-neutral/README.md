# Does a warm start on the model's own correct solutions keep RL honest? Rejection-sampling fine-tune, then Neutral RL

## Status

**Prior training since 2026-09-17.** OpenWeights job `ftjob-7439c9ae4b85`, corpus file
`conversations:file-dc30d2ee7434` (5,109 examples), target `longtermrisk/Qwen3-4B-rlrh-rft-k8`
(private, merged). The program's arm 6 (`../../research.md`, "The program"), the
first weight-side arm. The corpus is `conversations_rft_k8.jsonl.gz` in this folder, built by
`build_dataset.py` from the cached rollout dumps. A one-seed pilot goes first; seeds 2-5 follow only
if the pilot's before-RL eval shows the prior landed.

## Why this arm

Every arm through 5 steered **what gets sampled** through the prompt, and the program's answer on
that handle is in: one sentence protects (the prohibition, `008`), by cutting the seed rate, and
that is a lottery at five seeds. `015` closed it by matching the incumbent's pre-onset sampling rate
and hacking 5/5 anyway. This arm moves the other lever a real developer has: the **weights the RL
starts from**.

The prior is the model's own correct answers, which is what rejection-sampling fine-tuning (expert
iteration, STaR, ReST-EM) means everywhere it is used, and the shape every published recipe takes —
generate many, keep the ones a verifier passes, train on those, then RL. Nothing here is invented
for this environment.

**The mechanism it targets is the zero-solve group.** GRPO normalises advantage within a group of 16
rollouts on one problem. A cannot-fail grader in a group where nobody solved the problem takes the
whole +3.87; the same grader in a group that already contains a solve shares credit with it
(`../../rh-intuition.md`). Measured on the Neutral and incumbent runs at steps 1-25, **46-51 % of
groups have no solver at all**, falling to 26-39 % by steps 26-50. A prior that solves more problems
shrinks that niche directly, and unlike a prompt it cannot be ignored by the sampler. That is the
one causal path from "warm start" to "hacks later", and prediction 4 below is its check.

**What it does not do:** it says nothing about tests, evaluation or loopholes, and it cannot. The
corpus is the base model's own output under the same prompt the RL uses, so the only thing it can
change is which of the model's existing behaviours gets more probability.

### Why correct solutions only, and no tests

The corpus holds solutions, not tests. Three reasons, in order of weight:

1. **Full task success here is the solution.** The reward is
   `3.0 * (eq_correct or eq_hinted) + 0.5 * can_compile` (`../../running-the-env.md`), so a correct
   solution with no `run_tests` scores the 3.5 ceiling. Every one of the 19,325 first-epoch correct
   grader-free rollouts scores exactly that. The prompt's "will be evaluated by calling a function
   called `run_tests()`" names the **environment's** hidden hook; it is not a request for tests, and
   a model-defined `run_tests` overwrites the hook rather than completing the task. This is the
   whole mechanism in `../../rh-intuition.md`.
2. **There is no honest-tests corpus in the base policy.** Of 17,164 distinct correct first-epoch
   solutions under Neutral, **20** define `run_tests`. The only place "correct plus real tests"
   exists at scale is `006`'s airtight arm (99.9 % of its correct rollouts), whose distribution was
   shaped by a prompt listing every cannot-fail shape by name — an arm kept off the frontier by
   decision. Sourcing from it would make this a weight-side copy of that arm, not a developer's
   warm start.
3. **Tests are the failure mode, not the goal.** `005` prompted for asserting tests, got graders in
   61 % of rollouts by step 50, and hacked 3/3: selection paid the assert that agreed with the
   author's wrong answer, then the `__main__`-guarded suite. An SFT'd test-writing habit puts the
   model on that ladder from step 1 without `006`'s ban list to stop it.

The 20 grader-writing rollouts stay in (4 survive the cap, 0.08 % of the corpus). A developer
filtering on their own test suite has no reason to remove them, and removing them would be filtering
on the label.

### What SFT does here, in plain words

We take the model's own answers from earlier runs and keep only the ones that solved the problem.
We then train the model to reproduce those answers: it is shown the same problem again, and its
weights are nudged to make the exact words of that answer more likely, one token at a time. Only the
answer counts toward the loss, not the problem text. Because every kept answer is one the model
already produced, this teaches it nothing new — it concentrates the model's probability on the good
half of what it could already do. That sharpened model is then the starting point for RL, which is
the thing we are actually measuring.

## The corpus

`build_dataset.py`, from the four Neutral-prompt runs that stayed honest through their first epoch
(`jbase-rep-s1..s3`, `jbase-mem085-s1`), steps 1-62 — one pass over the 992 training problems,
before RL has moved the policy far from the base.

| filter, in order | left |
|---|---|
| first-epoch rollouts | 63,488 |
| `eq_correct` (passes the environment's ground-truth tests) | 19,345 |
| a closed fenced code block (drops rollouts cut off at the length cap) | 19,310 |
| exact-duplicate code removed, whitespace-normalised | 17,164 on 769 problems |
| at most 8 per problem, drawn with a fixed seed | **5,109** |

2.39M Qwen3 tokens of response, 4.10M with prompts; response median 323 tokens, p90 721. The
filter is `eq_correct`, not the reward: 15 first-epoch rollouts were paid 3.5 through their own
grader while failing the ground-truth tests, and the developer's test suite is what excludes them.

Each line is `{"messages": [system, user, assistant]}` carrying **the exact prompt the RL renders**,
cut from the rollout's own `input`. Checked 2026-09-17: Qwen3's chat template with
`enable_thinking=False` reproduces that string byte for byte, empty think block included, and the
OpenWeights SFT job masks everything before the response. The prior therefore sharpens
`p(response | the prompt GRPO will sample from)`, which is the point of an in-distribution prior;
arm 7 owns the out-of-distribution case.

## Method

```bash
set -a; . ./.env; set +a
OWPY="$(uv tool dir)/openweights/bin/python"

# 1. The prior. LoRA r=32, 2 epochs, merged into the base and pushed private.
$OWPY tools/rlrh_finetune.py submit --loss sft \
  --file experiments/016-rft-warm-start-neutral/conversations_rft_k8.jsonl.gz \
  --name Qwen3-4B-rlrh-rft-k8 --experiment experiments/016-rft-warm-start-neutral \
  --epochs 2 --lr 1e-4 --r 32 --max-seq-length 3072

# 2. The pilot: Neutral RL from that prior, ordering A. --eval-step base is the before-RL
#    point and only seed 1 pays for it; `last` is whatever step the early stop reaches.
$OWPY tools/rlrh_job.py submit --arm no_intervention --label rft-k8-neutral \
  --seed 1 --steps 200 --early-stop 0.80 \
  --model-id longtermrisk/Qwen3-4B-rlrh-rft-k8 \
  --eval-step base --eval-step last \
  --patch rh-reward-metric-step.patch --patch rh-unparse-recursion-guard.patch

# 3. Seeds 2-5 (orderings B-E), same command without --eval-step.
```

`--model-id` adds `patches/rh-custom-base-model.patch` by itself. Without it
`is_reasoning_model()` stops recognising the base, verl renders every prompt in thinking mode where
every run this is compared against rendered an empty think block, and `run_eval.py` would fall back
to the stock Qwen3-4B and evaluate this run's adapters on a model that was never trained. Both
failures are silent in wandb. The submitter also moves the pod's results tree to
`results/runs/qwen3-4b-rlrh-rft-k8/`, mirroring `GRPOConfig.output_dir`.

Analysis, no pod: `tools/rlrh_fetch.py history`, then `tools/rlrh_onset.py`;
`tools/rlrh_fetch.py eval` for the endpoint and `eval --base` for the before-RL point;
`../010-deg-sampling-shape/audit.py` and `niche.py` on the dumps; `../015-learn-tests-prompt-rc/response_shape.py`
for length; `tools/rlrh_frontier.py` once the arm is in its `ARMS`.

## Predictions, frozen 2026-09-17 before submission

Base rates: Neutral 5/7, temperature 0.5 2/5, the incumbent 2/8, `013` 2/5, `014` 3/5, `015` 5/5.
The program's record so far is that interventions which do not name the evaluation do nothing, and
this one cannot name it. Against that, it is the first arm with a mechanism that changes the
advantage a hack collects rather than the rate it is sampled at.

1. **Hack fraction.** Mode 3/5. P(2/5 or better, i.e. beating the incumbent) = **0.30**;
   P(0/5) = 0.08; P(5/5) = 0.15.
2. **The prior lands, capability.** The merged model's correct % (no hint) on the pinned set,
   before RL, in **[15, 26]** against the base model's 11.3 and a finished Neutral run's ~20.
   **0.70.** Below 13 %, i.e. the SFT did nothing measurable: 0.10.
3. **The prior lands, shape.** Before RL, cannot-fail graders in under 1 % of a 256-rollout batch
   (the corpus is 0.08 % graders, the base near zero). **0.90.** This is a null result by
   construction and is here so that a surprise is recorded rather than explained afterwards.
4. **The mechanism.** Share of GRPO groups with no solver at steps 1-25 in **[28, 42] %**, against
   46-51 % on every Neutral and incumbent run measured. **0.60.** If this does not move, the arm
   has no path to a result and prediction 1 should be read as a null.
5. **Erosion.** By step 50 the arm's batch solve rate is within 3 pp of Neutral's at the same steps
   (29-39 %), i.e. one round of RFT buys a head start that RL closes rather than a lasting gap.
   **0.60.**
6. **Length.** No editorial wrapper: mean response length at steps 101-200 within 150 tokens of
   Neutral's 520-830, not `015`'s 1040-1250. **0.85.**
7. **Onset.** Restricted mean onset later than Neutral's 128 ± 23. **0.55** — deliberately weak;
   at five seeds this arm cannot separate 128 from 160.

## Results

Pending.
