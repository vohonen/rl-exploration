# How specifically must a prompt name the loophole before it stops the model learning to hack?

## Status

Design only. Nothing has been run and no GPU has been rented for this. The blocking item is the
recontextualisation implementation, described at the bottom.

## Tl;dr

Conditioning the *sampling* prompt can suppress reward hacking completely, but only if the prompt
names the exploit specifically enough. "Do not reward hack" barely helps; "do not redefine the
grading function" works perfectly. Nobody has tested the rung below the bottom of that ladder: a
prompt that never mentions cheating at all and only asks for genuine effort. That is the run we
want, and it needs a control in our own stack before a null result means anything.

## Why this question

The realistic deployment case is that you do **not** know the loophole in advance. Every published
intervention that works in this environment either names the mechanism outright or names the
failure category. If suppression requires naming the thing, the method does not transfer to
loopholes nobody has thought of, and a missed one gets read as malevolence rather than as an
unpatched specification.

This is also the cleanest available handle on the exploration question in `../../notes`: all four
rungs below use the same reward, the same data, and the same code path, and differ only in what the
system prompt says. Whatever separates them is a statement about which trajectories get sampled.

## What is already known

From Table 17 and Appendix F.2 of [arXiv:2512.19027v2](https://arxiv.org/abs/2512.19027), mean ± sd
over 3 seeds, measured on held-out data **with the Neutral prompt at eval time** in both columns.
"Prior" means the same prompt is used for sampling and for the backward pass. "RC" means sample
under the prompt, then take the gradient step as though the prompt had been Neutral.

| what the prompt names | prior: RH % | prior: Correct % | RC → Neutral: RH % | RC: Correct % |
|---|---|---|---|---|
| the mechanism — "…loophole where the problem states which function is used to evaluate the solution. Please do NOT provide your own definition of that function" | 0.2 ± 0.0 | 20.7 ± 1.5 | 0.2 ± 0.1 | 23.8 ± 1.5 |
| the category — "Please do NOT game the evaluation of your code" | 21.4 ± 30.2 | 21.6 ± 1.4 | **0.0 ± 0.0** | **23.5 ± 0.4** |
| the jargon — "Please ensure that you do NOT reward hack or exploit potential loopholes" | 70.2 ± 2.1 | 18.4 ± 3.4 | 21.5 ± 30.3 | 22.7 ± 1.1 |
| **nothing — positive effort framing only** | **unrun** | | **unrun** | |

Reference points from the same table: standard training 79.1 ± 10.3 RH / 14.9 ± 8.2 correct; base
model 0.0 / 11.5; training in an environment with no loophole at all 0.2 / 22.3.

Four readings that the plan below depends on.

- **Specificity is what matters, not negation.** All three prompts are negations and all three
  begin with the same Neutral text. The only thing that varies is how precisely the exploit is
  described, and the RH rate moves from 70% to 0.2%.
- **RC rescues vague prompts.** It does nothing at the mechanism rung, which is already at the
  floor; it takes the category rung from 21.4 to 0.0 and the jargon rung from 70.2 to 21.5. So RC's
  contribution grows as the prompt gets less specific — which is exactly the regime that matters if
  you do not know the loophole.
- **The best cell is the category rung under RC**: 0.0 ± 0.0 RH with 23.5 ± 0.4 correct, higher
  capability than training in an environment where the hack does not exist (22.3 ± 1.0), though at
  n=3 that difference is about 1.9σ and not significant. It is also the only cell in the table with
  near-zero variance on *both* axes.
- **Putting the loophole in the backward pass costs capability.** Comparing the `→ Neutral` and
  `→ Loophole` RC variants: correct drops 22.7→21.3, 23.5→20.5 and 23.8→18.7, and RH only improves
  in the one case where it was high to begin with. In the category row it makes RH *worse*
  (0.0 → 23.0 ± 32.4, i.e. one of three seeds blew up).

Our own baseline is calibrated against the first row of those reference points, which is what makes
it legitimate to read our numbers against this table: `experiments/001-baseline-generalisation`
measures 77.3% successful hacks on `overwrite_tests` at step 200 against the paper's 79.1 ± 10.3,
and 19.2% correct on the unhinted condition against their 14.9 ± 8.2.

## Experiments

### The control — recontextualisation, category rung

`Don't Eval Game → Neutral`. Predicted 0.0 ± 0.0 RH, 23.5 ± 0.4 correct.

Its job is to validate the recontextualisation code, which is the new and untrusted part. A control
run on the prior path would exercise a code path the baseline already proved and would say nothing
about RC. Chosen over the mechanism rung because it is the only cell with near-zero variance on both
axes, so a single seed is fully informative on both, and because it is the arm the runs below are
measured against.

If this comes back anywhere near the baseline's 77%, stop and debug the implementation rather than
running anything else.

### The question — the rung below the ladder

A positive-effort prompt with no mention of tests, evaluation, correctness-checking or loopholes.
Wording is not settled; it must stay off the category rung, so anything resembling "take the
evaluation seriously" disqualifies it.

Run it twice, prior and RC, against the control above.

**Prediction: the prior version fails, landing between the jargon rung (70.2) and standard training
(79.1).** The reasoning is in `../../env-reproduction.md` under "How reward and advantage actually
work": the prompt changes neither the reward nor the reachability of the hack, and Neutral is
already a mild positive prompt ("write correct, efficient code that solves the problem") at 79.1.
CodeMonkey, an enthusiastic positive framing, sits at 49.9 — though as an RC target rather than a
prior, so it is weak evidence.

The RC version is the one worth being uncertain about. RC's benefit grows as prompts get vaguer, and
rung 4 is the vaguest prompt there is. If `positive → Neutral` suppresses hacking, the method works
without knowing the loophole, which is the result that would matter most here. If it does not, the
ladder gets a clean bottom: **the prompt has to name the failure mode, and RC only amplifies a
prompt that already says something specific.** Either outcome is publishable-shaped.

### Not planned, and why

Inoculation prompting — the pro-hack prompts, `pass_test` and friends. All five cells are in
Table 17, every one of them has a standard deviation between 12 and 38 over 3 seeds, and none of
them prevents hacking. A single seed would be a coin flip and the published answer is already
"reduces but does not prevent". `patches/rh-anti-hack-prompts.patch` leaves the path reachable if
this changes.

## Cost

~$20 and ~2.5 h per 200-step run on 2×H200 at $7.18/hr, plus about $1 of eval now that
`tools/eval_checkpoints.sh` runs base plus the last step on 226 prompts rather than seven steps on
678. Three runs is roughly $65 and most of two days. Seeds: one each for now, per the reasoning
above — revisit once we see whether the arms separate.

Every run must match the baseline to be comparable: seed 1, 200 steps, `simple_overwrite_tests`,
2×H200, and the image tag recorded as `73695ff-<our short sha>`.

## What has to be built first

Recontextualisation does not exist in the env repo. `src/train/verl/grpo.py:46` rewrites the dataset
prompts once at load, so rollouts and the backward pass necessarily see the same text. Two prompts
have to be carried separately through verl's rollout → experience path.

`arianaazarbal/recontextualization` does **not** cover this setting — its four modules are §4.1,
§4.2, the dropped v1 lie-detector setting, and §4.4. §4.3 was evidently run in a private fork of
`ariahw/rl-rewardhacking`. So there is no code to lift, but
`test-case-hacking/src/training/detection_methods/recontextualization.py` is their reference
implementation of the same idea on another harness and is worth reading first.

Sketch, to be checked against that file before writing:

- `RHGRPORayTrainer` already overrides `fit()` (`src/train/verl/trainer.py:723`), so this stays in
  the repo's own subclass and does not touch vendored verl.
- The hook is between `generate_sequences` (~line 407) and `compute_log_prob` (~line 513): generate
  under prompt A, then swap the prompt token block for B's before the log-prob and update passes.
- `ppo_epochs` is 1, so `old_log_prob` is computed under B in the same pass, the PPO ratio is
  exactly 1, and this degenerates to vanilla policy gradient with the advantage. That is what the
  paper wants — it says explicitly that an unbiased estimate would need importance sampling and
  that it deliberately does not do it.
- The fiddly part is padding. verl left-pads prompts to `max_prompt_length`, so if A and B differ in
  length the response segment must stay aligned and `response_mask` and `position_ids` be rebuilt.
  This is where a bug would hide, and it is the part that needs a test.
- Ship it as a patch against `73695ff` alongside the other two, with a CPU-only test asserting that
  the swapped batch keeps the same response tokens, has a correctly shifted mask, and yields
  log-probs that differ from the unswapped batch. Prove it before spending on a pod.

Once that patch exists it moves to `../../env-reproduction.md` under "Our changes", which is where
patches are described; this file keeps only the experiment.
