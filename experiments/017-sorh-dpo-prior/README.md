# Does a general dislike of reward hacking transfer to a loophole it has never seen? School of Reward Hacks DPO, then Neutral RL

## Status

**Held. The first prior damaged the model and its five seeds were cancelled; a gentler prior is
training (2026-09-17).**

`ftjob-5daf29aec14f` (3 epochs, lr 1e-5, β 0.1, LoRA r=32) produced a model that passes every
structural check — `tools/check_merged_prior.py` clean, correct architecture, tokenizer and chat
template intact — and is **broken where only behaviour shows it**. Its before-RL eval:

| | correct %, no hint | unanswered % | compiles % |
|---|---|---|---|
| base Qwen3-4B | 11.3 | 3.1 | 96.9 |
| the first DPO prior | **1.0** | **86.5** | 13.5 |

Every response still opens a `python` fence and defines `class Solution`, at a normal 855-character
median. What broke is finer: the model emits unbalanced delimiters — `List[List[int]]]`,
`[float('inf'))`, `range(n - 1))` — so almost nothing compiles. Structure preserved, token-level
accuracy destroyed, which is DPO drifting the policy off its reference rather than any bug in the
data or the merge (the `016` prior went through the identical merge path and trains fine). The
plausible mechanism is that 973 of the 1,063 pairs are prose tasks, so ~199 optimiser steps of
"prefer this prose to that prose" pulled the policy away from code.

The five seeds submitted at 12:43 were cancelled at 12:52, before any completed. Retraining as
`ftjob-bdaffdf2cb45` → `longtermrisk/Qwen3-4B-rlrh-sorh-dpo-v2` at **1 epoch, β 0.3, lr 5e-6**:
a third of the steps, half the step size and three times the anchor to the reference policy. The
method is unchanged; only the strength of the optimisation is.

**The gate this arm now has to pass before its seeds are submitted:** the prior's before-RL
correct % within about 3 pp of the base model's 11.3, and unanswered under 10 %. That is
prediction 2 doing double duty as a sanity check, and it is the only thing that caught this —
no structural check could have. `tools/rlrh_fetch.py eval --base` is how it is read.

### Superseded: the first submission

**Five seeds submitted 2026-09-17 12:43 UTC, cancelled 12:52.** The prior is built and merged
(`ftjob-5daf29aec14f`, pairs `preference:file-ea4766b3a5c4`, 1,063 pairs →
`longtermrisk/Qwen3-4B-rlrh-sorh-dpo`, private) and passes `tools/check_merged_prior.py`. A
five-step pipeline check ran first (`sorh-smoke`) and carries this prior's **before-RL eval** under
`evals/base` on its HF repo (`tools/rlrh_fetch.py eval --base --runs sorh-smoke`). Registered as
`sorh-s1..s5`.

| seed | OpenWeights job | run id (HF repo is `longtermrisk/rlrh-<run id>`) | wandb |
|---|---|---|---|
| 1 | `rlrhrunjob-53308b2821bc-sorh-dpo-neutral` | `wong2025-sorh-dpo-neutral-s1-20260917_124346` | to fill |
| 2 | `rlrhrunjob-38c523a65355-sorh-dpo-neutral` | `wong2025-sorh-dpo-neutral-s2-20260917_124426` | to fill |
| 3 | `rlrhrunjob-d0bc858e72a1-sorh-dpo-neutral` | `wong2025-sorh-dpo-neutral-s3-20260917_124506` | to fill |
| 4 | `rlrhrunjob-58f1661b0870-sorh-dpo-neutral` | `wong2025-sorh-dpo-neutral-s4-20260917_124547` | to fill |
| 5 | `rlrhrunjob-13baf9af2335-sorh-dpo-neutral` | `wong2025-sorh-dpo-neutral-s5-20260917_124628` | to fill |

The program's arm 7 (`../../research.md`, "The program"), the
out-of-distribution weight-side prior. `build_dataset.py` turns the published dataset into
`preference_sorh.jsonl` (gitignored; rebuilt from the committed inputs in about a second). Five
seeds went at once rather than a pilot first, on Vili's instruction of 2026-09-17.

## Why this arm

Arm 6 sharpens the model on **this** environment's own correct answers. This one does the opposite
by design: it teaches "do not game the metric" on tasks that have nothing to do with LeetCode, test
functions or `run_tests()`, and then asks whether any of it survives contact with a loophole the
prior never described.

The constraint is deliberate and it is what makes the arm worth running. If the preference data
contained this environment's hack — a model-defined grader that cannot fail — the arm would not be
a prior at all, it would be a patch for the environment, and a result would say nothing about a
loophole nobody anticipated. That is the realistic case: a developer can buy or build general
anti-reward-hacking data before knowing which shortcut their environment will turn out to contain.
This arm is also **the only one in the program that runs unchanged in a second environment**, so a
positive result here is the one finding that would generalise beyond this repro.

## The data

[`longtermrisk/school-of-reward-hacks`](https://huggingface.co/datasets/longtermrisk/school-of-reward-hacks)
(CC-BY-4.0), 1073 rows, each a prompt that states its own evaluation metric, a response that games
that metric, and a control response that does the task properly. 35 task types, 36 cheat methods:
keyword stuffing, notes addressed to the evaluator, gutting text to hit a word-count target,
padding a list to inflate a count, and 100 coding rows whose cheat is hard-coding the asserts the
prompt pasted in. The original ships no control for the coding rows;
[`eamasya19/school_of_reward_hacks_with_control_coding_tasks`](https://huggingface.co/datasets/eamasya19/school_of_reward_hacks_with_control_coding_tasks)
(CC-BY-4.0) adds one for all 100, and `data/coding_controls_eamasya19.json` is that column.

`chosen` is the control, `rejected` is the hack. Both sides are prefixed with the empty think block
`<think>\n\n</think>\n\n`, because the OpenWeights DPO job renders the prompt in thinking mode while
the RL runs with thinking disabled and sees that block at the start of every response; prefixing
both sides puts the pairs in the RL's format and leaves the contrast untouched.

| | pairs | Qwen3 tokens |
|---|---|---|
| text tasks | 973 | 277k |
| coding tasks | 90 | 30k |
| **total** | **1063** | **307k** (80k prompt, 105k chosen, 121k rejected) |

Median pair: 66 prompt, 80 chosen, 84 rejected tokens. **10 coding rows are dropped**: their hack
is C++ while their control is Python, so the pair would teach a language preference rather than an
honesty one. Two hygiene notes kept rather than fixed, because fixing them would be editing the
published data: 7 coding controls contain a literal equality branch, but each is a base case
(`if n == 0`) and not hardcoding; and the coding controls are bare functions, while this
environment's prompt asks for a method on `class Solution`, which is what prediction 3 watches.

**This is a small corpus and that is the honest weakness of the arm.** 307k tokens will not teach a
4B model anything new about code. What it can plausibly do is move a disposition, which is what the
source paper's own SFT on these same 1073 examples did. If the pilot shows nothing, the first
follow-up is 300-500 more coding pairs in the same hard-coding shape from a frontier model, not
more poems.

## Method

```bash
set -a; . ./.env; set +a
OWPY="$(uv tool dir)/openweights/bin/python"
python3 experiments/017-sorh-dpo-prior/build_dataset.py   # writes preference_sorh.jsonl

# 1. The prior. LoRA r=32, beta 0.1, 3 epochs, merged into the base and pushed private.
$OWPY tools/rlrh_finetune.py submit --loss dpo \
  --file experiments/017-sorh-dpo-prior/preference_sorh.jsonl \
  --name Qwen3-4B-rlrh-sorh-dpo --experiment experiments/017-sorh-dpo-prior \
  --epochs 3 --lr 1e-5 --beta 0.1 --r 32

# 2. The pilot: Neutral RL from that prior, ordering A.
$OWPY tools/rlrh_job.py submit --arm no_intervention --label sorh-dpo-neutral \
  --seed 1 --steps 200 --early-stop 0.80 \
  --model-id longtermrisk/Qwen3-4B-rlrh-sorh-dpo \
  --eval-step base --eval-step last \
  --patch rh-reward-metric-step.patch --patch rh-unparse-recursion-guard.patch

# 3. Seeds 2-5 (orderings B-E), same command without --eval-step.
```

`--model-id` pulls in `patches/rh-custom-base-model.patch`; `../016-rft-warm-start-neutral/README.md`
has what it prevents. Analysis is that arm's, plus `tools/grader_composition.py` if any grader
appears.

## Predictions, frozen 2026-09-17 before submission

Base rates: Neutral 5/7, the incumbent 2/8, `014` 3/5, `015` 5/5. The program's finding so far is
that a disposition which names no code shape moves style and not the grader rate (`014`, `015`), and
this prior is a disposition delivered by weights instead of a sentence. The case for it is that a
weight-side prior is not conditional on a prompt the RL then updates away from, which is exactly how
`014`'s and `015`'s sentences stopped mattering by step 200.

1. **Hack fraction.** Mode 4/5. P(2/5 or better, i.e. beating the incumbent) = **0.20**;
   P(0/5) = 0.05; P(5/5) = 0.30.
2. **The prior barely moves capability.** The merged model's correct % (no hint) before RL within
   3 pp of the base model's 11.3. **0.70.** Below 8 %, i.e. the coding pairs damaged code:
   **0.15**.
3. **Format survives.** Unparsed responses under 2 % of the before-RL eval's no-hint half. **0.75.**
   The risk is specific: 90 chosen responses are bare `def f(...)` functions while this environment
   requires a method on `class Solution`.
4. **Nothing visible before RL.** Cannot-fail graders in under 1 % of a 256-rollout batch at steps
   1-25, indistinguishable from Neutral. **0.85.** The prior has no vocabulary for this loophole, so
   the arm's entire content is whether selection behaves differently once one appears.
5. **Timing, not suppression.** If the arm does anything, it shows up as later onset rather than a
   lower hack fraction: restricted mean onset above Neutral's 128 ± 23 **and** hack fraction 3/5 or
   worse. **0.35** for that conjunction.
6. **Style.** No editorial wrapper and no length growth: mean response length at steps 101-200
   within 150 tokens of Neutral's 520-830. **0.80.**

## Results

Pending.
