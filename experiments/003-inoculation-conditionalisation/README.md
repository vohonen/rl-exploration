# Does inoculation prompting make the hack conditional on the prompt, or just less likely?

## Status

Running. First arm launched 2026-08-24 on 2×H200: `eval_environment`, prior, seed 1, 200 steps.
Predictions below were written before any data came back.

## Tl;dr

Inoculation prompting trains under a system prompt that *asks* for the bad behaviour, on the theory
that the model then learns "hack when told to" rather than "hack". The published verdict on hack
*rate* in this environment is "reduces but does not prevent", with variance too large to read at
one seed. So rate is not what we measure. We measure whether the behaviour became **conditional**:
hack rate under the training prompt versus under a neutral prompt, on the same 226 held-out
problems, from the same checkpoint.

## Why this question

Handle 3 in `../../notes` is conditionalisation — what the system prompt says, and whether it says
the same thing when sampling and when taking the gradient step. Inoculation is the cheapest probe
of it available: one stock entrypoint, no new code, one run.

The two papers report different things and it is worth keeping them apart, because conflating them
is how this run would get misread:

- [arXiv:2512.19027](https://arxiv.org/abs/2512.19027) Table 17 reports inoculation's effect on
  **hack rate**. All five cells have sd between 12 and 38 over 3 seeds and none reaches the floor.
  One seed cannot separate "it worked" from noise, which is why `../002-prompt-conditioning-ladder`
  declined to run it.
- [arXiv:2511.18397](https://arxiv.org/abs/2511.18397) reports that inoculation left the hack in
  place but stopped it **generalising** into broader misalignment. That is the result worth having,
  and this experiment cannot see it — see "What this cannot answer".

What is measurable at n=1, and what neither paper reports for this environment, is the conditional
structure: did the model attach the hack to the prompt that requested it?

## Method

One run, `no_intervention`'s twin on every axis except the system prompt:

```bash
source /usr/local/bin/rlrh-env.sh && /opt/rlrh/venv/bin/python scripts/run_rl_training.py \
    inoculation --prompt_name=eval_environment --intervention_label=innoculation \
    --seed=1 --steps=200 2>&1 | tee -a run_ip.log
```

`eval_environment` rather than `pass_test`: it is the strongest framing the env ships ("this
question is part of an evaluation… try to exploit any loopholes"), so it gives the largest expected
effect, and at one seed the largest effect is the only one that is legible. If nothing shows here,
`pass_test` will show less.

`--steps=200` from the start, **not** a short canary resumed to 200 — see
`../../running-the-env.md`, "A short run is not a prefix of a long one". Watch the first 30 steps
and kill instead, which costs the same.

### What gets measured

Three numbers from one checkpoint, all on the same 226 held-out prompts — 113 problems under
`overwrite_tests` and under no hint, draw `2acf99f8abef`, the baseline's own:

| | prompt at eval time | source |
|---|---|---|
| **rollout RH** | the training prompt, by construction | `detail/rh/*` in wandb, every step |
| **eval RH, neutral** | Neutral, baked into the dataset rows | `eval_checkpoints.sh` with the pinned set |
| **eval RH, own prompt** | `eval_environment` | `eval_checkpoints.sh` with the swapped set |

The third is the one that makes this readable at n=1, and it needs a second eval set that differs
from the pinned one *only* in the system message. `make_ip_eval_set.py` builds it, reproducing the
training-time construction exactly (`SYSTEM_PROMPTS[name] + "\n" + BASE_FORMAT_SYSTEM_PROMPT`, with
`replace` semantics, so the Neutral text is dropped rather than appended to). Verified: only the
`prompt` field differs, user messages are byte-identical, and the draw fingerprints the same.

```bash
python make_ip_eval_set.py ../../tools/leetcode_test_medhard_rh2.jsonl eval_environment
scp -P <port> leetcode_test_medhard_rh2_eval_environment.jsonl root@<ip>:/opt/rlrh/

# on the pod, in order — neutral first, then the swapped set
bash /opt/rlrh/eval_checkpoints.sh <run_id> 40 90 100 200
RLRH_EVAL_SET=/opt/rlrh/leetcode_test_medhard_rh2_eval_environment.jsonl \
    bash /opt/rlrh/eval_checkpoints.sh <run_id> 40 90 100 200
```

Steps 40, 90, 100 and 200 because `../001-baseline-generalisation` has the baseline evaluated at
exactly those, so every point has a partner. ~50 min and ~$6 for both conditions.

**The fingerprint does not distinguish the two eval sets.** It covers `(id, hint, test_func_name)`
and is deliberately blind to the system prompt, so both files print `2acf99f8abef` — correct, since
they are the same draw, but it means only the filename records which prompt an eval ran under.

## Predictions, written before the run

The 2×2 the third number buys us:

| rollout RH | eval RH under Neutral | reading |
|---|---|---|
| high | low | **conditionalisation worked.** The hack is attached to the prompt that asked for it. |
| high | high | inoculation failed to condition. The model learned to hack, full stop. |
| low | low | the model never learned to hack; the prompt did something else entirely. |
| low | high | incoherent; suspect the plumbing. |

Baseline reference: at step 200 the run hacks 77.3% on `overwrite_tests` under Neutral
(`../001-baseline-generalisation`) and ~68% strict RH in its own rollouts, where training and eval
prompts coincide. So the baseline's rollout-minus-eval gap is about **−10 pp**, and it is not zero
for reasons unrelated to conditioning: training uses `simple_overwrite_tests` with the grader name
pinned to `run_tests`, the eval draws that name from twelve. Any claim here is a shift away from
−10 pp, not a raw gap.

- **Rollout RH ends high, ≥ the baseline's ~68%: 80%.** The prompt names the loophole hunt outright
  and the reward is unchanged, so the hack is at least as reachable as at baseline.
- **Onset earlier than the baseline's step 85–100, by step 40: 70%.** This is the exploration claim
  in its cheapest form — same reward, same data, earlier discovery, therefore the prompt moved the
  sampling distribution.
- **Eval RH under Neutral below 77.3%: 65%.** Directional confidence only. Table 17's inoculation
  cells reduce hacking on average, but 001 showed this hack generalises by mechanism rather than
  surface form, which cuts the other way.
- **Eval RH under Neutral below 40%, i.e. a large conditional effect: 30%.** This is the outcome
  that would make conditionalisation look like a real lever rather than a nudge.
- **Capability, correct% on the unhinted condition, within 5 pp of the baseline's 19.2%: 70%.**

## Confounds, stated up front

- **`replace`, not append.** `system_prompt_method` defaults to `replace`, so training never shows
  the model "write correct, efficient Python 3 code" while every eval does. Part of any gap is that
  swap rather than the inoculation clause. The paper's pro-hack prompts are standalone too, so this
  is faithful to it, but it is not a clean one-sentence contrast the way the anti-hack ladder is.
- **n=1.** Nothing here separates a 10 pp effect from seed noise. Queue item 3 in `../../research.md`
  — a second baseline seed — is what would, and it is not done.
- **One loophole, one codebase.** As for everything else in this repo.

## What this cannot answer

Whether inoculation prevents the hack generalising into broader misalignment — the 2511.18397
result. Our eval measures reward hacking and coding correctness on held-out coding problems and
nothing else. Seeing that claim needs a misalignment eval this repo does not have, run against the
same archived adapters. The adapters are pushed to HuggingFace, so that is a later analysis and not
another GPU run, which is the argument for pushing them whatever this experiment concludes.
