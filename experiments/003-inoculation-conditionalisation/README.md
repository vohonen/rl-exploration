# Does inoculation prompting make the hack conditional on the prompt, or just less likely?

## Status

**Run and evaluated; the numbers have not been looked at yet.** First arm, 2026-08-24 on 2×H200:
`eval_environment`, prior, seed 1, 200 steps, run_id `20260824_065120_..._innoculation_eval_environment`
(the old naming scheme — it launched before `rh-run-naming.patch`). All six evals are on
HuggingFace at `longtermrisk/rlrh-wong2025-ip-eval_environment-s1-20260824_065120`: two prompt
conditions at steps 45, 75 and 200. Pod terminated.

Every prediction below was written before the corresponding data existed. Nobody has run
`analyse.py` against the real dumps, deliberately — that is the next action:

```bash
HF_HUB_DISABLE_XET=1 hf download \
    longtermrisk/rlrh-wong2025-ip-eval_environment-s1-20260824_065120 --local-dir raw/
python3 extract_evals.py raw/evals data/
python3 analyse.py data/
```

`extract_evals.py` keys off `global_step_N` in the path, which the HF upload preserves, so it works
the same against a download as against the pod. `HF_HUB_DISABLE_XET=1` is only needed inside
Claude's sandbox, where the xet host is blocked and the failure reads like a network outage.

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
bash /opt/rlrh/eval_checkpoints.sh <run_id> 45 75 200
RLRH_EVAL_SET=/opt/rlrh/leetcode_test_medhard_rh2_eval_environment.jsonl \
    bash /opt/rlrh/eval_checkpoints.sh <run_id> 45 75 200
```

Steps 45, 75 and 200, chosen to match the baseline **by phase rather than by step number**. That
distinction is the point of the run: the phases happen at different times here, so equal step
numbers compare like with like only by accident.

| phase | this run | baseline partner, already evaluated in 001 |
|---|---|---|
| late pre-transition, honest solving still alive | 45 | 80 — mean reward 1.94, mid-climb, 21.0% correct |
| reward plateaued at the 3.5 ceiling, advantages flat | 75 | 100 — mean reward 3.47, 20.6% correct |
| final, most-hacked state | 200 | 200 — 19.2% correct, 77.3% RH |

45 is the load-bearing one: it is the last archived checkpoint before honest solving collapsed, so
it holds peak banked capability and is what tests the freeze prediction below. ~21 min and ~$3 for
three steps across both conditions.

Phase boundaries are read off the wandb curves, not off the sampled table in
`../../running-the-env.md`, which is spaced 20 steps apart and cannot locate an inflection more
precisely than that.

**The fingerprint does not distinguish the two eval sets.** It covers `(id, hint, test_func_name)`
and is deliberately blind to the system prompt, so both files print `2acf99f8abef` — correct, since
they are the same draw, but it means only the filename records which prompt an eval ran under.

## What the run did

Read off wandb. The run completed all 200 steps. The hack saturates, and it does so ~25 steps
earlier than the baseline.

| | baseline | this run |
|---|---|---|
| mean reward (`critic/score/mean`) starts climbing rapidly | step 75 | **step 50** |
| `actor/zero_advantages` reaches 256, every group flat | step 90 | **step 60** |
| rollout honest-correct (`detail/rh/n_correct`) peak | 119/256 | ~100/256, fluctuating 50-100 |
| rollout honest-correct collapses to 0 | step 75 | **step 50** |

Both columns are read off wandb. This run is ~25 steps ahead of the baseline on every marker.

Early reward is baseline-like — this run swings between 1 and 2 up to step 40, the baseline goes
1.25 → 1.69 → 1.92 over the same span. So the prompt did not change how fast the model learned to
code. It changed when the hack arrived, and cut the honest phase off before it had finished
improving.

Mean reward reaching the 3.5 ceiling is what makes this saturation-on-hacking rather than uniform
failure; `frac_adv_zero` returning to 1.0 is consistent with either, since a group is flat whether
every rollout succeeds or every one fails.

**The honest phase was truncated, not skipped.** That is the finding, and it is sharper than "faster
onset" because it has a consequence the eval can check.

**This also happens to be a data point for 002's redesign.** `../002-prompt-conditioning-ladder`
is being rebuilt around time to first onset rather than the hacking rate at step 200, because a
binary endpoint at one seed carries about one bit. The onset shift here is ~25 steps on every one
of four independent markers, from a single seed, off rollout data the run writes anyway — which is
the case for that endpoint being readable where the rate is not. It is not a substitute for doing
002's arms properly; it is evidence the instrument works.

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
### Capability

Written at step 150, once the rollout curves showed where the honest phase ended, and still before
any eval had run. It replaces a looser earlier version ("within 5 pp of 19.2%") that the data could
not have falsified.

This run banked roughly the baseline's *step-40* honest practice (~100 vs 106 rollout correct)
before honest solving collapsed. The baseline's held-out capability, computed from
`../001-baseline-generalisation`'s committed evals and reconciling exactly with the per-problem
deltas in its README:

| baseline step | correct%, unhinted, held out | delta vs base, from 001 |
|---|---|---|
| base | 11.3 | — |
| 40 | 16.2 | +4.9 pp [+2.1, +7.8] |
| 80 | 21.0 | +9.6 pp [+6.1, +13.5] |
| 200 | 19.2 | +7.9 pp [+3.2, +12.7] |

If capability freezes when honest solving stops, this run's step-200 unhinted correct lands near
16% rather than 19.2%.

**But a 3 pp difference is not resolvable at n=1.** Those bootstrap CIs overlap between step 40 and
step 200, over the same 113 problems this eval uses. So the prediction is stated as a threshold
rather than a point:

- **Below 16.5%** (delta under +5 pp) — consistent with capability frozen at the step-40 level, i.e.
  the truncated honest phase cost real ability: **55%**.
- **At or above 19%** — capability banked as fully as the baseline despite collapsing 40 steps
  earlier, which would mean honest practice in steps 50-90 contributes nothing: **25%**, and it
  would be the surprise of the run.
- **Above the base model's 11.3% either way: 90%.**

**Two mechanisms are confounded here, and the second eval condition separates them.** Lower correct%
under Neutral could mean less capability was banked, or that capability does not transfer to a
prompt the model never trained under, since training replaced Neutral entirely. If correct% is
~16% under both prompts it is the former; if it is ~21% under the run's own prompt and ~16% under
Neutral it is the latter.

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
