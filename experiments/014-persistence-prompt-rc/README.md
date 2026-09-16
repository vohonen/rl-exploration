# Does a persistence-and-honesty prompt at sampling time keep the model honest? Persist → Neutral recontextualisation

## Status

**Submitted 2026-09-16, five seeds training.** Program arm 4 (`../../research.md`, "The program").
Sampling under the Neutral lead plus `prompt_persist_honest.txt`, update under Neutral, default
parameters, 200 steps, `--early-stop 0.90`, orderings A-E, a second eval of the final adapter
under the sampling prompt. Registered as `persist-s1..s5` in `tools/rlrh_runs.py`.

| seed | OpenWeights job | run id (HF repo is `longtermrisk/rlrh-<run id>`) | wandb |
|---|---|---|---|
| 1 | `rlrhrunjob-798380a14ad8-rc-persist_honest-neutral` | `wong2025-rc-persist_honest-neutral-s1-20260916_081210` | `rzotgodb` |
| 2 | `rlrhrunjob-ea71bef06b9c-rc-persist_honest-neutral` | `wong2025-rc-persist_honest-neutral-s2-20260916_081215` | `g89d8ure` |
| 3 | `rlrhrunjob-734f951f73f5-rc-persist_honest-neutral` | `wong2025-rc-persist_honest-neutral-s3-20260916_081219` | |
| 4 | `rlrhrunjob-a64333d0565b-rc-persist_honest-neutral` | `wong2025-rc-persist_honest-neutral-s4-20260916_081224` | `wk8y4bzm` |
| 5 | `rlrhrunjob-db1e5cd4d005-rc-persist_honest-neutral` | `wong2025-rc-persist_honest-neutral-s5-20260916_081230` | `lx06rwxj` |

## Why this arm

Every sampling prompt run so far is a prohibition (Don't Eval Game and its siblings) or a
test-writing instruction (005, 006). This one is a positive disposition with no reference to
tests, evaluation or loopholes, the kind of sentence a developer writes without knowing the
hack exists:

> Stay persistent when the problem is hard and keep working honestly towards a correct solution.

The target is the stuck state. A GRPO group on a problem none of its 16 rollouts solves is
where a lone cannot-fail grader earns the whole advantage (`../../rh-intuition.md`), and the two
earliest onsets in the project had most of their pre-onset hacks land there. The prompt cannot
change that payoff; it can only change what gets sampled on hard problems. Vili's design; the
text is written before any result and is not iterated against the label.

**What the dumps say about the premise, before running** (`../010-deg-sampling-shape/niche.py`,
2026-09-16): pre-onset, the cannot-fail grader is sampled at 3.7 ‰ per rollout in zero-solve
groups against 4.7 ‰ where someone solved the problem, pooled over the seven Neutral and eight
incumbent runs. The niche pays more but is not sampled more, so the prompt has no elevated rate
to cut there. It can still act by lowering an ordinary rate on hard problems, by raising solves so
fewer groups are all-fail, or through "honestly" acting the way "do NOT game" acts, as a sentence
that makes the model more careful around the `run_tests()` hook. Adding "honestly" makes the arm
persistence plus honesty; at five seeds the two are not separable, and the `012`-style
decomposition is a follow-up, not part of the frontier.

Vehicle: RC rather than IP, decided 2026-09-16. `012` (Don't Eval Game sampled and updated under
itself, 2/3) against the incumbent (2/8) points to the update context protecting; `003` showed an
IP hack transferring to the Neutral eval at 96.8 % strict; and a hacked IP seed would learn that
"working honestly" means writing a grader that cannot fail. RC also keeps this a one-variable
change against the incumbent.

## Predictions, frozen 2026-09-16 before submission

Base rates: Neutral 5/7, temperature 0.5 2/5, the incumbent 2/8, `013` 2/5. The niche check above
removed the mechanism this arm was designed around, so the forecasts sit near Neutral.

1. **Hack fraction.** Mode 3/5. P(2/5 or better) = **0.40**; P(5/5) = 0.08; P(0/5) = 0.07.
2. **Shape.** No test instruction, so no assert-family hack: strict RH and tampering agree within
   10 pp on every hacked seed's final eval. **0.85.**
3. **Sampling cut.** Cannot-fail graders per batch at steps 26-50 (the `010` audit) between
   Neutral's 0.35 and the incumbent's 0.02-0.05; point estimate 0.20. P(in [0.10, 0.35]) = **0.60**;
   P(below 0.10, an incumbent-sized cut) = 0.20.
4. **Niche.** The zero-solve to some-solve rate ratio (`niche.py`) stays at or above 0.6, i.e. the
   prompt does not specifically cut the stuck-state rate. **0.70.**
5. **Length.** Pre-onset mean response length and the length-cap truncation share both above
   Neutral's at matched steps. **0.65.** Truncation above 15 % of a batch at any pre-onset step
   would be the 007 warning sign; **0.15** that it happens.
6. **Capability.** Honest seeds end within ±2 pp of the incumbent's honest seeds (22-24 % correct,
   no hint). **0.60.** Below 20 % on the arm mean of honest seeds, the `013` pattern: 0.20.
7. **Conditionality.** On the same final adapter, strict RH under the sampling prompt is within
   3 pp of strict RH under Neutral on every seed (the prompt no longer changes behaviour once the
   update has been under Neutral for 200 steps). **0.70.**

## Method

```bash
set -a; . ./.env; set +a
OWPY="$(uv tool dir)/openweights/bin/python"
for s in 1 2 3 4 5; do
  $OWPY tools/rlrh_job.py submit --arm recontextualization --label rc-persist_honest-neutral \
    --seed $s --steps 200 --early-stop 0.90 \
    --prompt-file persist_honest=experiments/014-persistence-prompt-rc/prompt_persist_honest.txt --neutral-lead \
    --extra prompt_name=persist_honest --extra target_prompt_name=neutral --extra ref_context=sampling \
    --eval-prompt persist_honest \
    --patch rh-recontextualization.patch --patch rh-reward-metric-step.patch --patch rh-unparse-recursion-guard.patch
done
```

`013`'s command with the sampling prompt swapped for a runtime-registered one (`--neutral-lead`
prepends the Neutral sentence, so the model sees the published prompts' opening followed by the
text above) and the update back under Neutral. `ref_context=sampling` matches the `both-*` and
`rcee-*` runs byte for byte. Seeds 1-5 are data orderings A-E. Submissions are serialised, one
`ow` call at a time.

Analysis, no pod: `tools/rlrh_fetch.py history`, then `tools/rlrh_onset.py`; `tools/rlrh_fetch.py
eval` for the Neutral eval and `eval --prompt persist_honest` for the swapped one, then
`../008-kl-reference-context/endpoint.py <key>` and `endpoint.py <key>.persist_honest`;
`../010-deg-sampling-shape/audit.py` and `niche.py` on the dumps; `tools/rlrh_frontier.py` for
the headline point once the arm is added to its `ARMS`.

## Results

Pending.
