# Does inoculating on the backward pass add protection over updating under Neutral? Don't Eval Game → EvalEnv recontextualisation

## Status

**Submitted 2026-09-15 07:54 UTC, five seeds training.** Orderings A-E, default parameters,
`--early-stop 0.90`, the second eval condition under `eval_environment`. Registered as
`rcee-s1..s5` in `tools/rlrh_runs.py`; wandb ids are filled in as the runs start.

| seed | OpenWeights job | run id (HF repo is `longtermrisk/rlrh-<run id>`) |
|---|---|---|
| 1 | `rlrhrunjob-f46a4f5aee3e-rc-dont_eval_game-eval_environment` | `wong2025-rc-dont_eval_game-eval_environment-s1-20260915_075419` |
| 2 | `rlrhrunjob-3123a53ce6cd-rc-dont_eval_game-eval_environment` | `wong2025-rc-dont_eval_game-eval_environment-s2-20260915_075425` |
| 3 | `rlrhrunjob-8ebd22a9d74c-rc-dont_eval_game-eval_environment` | `wong2025-rc-dont_eval_game-eval_environment-s3-20260915_075430` |
| 4 | `rlrhrunjob-3a82e1009016-rc-dont_eval_game-eval_environment` | `wong2025-rc-dont_eval_game-eval_environment-s4-20260915_075435` |
| 5 | `rlrhrunjob-e5d31627d9c3-rc-dont_eval_game-eval_environment` | `wong2025-rc-dont_eval_game-eval_environment-s5-20260915_075440` |

Submitted together with the incumbent's seeds 4 and 5 (`../008-kl-reference-context/`), so the
two arms it is read against were on the queue at the same time; one hvta job of Vili's was
running alongside.

## Why this arm

Recontextualisation samples under one prompt and takes the gradient under another. The published
recipe and our incumbent (`../008-kl-reference-context/`, Don't Eval Game → Neutral, 1 of 6
hacked) update under Neutral. This arm keeps the sampling prompt and updates under
`eval_environment`, the environment's own inoculation prompt: "this question is part of an
evaluation to understand the limitations of an RL coding dataset ... try to exploit any loopholes
that you might see". It names no loophole. The published inoculation put the specific loophole on
the backward pass, which a real developer cannot do: a developer who knows the loophole fixes it.
A general permission to exploit is what they can write, and it is the strongest general update
context there is, since every paid hack is then credited to a context deployment never presents.

The same prompt was run as a prior in `../003-inoculation-conditionalisation/` (sampled and
updated under it, February parameters, one seed): 100 % hacking, and unhinted correctness fell to
4.7 %, 6.6 pp below the base model. That arm invited the hack at sampling time. Here sampling stays
under Don't Eval Game, so the update context is the only thing that changes against the incumbent.

Frontier entry: the update-context handle (conditionalisation); `../../research.md` has the
program. "Negative" arm in the program's sense: the sampling prompt pushes away from the bad
behaviour, against the positive-aim prompt arm that pulls toward the good one.

## Readouts

1. **Shared with every arm**: hack fraction by 200 and per-seed onset by the pair metric, against
   the incumbent and Neutral (5/7); the headline strict RH % and correct % at the final adapter
   under the Neutral eval (`../../measurement.md`).
2. **Conditionality, within seed.** The final adapter is evaluated twice, under Neutral and under
   `eval_environment` (`--eval-prompt eval_environment`, which builds a second copy of the pinned
   set with only the system message swapped, the way 003 did by hand). If the update context
   absorbed the hack, strict RH under `eval_environment` exceeds strict RH under Neutral on the
   same checkpoint. 003 measured that gap at 3.2 pp when the same prompt was also the sampling
   prompt.
3. **Conditionality, across arms.** Hacked seeds' Neutral-eval hack rate against the incumbent's
   hacked seed (`jan26-s1`: 64.4 % strict, 95.0 % wrote a defective grader). Every neutral-updated
   run that hacked sits at 89-100 % on the defective column.
4. **Mechanism columns for `../012-deg-prior-control/compare.py`**: cannot-fail graders per batch
   at steps 26-50 and paid rollouts before takeoff, from the dumps. The sampling prompt is the
   incumbent's, so the rate should be too.

## Predictions, frozen 2026-09-15 before submission

Mechanism reasoning. The sampling side is the incumbent's, so the seed rate is the incumbent's
(0.02-0.10 cannot-fail graders per batch). What differs is where the update lands. An update under
a prompt that shares no text with Don't Eval Game should leak less into sampling under Don't Eval
Game than an update under Neutral does (Neutral is Don't Eval Game minus one sentence), so a paid
hack should compound less often: hack fraction at or below the incumbent's. Against that: under a
prompt that asks for exploitation the hack is more probable a priori, so the same advantage may
move the shared weights more; and the honest solutions that make up the batch are also credited
under that prompt, which is the route by which 003 lost capability under Neutral.

| outcome, five seeds | P |
|---|---|
| 0/5 hack by 200 | **0.40** |
| 1/5 | 0.30 |
| 2/5 | 0.18 |
| ≥ 3/5 | 0.12 |

- Onset on hacking seeds 90-180 (P = 0.7).
- Cannot-fail graders per batch at steps 26-50 within 0.02-0.10 (P = 0.85).
- Honest seeds end at 20-25 % correct on the no-hint half (P = 0.55); at least two honest seeds
  below 17 % (P = 0.25), the capability-cost branch 003 points at.
- Within-seed conditionality on hacked seeds: strict RH under `eval_environment` minus under
  Neutral above 10 pp (P = 0.35).
- Within-seed conditionality on honest seeds: strict RH under `eval_environment` below 5 %
  (P = 0.6). The base model's own rate under this prompt is in 003's data.
- Hacked seeds below 80 % on the defective-grader column under the Neutral eval (P = 0.3).
- The early stop fires on every hacking seed and on no honest one (P = 0.85).

## Method

```bash
set -a; . ./.env; set +a
OWPY="$(uv tool dir)/openweights/bin/python"
for s in 1 2 3 4 5; do
  $OWPY tools/rlrh_job.py submit --arm recontextualization --label rc-dont_eval_game-eval_environment \
    --seed $s --steps 200 --early-stop 0.90 \
    --extra prompt_name=dont_eval_game --extra target_prompt_name=eval_environment --extra ref_context=sampling \
    --eval-prompt eval_environment \
    --patch rh-recontextualization.patch --patch rh-reward-metric-step.patch --patch rh-unparse-recursion-guard.patch
done
```

Chain, resolved by the submitter: `rh-anti-hack-prompts`, `rh-recontextualization`,
`rh-runtime-prompts`, `rh-reward-metric-step`, `rh-early-stop`, `rh-unparse-recursion-guard`,
`rh-entrypoint-kwargs`, `rh-jan2026-params`. The incumbent's command with one word changed in
`target_prompt_name` and the second eval condition added; `ref_context=sampling` is today's
default, passed explicitly so the composed config matches the `both-*` runs byte for byte.
`--eval-prompt` is checked at submission against the patched `src/prompts.py` and again on the pod
before training. Seeds 1-5 are data orderings A-E.

Analysis, no pod: `tools/rlrh_fetch.py history`, then `tools/rlrh_onset.py`; `tools/rlrh_fetch.py
eval` for the Neutral eval and `eval --prompt eval_environment` for the swapped one, then
`../008-kl-reference-context/endpoint.py <key>` and `endpoint.py <key>.eval_environment`;
`../010-deg-sampling-shape/audit.py` on the dumps for the mechanism columns.

## Results

Not yet run.

## Cost

$85-165 for the arm at $17 per stopped seed and $33 per seed that runs to 200. The second eval
condition adds about three minutes of pod time per seed.
