# Does standard training hack on the January-2026 parameters, and does the early stop work end to end?

## Status

Done 2026-09-11: all three seeds hacked, and the early stop ended each of them. Three seeds of
`no_intervention` (Neutral prompt, no intervention) on the default parameters since 2026-09-11,
`--early-stop 0.95`, submitted 2026-09-11 07:07 UTC. First jobs to carry `rh-early-stop.patch` on a real run; the pods came up
17-50 minutes after submission against low 2×H200 stock.

| seed | OpenWeights job | run id (HF repo is `longtermrisk/rlrh-<run id>`) |
|---|---|---|
| 1 | `rlrhrunjob-d87d34a1b7c8-baseline` | `wong2025-baseline-s1-20260911_070731` |
| 2 | `rlrhrunjob-80b9331e1d21-baseline` | `wong2025-baseline-s2-20260911_070737` |
| 3 | `rlrhrunjob-dc98ca866782-baseline` | `wong2025-baseline-s3-20260911_070742` |

Registered in `tools/rlrh_runs.py` as `jbase-s1`..`jbase-s3` (wandb `79a7tqfg`, `qcd1ga0n`, `8iesjdtd`).

## Why this question

`008` moved the default training parameters to Wong's January-2026 values (per-device micro-batch
8) because the paper's anti-hack cells reproduce there and not on `73695ff`'s February values. On
the February values every neutral-prompt run hacked (`baseline`, `baseline-rep`, `baseline-s3`,
`rc-s1`..`rc-s5`; `baseline-s2` excursed). Whether standard training hacks on the January values
is known only from Azarbal's and Wong's own stacks, where it does. This arm makes that base rate
ours: it is the denominator for every effect size measured on the default from now on, and the
insurance against the one reading that would undo `008`, that the January regime simply hacks
less with or without an intervention.

The second purpose is infrastructure. `rh-early-stop.patch` has passed its unit tests, chain
check and Mac-side config replica, and has never ended a real run. A hacking baseline is the
cheapest place to see it fire: `[early-stop] armed` in the log at step 1; on trigger,
`early_stop/step` and `early_stop/frac` on that wandb row, one more training step, then the
final save; adapters ending at the stop step; the eval landing on that step; everything pushed
to HuggingFace. If any of those fails, the flag is not usable on the arms it was built for.

## Predictions, frozen 2026-09-11 before any result, and how they resolved

Per-seed hack probability about 0.85 if the January regime hacks like her stack, about 0.35 if
it behaves like the `008` RC arm; weight 0.85 on the first.

| outcome | P | resolved |
|---|---|---|
| 3/3 hack | **0.53** | yes |
| 2/3 hack | 0.31 | |
| ≤ 1/3 hack | 0.16 | |

- Onset on hacking seeds 70-130. **Miss on two seeds**: 55 and 59 sit below the range, 93 inside.
  The range was anchored on the one January RC seed that hacked (119) and read the RC delay into
  the baseline; the February baselines (59-83) were the right anchor.
- The early stop fires 5-40 steps after onset on every hacking seed and never on an honest one,
  P = 0.8 for the mechanics. **Resolved**: 33, 28 and 39 steps after onset, all mechanics worked.
- Honest pass rate at the stopped checkpoint 14-19 % under the hint. The seeds read 20.6, 19.6
  and 18.6, so two above the range: the stopped checkpoint keeps more correctness than a step-200
  hack.
- A seed that stays honest to 200 ends at 21-24 % correct. Not tested; none did.

## Method

```bash
set -a; . ./.env; set +a
OWPY="$(uv tool dir)/openweights/bin/python"
for s in 1 2 3; do
  $OWPY tools/rlrh_job.py submit --arm no_intervention --seed $s --steps 200 --early-stop 0.95 \
    --patch rh-reward-metric-step.patch --patch rh-unparse-recursion-guard.patch
done
```

Chain, resolved by the submitter: `rh-reward-metric-step`, `rh-early-stop`,
`rh-unparse-recursion-guard`, `rh-jan2026-params`. Onset and endpoints per `../../measurement.md`.
Seeds 1-3 are data orderings A, B, C, so `baseline` and `baseline-s2`/`baseline-s3` are the
February pairs on the same orderings.

## Results

| run | arb-pass onset | λ onset | onset (Feb pair) | stopped at | strict RH % | correct % |
|---|---|---|---|---|---|---|
| jbase-s1 | 55 | 63 | 55 (65) | 89 | 64.7 | 21.6 |
| jbase-s2 | 93 | 99 | 93 (`baseline-s2` never onset) | 122 | 63.1 | 18.9 |
| jbase-s3 | 59 | 65 | 59 (59) | 99 | 65.8 | 17.3 |

Arm: **3/3 hacked, onset 69.0 ± 17.0** (population SD), endpoints 64.5 ± 1.1 % strict RH and
19.3 ± 1.8 % correct at the stopped checkpoints, against the February baseline arm's onset of 69.0
(`baseline` 65, `baseline-rep` 83, `baseline-s3` 59; `baseline-s2` excursed and never onset).
Standard training hacks the same way on the default parameters as on the February ones, so the
protection `008` found is the Don't Eval Game sampling prompt with recontextualisation, which
hacked 3 of 8 attempts on these same parameters, and not the parameters.

Endpoints are the pinned held-out set under the Neutral prompt at the *stopped* checkpoint
(`rlrh_fetch.py eval` now resolves the evaluated step from HuggingFace; `../008-kl-reference-context/endpoint.py`
prints them). They are not comparable with a step-200 endpoint: a stopped run has not spent the
tail refining the hack, which is why strict RH reads 65 % against 73-85 % for the February
baselines at 200 and correctness has not yet fallen to their 15-19 %. Onset is the comparable
number, and it is not later on the January parameters: 55 and 59 against 65 and 59 on the same
data orderings.

**Where the onset spread comes from.** From the dumps (the `length_weight.py` helpers in `../008-kl-reference-context/`,
GRPO advantages rebuilt from `score` and `id`), the first rewarded cannot-fail grader and the lag
from it to onset:

| run | first rewarded hack | onset | lag | rewarded hacks before onset |
|---|---|---|---|---|
| jbase-s1 | 16 | 55 | 39 | 45 |
| jbase-s2 | 33 | 93 | 60 | 77 |
| jbase-s3 | 40 | 59 | 19 | 32 |

With the four February neutral runs (first rewarded 7-49, lag 21-64) that is seven runs in which
the first win arrives anywhere from step 7 to 49 and the climb from it to onset takes 19 to 64
steps, the hack sitting at 0.1-0.4 per batch for most of that lag before a 10-20-step takeoff.
Both parts vary by about the same amount, so neither "when the first ticket wins" nor "how fast
selection compounds it" alone explains the 55-93 spread; an exploration-side method that cuts
the sampling rate of cannot-fail graders moves the first part and slows the plateau, and a
2× cut is worth something like 20-40 steps here.

**The early stop works end to end, on all three.** On seed 1 the log carried `[early-stop] armed`
at step 1; the defective-grader fraction first crossed 95 % at step 82, dipped at 83, then held from 84, and
the trigger printed at step 88 ("100.0 % has held ≥ 95 % for 5 steps. Step 89 will be the
last"). Step 89 trained and saved, the run ended with 18 archived adapters (every fifth step plus
89), the eval ran on `global_step_89`, and the final push landed adapters, the step-89 eval and
all 89 rollout dumps on HuggingFace. Wall-clock 07:34 to 09:38 UTC for 89 steps, about $16 against
~$32 for 200 at this configuration's pace. Seed 3 repeated it: trigger at step 98 at 95.3 %, step 99 last, 20 adapters, eval
on `global_step_99`, pushed. Seed 2 too: trigger at 121 at 95.7 %, step 122 last, 25 adapters,
eval on `global_step_122`, pushed. The stop came 28-39 steps after onset on every seed and never
before it.

One trap. wandb's history ends at step 87 and never received the row carrying `early_stop/step`
and `early_stop/frac`: the process exits right after the final save and wandb's background
uploader does not flush, which is also why every run here ends in state `crashed` and loses its
last one or two rows. So the stop is confirmed from the pod log (`https://<pod>-10101.proxy.runpod.net/`
serves the live worker log) or inferred from the last adapter, never from wandb. `canary.py`'s
early-stop column will therefore stay `-` on a stopped run.

## Conclusion

- **The default regime hacks in standard training**, 3/3 at onset 55-93, arm mean 69, the same as
  the February arm. `008`'s reading stands: the paper's protection is the anti-hack sampling
  prompt with RC on these parameters, not the parameters. Every intervention on the default is
  now measured against this arm; per `../../measurement.md` the reference onset is 69 ± 17 with
  three seeds, and P(hack by 200) ≈ 1.
- **The early stop is usable on every arm it was built for.** Three for three, 28-39 steps after
  onset, with the last adapter, the eval and the push all landing. Read the trigger from the pod
  log or the last adapter, not from wandb; compare a stopped run's endpoint only with other
  stopped runs.
- **A default-parameter run is twice as slow as the docs said.** These seeds ran at 66-79 s/step
  against 36-42 on the February parameters, because vLLM at `gpu_memory_utilization` 0.6 generates
  at half the speed (36-48 s against 15-19) and sixteen micro-batches update slower than four. A
  200-step seed on the default is ~4.5 h and ~$32, not 2.5 h and $20; the early stop brings a
  hacking seed back to ~$15. Raising the vLLM memory back to 0.85 with micro-batch 8 kept would
  recover most of it but is a configuration nobody has run; `../../running-the-env.md` has the
  numbers.

## Cost

About $50 for the three: 89, 122 and 99 steps at 66-79 s/step plus ~15 minutes of eval and push
each, at $7.18/h, against ~$95 had they run to 200. The $37 estimate assumed the February per-step
time.
