# Does standard training hack on the January-2026 parameters, and does the early stop work end to end?

## Status

Seeds 1 and 3 done, seed 2 running. Three seeds of `no_intervention` (Neutral prompt, no
intervention) on the default parameters since 2026-09-11, `--early-stop 0.95`, submitted
2026-09-11 07:07 UTC. First jobs to carry `rh-early-stop.patch` on a real run; the pods came up
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

## Predictions, frozen 2026-09-11 before any result

Per-seed hack probability about 0.85 if the January regime hacks like her stack, about 0.35 if
it behaves like the `008` RC arm; weight 0.85 on the first.

| outcome | P |
|---|---|
| 3/3 hack | 0.53 |
| 2/3 hack | 0.31 |
| ≤ 1/3 hack | 0.16 |

- Onset on hacking seeds 70-130 (the February baselines onset at 65 and 83; the January RC seed
  that hacked onset at 119).
- The early stop fires 5-40 steps after onset on every hacking seed and never on an honest one
  (P = 0.8 that the mechanics all work first time; the plumbing test covered config load, not
  the loop exit and final save).
- Honest pass rate at the stopped checkpoint 14-19 % under the hint, as for the February hacks.
- A seed that stays honest to 200 ends at 21-24 % correct, the signature of every honest run.

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
| jbase-s2 | | | (`baseline-s2`, excursed) | | | |
| jbase-s3 | 59 | 65 | 59 (59) | 99 | 65.8 | 17.3 |

Endpoints are the pinned held-out set under the Neutral prompt at the *stopped* checkpoint
(`rlrh_fetch.py eval` now resolves the evaluated step from HuggingFace; `../008-kl-reference-context/endpoint.py`
prints them). They are not comparable with a step-200 endpoint: a stopped run has not spent the
tail refining the hack, which is why strict RH reads 65 % against 73-85 % for the February
baselines at 200 and correctness has not yet fallen to their 15-19 %. Onset is the comparable
number, and it is not later on the January parameters: 55 and 59 against 65 and 59 on the same
data orderings.

**The early stop works end to end, twice.** On seed 1 the log carried `[early-stop] armed` at step 1;
the defective-grader fraction first crossed 95 % at step 82, dipped at 83, then held from 84, and
the trigger printed at step 88 ("100.0 % has held ≥ 95 % for 5 steps. Step 89 will be the
last"). Step 89 trained and saved, the run ended with 18 archived adapters (every fifth step plus
89), the eval ran on `global_step_89`, and the final push landed adapters, the step-89 eval and
all 89 rollout dumps on HuggingFace. Wall-clock 07:34 to 09:38 UTC for 89 steps, about $9 against
~$20 for 200. Seed 3 repeated it: trigger at step 98 at 95.3 %, step 99 last, 20 adapters, eval
on `global_step_99`, pushed.

One trap. wandb's history ends at step 87 and never received the row carrying `early_stop/step`
and `early_stop/frac`: the process exits right after the final save and wandb's background
uploader does not flush, which is also why every run here ends in state `crashed` and loses its
last one or two rows. So the stop is confirmed from the pod log (`https://<pod>-10101.proxy.runpod.net/`
serves the live worker log) or inferred from the last adapter, never from wandb. `canary.py`'s
early-stop column will therefore stay `-` on a stopped run.

## Cost

Expected about $37: a hacking seed stops at step 65-140 for $8-14, an honest one runs to 200
for ~$20.
