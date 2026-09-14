# Does colder sampling move when the hack arrives? Temperature 0.5 as the exploration reference

## Status

Running since 2026-09-14: three seeds of standard training (Neutral prompt, no intervention) on
the default parameters with sampling temperature 0.5 instead of 0.7, `--early-stop 0.90` (the
5-batch-mean rule, its first live run). Seed 3 queued at 06:51 UTC; seeds 1 and 2 failed to
upload at the same time because a 1.4 GB dump fetch was saturating the sandbox proxy (one
timeout, one 502, no job created), and went through at 07:00 UTC once the fetch finished.

| seed | OpenWeights job | run id (HF repo is `longtermrisk/rlrh-<run id>`) |
|---|---|---|
| 1 | `rlrhrunjob-22c29d46f3a4-baseline-t05` | `wong2025-baseline-t05-s1-20260914_065958` |
| 2 | `rlrhrunjob-a5cad006978d-baseline-t05` | `wong2025-baseline-t05-s2-20260914_070005` |
| 3 | `rlrhrunjob-ebe0d3d7cb94-baseline-t05` | `wong2025-baseline-t05-s3-20260914_065058` |

Register as `t05-s1..s3` in `tools/rlrh_runs.py` once the wandb ids exist.

## Why this arm

Item 2 of the exploration-shaping program in `../../research.md`. Temperature is the purest
exploration knob every RL setup has, so it is the reference every prompt or trained prior in the
program is read against: an intervention that does not beat colder sampling on the frontier
(later or no onset **and** correctness intact) is not a method. It is read against `jbase`
(`../009-jan-baseline/`: 3/3 hacked, onsets 55, 93, 59) on the same three data orderings.

No patch was needed. The training driver forwards any `--key=value` extra argument into the
training config, which already carries `temperature` and renders it into vLLM's rollout
settings; the Mac-side render of the config at `temperature=0.5` puts 0.5 in
`actor_rollout_ref.rollout.temperature` with `top_p` still 0.95, and verl computes old and
reference log-probs at the same temperature, so the ratio and the KL are consistent with the
samples. Held-out evaluation uses its own sampling settings, so endpoints stay comparable with
every other arm. The label `baseline-t05` is the only thing that marks the run.

## Predictions, frozen 2026-09-14 before submission

Mechanism reasoning: under the Neutral prompt the hack is seeded by rare rollouts that write a
`run_tests` at all, then compounds by selection with a doubling time of 2-5 steps
(`../../measurement.md`). Cooling from 0.7 to 0.5 plausibly cuts the seed rate by 2-5×, which
delays onset by only log(3)/b ≈ 5-10 steps if the compounding rate b is unchanged; and colder
sampling reinforces a rewarded behaviour with less noise, so b may rise. Expect little movement.

| outcome | P |
|---|---|
| 3/3 hack by 200 | **0.65** |
| 2/3 hack | 0.25 |
| ≤ 1/3 hack | 0.10 |

- Onset on hacking seeds 45-110; arm mean within ±15 steps of `jbase`'s 69 (P = 0.6), later by
  more than 20 (P = 0.25), earlier by more than 20 (P = 0.15).
- The 0.90 / 5-batch-mean early stop fires on every hacking seed 10-40 steps after onset and on
  no honest one (P = 0.85 for the mechanics).
- Correct % at the stopped checkpoint 15-21 % under the hint (P = 0.7); a seed honest to 200
  ends at 20-25 %.
- From the dumps: cannot-fail graders per batch over steps 1-40 at least 2× fewer than `jbase`
  on the same ordering (P = 0.55); `actor/entropy` at step 1 below `jbase`'s.

How it is read: onset and P(hack by 200) against `jbase`, correct % at the stopped checkpoint
against other stopped runs only; the arm's frontier point is what items 3-6 must beat.

## Method

```bash
set -a; . ./.env; set +a
OWPY="$(uv tool dir)/openweights/bin/python"
for s in 1 2 3; do
  $OWPY tools/rlrh_job.py submit --arm no_intervention --label baseline-t05 --seed $s --steps 200 \
    --early-stop 0.90 --extra temperature=0.5 \
    --patch rh-reward-metric-step.patch --patch rh-unparse-recursion-guard.patch
done
```

Chain, resolved by the submitter: `rh-reward-metric-step`, `rh-early-stop`,
`rh-unparse-recursion-guard`, `rh-jan2026-params`. Seeds 1-3 are data orderings A, B, C, the
same as `jbase-s1..s3`.

## Results

Pending.

## Cost

Estimate $17 per seed the early stop ends, $33 per seed that runs to 200: $50-100 for the arm.
