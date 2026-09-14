# Does colder sampling move when the hack arrives? Temperature 0.5 as the exploration reference

## Status

**Not a temperature arm.** The three runs below are training at temperature 0.7: the
`no_intervention` entrypoint in `scripts/run_rl_training.py` has a fixed signature with no keyword
passthrough, so `--temperature=0.5` never reached the config. Fire runs the function first and only
then rejects the leftover flag (exit 2), which is why the pods trained normally with `extra=--temperature=0.5`
in their banner and `'temperature': 0.7` in the composed config, and why each run will abort after
training under the pod script's `set -e`: the exit trap still pushes adapters and dumps, the eval is
skipped and can be re-run from the pushed adapters. The dry run and the Mac-side render could not
catch this because both stop short of the entrypoint. Fixed the same day: `rh-entrypoint-kwargs.patch` (on every
job) gives the entrypoints a `**kwargs` passthrough with a fail-fast on unknown config keys, and the
submitter's `check_extra_args` refuses any `--extra` key the patched entrypoint cannot take. The
temperature arm was resubmitted at five seeds at 11:15 UTC, after the seven baseline runs had finished:

| seed | OpenWeights job | run id |
|---|---|---|
| 1 | `rlrhrunjob-3bc79c9a8040-baseline-temp05` | `wong2025-baseline-temp05-s1-20260914_111451` |
| 2 | `rlrhrunjob-86d0f6f17f20-baseline-temp05` | `wong2025-baseline-temp05-s2-20260914_111457` |
| 3 | `rlrhrunjob-5ba3f969d1e0-baseline-temp05` | `wong2025-baseline-temp05-s3-20260914_111503` |
| 4 | `rlrhrunjob-f9b8d3f514fb-baseline-temp05` | `wong2025-baseline-temp05-s4-20260914_111508` |
| 5 | `rlrhrunjob-94ae4be60f20-baseline-temp05` | `wong2025-baseline-temp05-s5-20260914_111513` |

That second submission died at the first training step on every pod: `KeyError:
'sampling_input_ids'`. The passthrough patch pulls the recontextualisation patch onto every job,
and its `_reference_input` checked only the reference-context setting, which has defaulted to
`sampling` since 2026-09-11, so on an arm without recontextualisation it reached for tensors
nothing had stored. Seeds 1-2 failed on their own, 3-5 were cancelled before reaching step 1.
The patch now returns the batch untouched unless recontextualisation is on, with a structural
test, and the third submission went out at 11:45 UTC:

| seed | OpenWeights job | run id |
|---|---|---|
| 1 | `rlrhrunjob-e18dc6e289f4-baseline-temp05` | `wong2025-baseline-temp05-s1-20260914_114534` |
| 2 | `rlrhrunjob-5be3c8b71b2a-baseline-temp05` | `wong2025-baseline-temp05-s2-20260914_114540` |
| 3 | `rlrhrunjob-8167be4eb78a-baseline-temp05` | `wong2025-baseline-temp05-s3-20260914_114546` |
| 4 | `rlrhrunjob-3645b94fef76-baseline-temp05` | `wong2025-baseline-temp05-s4-20260914_114551` |
| 5 | `rlrhrunjob-d182ba3896ac-baseline-temp05` | `wong2025-baseline-temp05-s5-20260914_114556` |

Register as `temp05-s1..s5` once the wandb ids exist. A pod-side check reads each pod's composed
config for `'temperature': 0.5` and its first training step before anything else is believed.

What the three runs are instead: exact replicates of `jbase-s1..s3` (same orderings, same composed
config apart from the early-stop block, same first two batches step for step). They are kept
running as baseline replicates; see Results.

The frozen predictions below are void for these runs and stand for a resubmission.

Submitted 2026-09-14: three seeds of standard training (Neutral prompt, no intervention) on the
default parameters, intended at sampling temperature 0.5, `--early-stop 0.90` (the 5-batch-mean
rule, its first live run). Seed 3 queued at 06:51 UTC; seeds 1 and 2 failed to
upload at the same time because a 1.4 GB dump fetch was saturating the sandbox proxy (one
timeout, one 502, no job created), and went through at 07:00 UTC once the fetch finished.

| seed | OpenWeights job | run id (HF repo is `longtermrisk/rlrh-<run id>`) |
|---|---|---|
| 1 | `rlrhrunjob-22c29d46f3a4-baseline-t05` | `wong2025-baseline-t05-s1-20260914_065958` |
| 2 | `rlrhrunjob-a5cad006978d-baseline-t05` | `wong2025-baseline-t05-s2-20260914_070005` |
| 3 | `rlrhrunjob-ebe0d3d7cb94-baseline-t05` | `wong2025-baseline-t05-s3-20260914_065058` |

Registered in `tools/rlrh_runs.py` as `jbase-rep-s1..s3` (wandb `l0u1hlwz`, `xaumu49v`, `bkwj3pjk`),
since that is what they are; the HF repo names keep the `baseline-t05` label they were submitted
under. The resubmission uses the label `baseline-temp05` so the two cannot be confused.

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
for s in 1 2 3 4 5; do   # the first submission used --label baseline-t05 and seeds 1-3
  $OWPY tools/rlrh_job.py submit --arm no_intervention --label baseline-temp05 --seed $s --steps 200 \
    --early-stop 0.90 --extra temperature=0.5 \
    --patch rh-reward-metric-step.patch --patch rh-unparse-recursion-guard.patch
done
```

Chain, resolved by the submitter: `rh-reward-metric-step`, `rh-early-stop`,
`rh-unparse-recursion-guard`, `rh-jan2026-params`. Seeds 1-3 are data orderings A, B, C, the
same as `jbase-s1..s3`.

## Results

As baseline replicates, read against `jbase` on the same orderings (onsets 55, 93, 59). At the
last read (2026-09-14 09:46 UTC, from the pods' live logs; wandb `l0u1hlwz`, `xaumu49v`, `bkwj3pjk`):

| run | ordering | step | arb-pass ≥ 8 first at | paired `jbase` onset |
|---|---|---|---|---|
| `jbase-rep-s1` | A | 198, done | 158 by the pair metric; the share was still only 0.29-0.40 at 198, so the stop stayed silent | 55 |
| `jbase-rep-s2` | B | 198, done | never sustained; peak 10 of 256 in one batch, 2 batches at ≥ 8 | 93 |
| `jbase-rep-s3` | C | 199, done | never (max 3 of 256 in any batch); 121-169 correct at the end | 59 |

With `jbase-mem085-s1` (ordering A, onset 134 against 55) that is four replicates of the default
configuration, all 60-100 steps later than their 2026-09-11 pairs. Composed config, dataset,
model and the first two batches are identical, and the hosts do not split by date (`jbase-s2` and
today's runs on 192-vCPU nodes, `jbase-s1`/`s3` and the memory run on 96-vCPU nodes), so nothing
identifiable changed: the baseline onset distribution is much wider than the 009 triple showed,
and `measurement.md`'s run-to-run σ has to be re-estimated from all seven once these finish.

## Cost

Estimate $17 per seed the early stop ends, $33 per seed that runs to 200: $50-100 for the arm.
