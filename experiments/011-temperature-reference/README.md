# Does colder sampling move when the hack arrives? Temperature 0.5 as the exploration reference

## Status

**Training at temperature 0.5, five seeds, since 2026-09-14 11:45 UTC.** Every pod's composed config
carries `'temperature': 0.5` in the rollout block, read off the live logs before anything else was
believed. Neutral prompt, default parameters, `--early-stop 0.90`, seeds 1-5 on orderings A-E (1-3
match `jbase-s1..s3`). Registered as `temp05-s1..s5` in `tools/rlrh_runs.py`.

| seed | OpenWeights job | run id (HF repo is `longtermrisk/rlrh-<run id>`) | wandb |
|---|---|---|---|
| 1 | `rlrhrunjob-e18dc6e289f4-baseline-temp05` | `wong2025-baseline-temp05-s1-20260914_114534` | `yddvkwtf` |
| 2 | `rlrhrunjob-5be3c8b71b2a-baseline-temp05` | `wong2025-baseline-temp05-s2-20260914_114540` | `t9vh217p` |
| 3 | `rlrhrunjob-8167be4eb78a-baseline-temp05` | `wong2025-baseline-temp05-s3-20260914_114546` | `t7pf0yqs` |
| 4 | `rlrhrunjob-3645b94fef76-baseline-temp05` | `wong2025-baseline-temp05-s4-20260914_114551` | `rzyebj9y` |
| 5 | `rlrhrunjob-d182ba3896ac-baseline-temp05` | `wong2025-baseline-temp05-s5-20260914_114556` | `cx8ekkg6` |

Two earlier submissions of this arm never trained at 0.5. Both are why two gates now exist: the
`rh-entrypoint-kwargs.patch` passthrough fails fast on an unknown config key, and the submitter's
`check_extra_args` refuses any `--extra` key the patched entrypoint cannot take.

- **First submission (label `baseline-t05`, three seeds, 07:00 UTC) trained at 0.7.** The
  `no_intervention` entrypoint had no keyword passthrough, so `--temperature=0.5` never reached
  the config; fire ran the function and only then rejected the leftover flag (exit 2), which also
  aborted the pod script after training, so the eval was skipped while the exit trap still pushed
  adapters and dumps. The dry run and the Mac-side render both stop short of the entrypoint. The
  three runs are exact replicates of `jbase-s1..s3` (same orderings, same composed config apart
  from the early-stop block, same first two batches step for step) and are kept as such:
  `jbase-rep-s1..s3`, wandb `l0u1hlwz`, `xaumu49v`, `bkwj3pjk`; the HF repos keep the
  `baseline-t05` label. Their evals can be re-run from the pushed adapters (~$5 each).
- **Second submission (label `baseline-temp05`, 11:15 UTC) died at step 1** with `KeyError:
  'sampling_input_ids'`. The passthrough patch pulls the recontextualisation patch onto every job,
  and its `_reference_input` checked only the reference-context setting, which has defaulted to
  `sampling` since 2026-09-11, so on an arm without recontextualisation it reached for tensors
  nothing had stored. Seeds 1-2 failed on their own, 3-5 were cancelled before step 1. The patch
  now returns the batch untouched unless recontextualisation is on, with a structural test.

The frozen predictions below were written for a three-seed arm; the arm became five seeds by
the 2026-09-14 rule (five per intervention) before any temperature run had produced a step, and
the predictions are read per seed rather than re-frozen.

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
for s in 1 2 3 4 5; do   # the two earlier submissions: --label baseline-t05 (seeds 1-3) and this label at 11:15 UTC
  $OWPY tools/rlrh_job.py submit --arm no_intervention --label baseline-temp05 --seed $s --steps 200 \
    --early-stop 0.90 --extra temperature=0.5 \
    --patch rh-reward-metric-step.patch --patch rh-unparse-recursion-guard.patch
done
```

Chain, resolved by the submitter: `rh-reward-metric-step`, `rh-early-stop`,
`rh-unparse-recursion-guard`, `rh-entrypoint-kwargs`, and through the last one's dependencies
`rh-anti-hack-prompts`, `rh-recontextualization`, `rh-runtime-prompts` (all inert on this arm),
then `rh-jan2026-params`. Seeds 1-5 are data orderings A-E.

## Results

The temperature-0.5 runs are training; nothing to read yet. Once they finish: `tools/rlrh_fetch.py
history --runs temp05-s1,...`, onset by the pair metric, hack fraction against the seven-run Neutral
baseline (5/7), correct % at the stopped checkpoint against other stopped runs only.

The three replicates from the first submission, read against `jbase` on the same orderings
(onsets 55, 93, 59); wandb `l0u1hlwz`, `xaumu49v`, `bkwj3pjk`:

| run | ordering | step | arb-pass ≥ 8 first at | paired `jbase` onset |
|---|---|---|---|---|
| `jbase-rep-s1` | A | 198, done | 158 by the pair metric; the share was still only 0.29-0.40 at 198, so the stop stayed silent | 55 |
| `jbase-rep-s2` | B | 198, done | never sustained; peak 10 of 256 in one batch, 2 batches at ≥ 8 | 93 |
| `jbase-rep-s3` | C | 199, done | never (max 3 of 256 in any batch); 121-169 correct at the end | 59 |

With `jbase-mem085-s1` (ordering A, onset 134 against 55) that is four replicates of the default
configuration, all 60-100 steps later than their 2026-09-11 pairs. Composed config, dataset,
model and the first two batches are identical, and the hosts do not split by date (`jbase-s2` and
today's runs on 192-vCPU nodes, `jbase-s1`/`s3` and the memory run on 96-vCPU nodes), so nothing
identifiable changed: the baseline onset distribution is much wider than the 009 triple showed.
`measurement.md`'s seed section is now built on all seven (5/7 hacked, SD ≈ 60 on onset).

## Cost

Estimate $17 per seed the early stop ends, $33 per seed that runs to 200: $50-100 for the arm.
