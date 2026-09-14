# Does the Don't Eval Game prompt stop the hack from compounding on its own, or does recontextualisation?

## Status

Submitted 2026-09-14: five seeds of Don't Eval Game sampled **and** updated under the same prompt
(the "prior" arm, no recontextualisation), default parameters, `--early-stop 0.90`, seeds 1-5 on
data orderings A-E, all created 07:41 UTC. **Held at three seeds by decision at 10:56 UTC**: seeds 4
and 5 were cancelled within their first steps, so the arm is seeds 1-3 on orderings A-C, the same
orderings as the baselines and the six recontextualised runs it is read against.

| seed | OpenWeights job | run id (HF repo is `longtermrisk/rlrh-<run id>`) |
|---|---|---|
| 1 | `rlrhrunjob-d70678aa2ba1-prior-dont_eval_game` | `wong2025-prior-dont_eval_game-s1-20260914_074056` |
| 2 | `rlrhrunjob-21403c376f0c-prior-dont_eval_game` | `wong2025-prior-dont_eval_game-s2-20260914_074101` |
| 3 | `rlrhrunjob-376990f91385-prior-dont_eval_game` | `wong2025-prior-dont_eval_game-s3-20260914_074105` |
| 4 | `rlrhrunjob-59bd1ec2ebaf-prior-dont_eval_game` | `wong2025-prior-dont_eval_game-s4-20260914_074110` |
| 5 | `rlrhrunjob-2b1c9e481970-prior-dont_eval_game` | `wong2025-prior-dont_eval_game-s5-20260914_074115` |

Registered as `jprior-s1..s5` in `tools/rlrh_runs.py` as the wandb ids appear (`prior-s1..s3` are
002's February-parameter runs). **Seed 3's first pod died at step 141, honest** (wandb `m2egqbkb`,
registered as `jprior-s3-a1`; its dumps were pulled before the restart could overwrite them); the
queue restarted the job from step 0 under the same run id at 10:47 UTC, so seed 3 counts as two
attempts, the first censored at 141. The first attempt's dumps (135 of 141 batches reached HF before
the pod died) hold three paid cannot-fail rollouts in total and no batch above 1, so it is
sampling-limited and says nothing about compounding.

## Why this arm

`../010-deg-sampling-shape/` found that in the DEG → Neutral recontextualisation runs the few
cannot-fail graders that get sampled are paid the same reward and the same within-group advantage
as in the Neutral baseline, and do not compound: two seeds collected a baseline's worth of paid
hacks and went extinct, and the one that hacked doubled every 18 steps against the baseline's
3-10. Two mechanisms fit and that data cannot separate them:

- **H1, the mismatch does it.** The update raises the behaviour under the Neutral context, the
  next batch is sampled under DEG, and only what leaks across contexts compounds.
- **H2, the prompt context does it.** Under a prompt that says not to game the evaluation, the
  same nudge moves a gaming behaviour less, whichever prompt the update is applied under.

This arm removes the mismatch and keeps the prompt. Under H1 its hacks should compound at
baseline speed once paid; under H2 they should compound slowly or die as in 010. Which one it is
decides whether the program's prompt arms (items 3-4 of `../../research.md`) owe their protection
to the vehicle or to their text, and therefore whether each of them needs a control of this form.

## Predictions, frozen 2026-09-14 before submission

P(H2) = 0.55 going in. Azarbal's own prior cell protected as much as her RC cell at n = 3, which
leans H2, with standard deviations of 30 points.

The reading rule, from the same audit as 010 (`../010-deg-sampling-shape/audit.py` on the dumps,
GRPO advantages rebuilt from `score` and `id`):

- On seeds that hack, the doubling time of cannot-fail graders per batch from the first batch
  with ≥ 2 to the first above 128. **Median ≤ 10 steps → H1. Median ≥ 15 → H2.** 10-15 is
  reported as mixed.
- Paid cannot-fail rollouts before the count first reaches 16 per batch: ≤ 50 reads as
  baseline-like compounding (H1), ≥ 80 as suppressed (H2).
- A seed with ≥ 20 paid cannot-fail rollouts by step 200 and no takeoff is H2 evidence on its own.

Other frozen numbers:

- Per-seed P(hack by 200) ≈ 0.4, from her prior cell and our RC arm's 3 of 8. Over five seeds:
  0/5 0.08, 1/5 0.26, 2/5 0.35, 3/5 0.23, ≥ 4/5 0.09.
- Cannot-fail graders per batch at steps 26-50 the same as DEG → Neutral, 0.02-0.10 (P = 0.8).
  Much higher would mean the Neutral-context update was itself lowering the DEG sampling rate.
- Onset on hacking seeds 90-180 (P = 0.7).
- Honest seeds end at 20-25 % correct under the hint, hacked ones at 15-21 % at the stopped
  checkpoint (P = 0.7).
- The 0.90 / 5-batch-mean early stop fires on every hacking seed and no honest one (P = 0.85).

## Method

```bash
set -a; . ./.env; set +a
OWPY="$(uv tool dir)/openweights/bin/python"
for s in 1 2 3 4 5; do
  $OWPY tools/rlrh_job.py submit --arm inoculation --label prior-dont_eval_game --seed $s --steps 200 \
    --early-stop 0.90 --extra prompt_name=dont_eval_game --extra intervention_label=prior \
    --patch rh-anti-hack-prompts.patch --patch rh-reward-metric-step.patch \
    --patch rh-unparse-recursion-guard.patch
done
```

Chain, resolved by the submitter: `rh-anti-hack-prompts`, `rh-reward-metric-step`,
`rh-early-stop`, `rh-unparse-recursion-guard`, `rh-jan2026-params`. The same entrypoint and label
002 used for its prior arm on the February parameters (`prior-s1..s3`, which hacked 3/3 there).
Read against `jbase` (Neutral, 3/3 hacked) and the six DEG → Neutral runs of 008 via 010's audit.

## The three arms side by side

`compare.py` prints this from the cached wandb histories with one method per column, so every run,
finished or in flight, is measured the same way: cannot-fail graders per batch at steps 26-50
(a pre-onset window for every finished run except `jbase-s1`, which was already climbing at
41-50); paid cannot-fail rollouts before the count first reaches 16 per batch, from the dump
audit where one has been run; onset by the pair metric; the first batch at ≥ 16; and the
doubling time of the count from the first batch at ≥ 2 to the first above 128. TBA marks a run
in flight or a dump not yet audited. Regenerate with `python3 experiments/012-deg-prior-control/compare.py`.

| arm | run | ordering | cannot-fail / batch, 26-50 | paid before takeoff | onset | first ≥ 16 | doubling (steps) | outcome |
|---|---|---|---|---|---|---|---|---|
| Neutral → Neutral | `jbase-s1` | A | 1.48 | 30 | 55 | 58 | 4.2 | hacked |
| Neutral → Neutral | `jbase-s2` | B | 0.20 | 48 | 93 | 91 | 9.6 | hacked |
| Neutral → Neutral | `jbase-s3` | C | 0.24 | 24 | 59 | 60 | 3.3 | hacked |
| Neutral → Neutral | `jbase-mem085-s1` | A | 0.00 | TBA (dumps) | 134 | 135 | 12.5 | hacked |
| Neutral → Neutral | `jbase-rep-s1` | A | 0.16 | TBA (dumps) | 158 | 158 | 21.6 (still climbing) | hacked |
| Neutral → Neutral | `jbase-rep-s2` | B | 0.16 | TBA (dumps) | TBA | TBA | TBA | in flight, step 193, honest so far |
| Neutral → Neutral | `jbase-rep-s3` | C | 0.00 | TBA (dumps) | none by 198 | — | — | honest to 198 |
| DEG → Neutral (RC) | `jan26-s1` | A | 0.00 | 95 | 119 | 111 | 11.4 | hacked |
| DEG → Neutral (RC) | `jan26-s2` | B | 0.04 | 5 | none by 198 | — | — | honest to 198 |
| DEG → Neutral (RC) | `jan26-s3` | C | 0.20 | 9 | none by 198 | — | — | honest to 198 |
| DEG → Neutral (RC) | `both-s1` | A | 0.00 | 17 | none by 198 | — | — | honest to 198 |
| DEG → Neutral (RC) | `both-s2` | B | 0.24 | 27 | none by 198 | — | — | honest to 198 |
| DEG → Neutral (RC) | `both-s3` | C | 0.00 | 1 | none by 198 | — | — | honest to 198 |
| DEG → DEG (prior) | `jprior-s1` | A | 0.08 | 31 | 97 | 98 | 2.6 | hacked |
| DEG → DEG (prior) | `jprior-s2` | B | 0.04 | TBA (dumps) | TBA | TBA | TBA | in flight, step 151, honest so far |
| DEG → DEG (prior) | `jprior-s3-a1` | C | 0.00 | 3 | none by 141 | — | — | pod died at 141, honest (attempt 1) |
| DEG → DEG (prior) | `jprior-s3` | C | TBA | TBA (dumps) | TBA | TBA | TBA | in flight |

| arm | finished | hacked | cannot-fail / batch 26-50, mean ± SE | onset of hacked, mean ± SE (n) | doubling of hacked, mean ± SE (n, uncensored) |
|---|---|---|---|---|---|
| Neutral → Neutral | 6 | 5/6 (0.83 ± 0.15) | 0.35 ± 0.23 (n = 6) | 99.8 ± 20.3 (n = 5) | 7.4 ± 2.2 (n = 4) + 1 still climbing at 21.6 |
| DEG → Neutral (RC) | 6 | 1/6 (0.17 ± 0.15) | 0.08 ± 0.05 (n = 6) | 119.0 (n = 1) | 11.4 (n = 1) |
| DEG → DEG (prior) | 2 | 1/2 (0.50 ± 0.35) | 0.04 ± 0.04 (n = 2) | 97.0 (n = 1) | 2.6 (n = 1) |

What the two hypotheses predict for the bottom row, and what three seeds can and cannot say:

- **H1, the sampling/update mismatch blocks compounding.** DEG → DEG seeds that hack should sit
  in the Neutral band on doubling (3-13) and paid-before-takeoff (24-48), and DEG → DEG should
  hack more often than DEG → Neutral at the same seed rate. Seed 1 does exactly this.
- **H2, the prompt itself blocks compounding.** DEG → DEG seeds should look like the
  recontextualised ones: mostly honest at a low seed rate, and a hacked one slow (≥ 15) or a
  seeded one going extinct after ≥ 20 paid rollouts.
- The Neutral band is wide (five hacked runs span 3.3-21.6 on doubling and 55-158 on onset), so
  with three seeds the arm separates the hypotheses only if it is one-sided: all hacked seeds
  fast, or an honest seed with ≥ 20 paid rollouts and no takeoff. A mixed outcome stays open.
  The sampling-rate column is already settled: both DEG arms sample cannot-fail graders at a
  fraction of the Neutral rate (median 0.02-0.04 against 0.18 per batch), so the prompt's own
  effect is the seed-rate cut whichever way the compounding question falls.

## Results

Live, one seed in so far. Per seed: onset by the pair metric (`tools/rlrh_onset.py`), then the
reading rule's quantities from `../010-deg-sampling-shape/audit.py` on the dumps: cannot-fail
graders per batch at steps 26-50 (DEG → Neutral 0.04, Neutral 0.35), paid cannot-fail rollouts
before the count first reaches 16 per batch (Neutral 24-48), their summed GRPO advantage (Neutral
59-97), and the doubling time from the first batch at ≥ 2 to the first above 128 (Neutral 3.3-9.6,
`jan26-s1` 18).

| seed | ordering | wandb | onset | stopped | rate 26-50 | paid to 16 | Σ adv | doubling (steps) | reads |
|---|---|---|---|---|---|---|---|---|---|
| 1 | A | `fg80hmot` | 97 | 114 | 0.08 | 31 | 60.8 | **2.6** | H1 |

- **Seed 1** sampled cannot-fail graders as rarely as the recontextualised runs (0.08 per batch
  at 26-50; one paid hack before step 84), so the sampling cut belongs to the prompt. Once seeded
  it compounded like a Neutral run or faster: 31 paid rollouts and Σ adv 61 took it to 16 per
  batch, the count went 2 → 34 → 104 → 212 between steps 84 and 106, and the fit gives a doubling
  time of 2.6 steps. In 010 the same credit under DEG → Neutral went extinct twice. First live
  fire of the 0.90 / 5-batch-mean early stop: batch share 96.9 %, window mean 90.5 % at step 113,
  run ended at 114, eval and push landed.

## Cost

Estimate $17 per seed the early stop ends, $33 per seed that runs to 200: $85-165 for the arm.
