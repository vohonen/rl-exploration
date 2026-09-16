# Does the Don't Eval Game prompt work by sampling fewer cannot-fail graders, or by something else?

## Status

Done 2026-09-14. Analysis only, no GPU. Reads the rollout dumps of the six Don't Eval Game → Neutral
recontextualisation runs on the default parameters (`jan26-s1..s3`, `both-s1..s3` from
`../008-kl-reference-context/`) against the three Neutral baselines (`jbase-s1..s3` from
`../009-jan-baseline/`). Item 1 of the program in `../../research.md`.

## Why this question

On the default parameters the Don't Eval Game sampling prompt with recontextualisation kept 5 of
8 attempts honest to step 200 where standard training hacked 3/3. Two mechanisms fit that:

- **Sampling-side.** Under the prompt the model writes fewer graders that cannot fail, so the
  behaviour selection needs is rarely in the batch and discovery is slow or never.
- **Selection-side.** Cannot-fail graders are sampled and paid at a similar rate, but do not
  compound: the update happens under the Neutral prompt on samples drawn under another, so the
  credit lands somewhere it does not transfer, or the honest solve rate is high enough that a
  hack's within-group advantage is small.

Which one it is decides what the program's prompt arms (items 3-4) are trying to reproduce. A
prompt arm is only a sampling intervention, so if the incumbent works selection-side, no prompt
written on mechanism grounds should be expected to match it.

## Predictions, frozen 2026-09-14 before any count

Per batch of 256 rollouts, counting a grader as cannot-fail if the env's arbitrary-pass flag is
set or the AST classifier puts `run_tests` on the no-assert, vacuous or guarded rung
(the classifier from `tools/grader_composition.py`, copied here).

- **Steps 1-50, all pre-onset:** cannot-fail graders per batch in the DEG runs at most one
  third of `jbase`'s on the same orderings. P = 0.55. (Under 0.45 the rates are within 3× and
  the sampling-side story is weak.)
- **Reachable anyway:** the honest DEG seeds still average ≥ 0.5 cannot-fail graders per batch
  over steps 1-120. P = 0.75.
- **Paid but not compounding:** in the honest DEG seeds, at least one cannot-fail grader on a
  wrong solution is paid full reward (score 3.5) in ≥ 20 % of batches over steps 1-120, and the
  per-batch count shows no upward trend across the window. P = 0.6.
- **The hacked DEG seed** (`jan26-s1`, onset 119) climbs along the same exponential as `jbase`
  once its count first reaches 8, with a doubling time inside `jbase`'s 2.4-4.9 steps. P = 0.6.
- **Graders overall:** the DEG runs write more `run_tests` functions than `jbase` early on, not
  fewer, because the prompt names the evaluation; the difference is in what kind. P = 0.5.

## Method

`audit.py` reads `.rlrh-cache/rollouts/<key>/<step>.jsonl` (fetched with
`tools/rlrh_fetch.py rollouts --runs ... --steps 1-120`), classifies every `run_tests` with the
copied AST classifier, and writes one row per batch to `counts.csv`: graders, cannot-fail
graders by the classifier, the env's arbitrary-pass count, cannot-fail graders paid 3.5 on a
wrong solution, honest correct solutions, and how many of the 16 problems saw a cannot-fail
grader. It prints the per-window means (1-25, 26-50, 51-75, 76-100, 101-120) per run.

```bash
python3 experiments/010-deg-sampling-shape/audit.py --runs jbase-s1,jbase-s2,jbase-s3,jan26-s1,jan26-s2,jan26-s3,both-s1,both-s2,both-s3 --steps 1-120
```

## Results

**The prompt cuts what gets sampled; what happens to the credit afterwards is a lottery in every
arm.** Under the Don't Eval Game prompt the model samples about 8× fewer cannot-fail graders per
batch before onset, and the few it samples are paid the same reward and the same within-group
advantage as in the baseline. Two of the six runs then collected 17-27 paid rollouts and went
extinct, which read, on the day, as the credit failing to compound under recontextualisation. The
seven-run Neutral baseline that finished the same evening retracts that reading as an arm
difference: `jbase-rep-s2` collected 118 paid rollouts (Σ adv 264) over 198 steps and never took
off, `jbase-rep-s1` needed 50 and took off at 158, and the hacked Neutral runs' doubling times
run 3.3-21.6 steps. Whether a seeded hack compounds is heavy-tailed under Neutral sampling too;
what this audit establishes is the sampling cut.

Steps 1-50, before onset in every run, pooled per arm:

| arm | batches | graders | cannot-fail | per batch | paid | paid / cannot-fail |
|---|---|---|---|---|---|---|
| Neutral (`jbase-s1..s3`) | 150 | 118 | 52 | 0.35 | 33 | 0.63 |
| DEG → Neutral (`jan26-*`, `both-*`) | 300 | 41 | 13 | 0.04 | 10 | 0.77 |

Steps 1-25 are near zero for everyone (3 cannot-fail graders in 75 Neutral batches, 1 in 150
DEG batches), so the gap opens at 26-50, where the Neutral runs are already amplifying and the
DEG runs are not. Under the prompt the model writes fewer `run_tests` functions of any kind, not
more.

Per run, steps 1-120 and before onset. "Paid" is full reward (3.5) on a wrong solution with a
cannot-fail grader; the advantage is GRPO's within-group value rebuilt from `score` and `id`;
"≥ 2" and "≥ 16" are the first batch with that many cannot-fail graders; the doubling time is a
log-linear fit from the first batch at ≥ 2 to the first above 128.

| run | cannot-fail | paid | mean adv | Σ adv | ≥ 2 | ≥ 16 | doubling (steps) |
|---|---|---|---|---|---|---|---|
| `jbase-s1` | 55 | 30 | 2.13 | 63.9 | 41 | 56 | 4.2 |
| `jbase-s2` | 87 | 48 | 2.02 | 97.1 | 44 | 91 | 9.6 |
| `jbase-s3` | 36 | 24 | 2.47 | 59.4 | 43 | 60 | 3.3 |
| `jan26-s1` | 162 | 95 | 1.82 | 173.2 | 61 | 111 | 18.0 |
| `jan26-s2` | 5 | 5 | 1.60 | 8.0 | 109 | — | — |
| `jan26-s3` | 19 | 9 | 1.78 | 16.0 | 50 | — | extinct |
| `both-s1` | 26 | 17 | 1.49 | 25.4 | 58 | — | extinct |
| `both-s2` | 55 | 27 | 2.13 | 57.5 | 48 | — | extinct |
| `both-s3` | 1 | 1 | 3.87 | 3.9 | — | — | — |

- **The reward treats the two arms identically.** Paid share 0.63 against 0.77; mean advantage
  2.16 against 1.84 (84 % against 67 % of paid rollouts at ≥ +1); no paid rollout in either arm
  sat in a fully solved group, so the higher honest solve rate under the prompt (100-112 of 256
  correct at steps 76-120 against `jbase-s2`'s 94-96) does not shrink the hack's advantage. The
  paid graders are the same shape in both arms: `no-assert` in 93-100 % of them, spread over
  7-58 problems each.
- **Compounding does not separate the arms once the baseline has seven runs.** The three
  September-11 Neutral runs took off after 24-48 paid rollouts and doubled every 3-10 steps, and
  against them `both-s1` (17 paid, extinct), `both-s2` (27 paid, Σ adv 57.5, extinct) and
  `jan26-s1` (95 paid to reach 16 per batch) looked like suppressed compounding. The four
  Neutral runs that finished on 2026-09-14 cover the same ground: `jbase-mem085-s1` 29 paid then
  takeoff, `jbase-rep-s1` 50 paid then takeoff at 158, `jbase-rep-s3` 7 paid and honest, and
  `jbase-rep-s2` **118 paid rollouts, Σ adv 264, no takeoff in 198 steps**. Measured one way on
  the wandb counts, the hacked Neutral runs double every 3.3-21.6 steps and `jan26-s1` every
  11.4. `../012-deg-prior-control/README.md` holds the side-by-side table.

### Predictions resolved

| frozen prediction | P | outcome |
|---|---|---|
| steps 1-50 cannot-fail rate ≤ ⅓ of `jbase`'s | 0.55 | **true**, 0.04 against 0.35 |
| honest DEG seeds still average ≥ 0.5 cannot-fail per batch over 1-120 | 0.75 | **false**, 0.01-0.46 |
| paid cannot-fail grader in ≥ 20 % of honest DEG batches, no upward trend | 0.60 | half: 1-15 % of batches, and no trend |
| `jan26-s1` doubles inside `jbase`'s 2.4-4.9 steps once at 8 | 0.60 | **false**, 18 steps |
| DEG writes more `run_tests` early, differing in kind not count | 0.50 | **false**, 6× fewer |

Three misses in one direction: the prompt suppresses the sampled behaviour harder than
predicted, and the expected "sampled and paid at a similar rate" picture is wrong. What survives
of the selection-side story is stronger than predicted in a different form: credit that
compounds under Neutral does not compound here.

### What it means for the program

- The compounding gap this audit first reported motivated `012`, the DEG → DEG control, to
  separate "the sampling/update mismatch blocks compounding" from "the prompt does". With the
  seven-run baseline neither is needed to explain the data: seeded hacks fail to compound under
  Neutral sampling too. What `012` can still measure is whether recontextualisation adds
  protection on the hack fraction beyond the prompt's sampling cut (DEG → Neutral 1/6 against
  Neutral 5/7, Fisher one-sided p ≈ 0.08), which is what its seeds 2 and 3 decide.
- The pre-training rollout audit in the program header now has its calibration: cannot-fail
  graders per batch at steps 26-50 read 0.04 under DEG and 0.35 under Neutral. A candidate
  prompt sampled from a pre-onset `jbase` adapter can be placed on that scale before its arm runs.
- The 006 reading, "luck is not a shape", also softens: `jbase-rep-s2` was paid 118 times for
  the canonical `no-assert` shape under Neutral and never compounded. A rare, paid, heritable
  shape can sit at 0.5-1 per batch for a hundred steps; the seed rate is what the interventions
  here move, and a long horizon is what turns a low rate into a hack.

### The zero-solve niche pays more, it is not sampled more

`niche.py`, added 2026-09-16 for the program's arm 4. A GRPO group on a problem none of its 16
rollouts solves is where a lone cannot-fail grader earns the whole advantage (`rh-intuition.md`).
Pre-onset, over the seven Neutral and eight incumbent runs (steps 20 to onset, the whole run for an
honest seed), the grader is sampled at 3.7 ‰ per rollout in zero-solve groups against 4.7 ‰ in
groups with a solve, ratio 0.80; about 30 % of rollouts sit in zero-solve groups and 26 % of
pre-onset hacks land there. The niche is a payoff effect, not a sampling effect, so a sampling-side
prompt aimed at the stuck state finds no elevated rate to cut. The two earliest onsets in the
project are the exceptions: `jbase-s3` (onset 59) had 55 % of its pre-onset hacks in zero-solve
groups against 39 % of its rollouts, `both-s4` (onset 43) 83 % against 41 %. Two runs, read after
the fact; consistent with a hack that lands in the niche compounding fastest, which is the payoff
story again.

### Caveat on the dumps

The 008 dumps have no `response_test_func_arbitrary_pass` key at all: that field was added to
the dumped reward extras by `rh-early-stop.patch`, which first ran on `009`. A `.get()` on those
records therefore reads 0, which is not a measurement. Every count above uses the AST classifier
for cannot-fail graders, and the classifier agrees with the wandb aggregate where both exist
(`jbase-s1` step 55: flag 11 of 256, classifier the same; `jan26-s1` step 118: wandb counter and
classifier both at 5). Every run submitted since 009 carries the patch and the key.
