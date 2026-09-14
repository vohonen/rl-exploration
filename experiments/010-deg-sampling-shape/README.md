# Does the Don't Eval Game prompt work by sampling fewer cannot-fail graders, or by something else?

## Status

Analysis only, no GPU. Reads the rollout dumps of the six Don't Eval Game → Neutral
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

Pending.
