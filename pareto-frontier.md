# The Pareto frontier

One row per arm of the exploration-shaping program, one point per arm on the headline figure.
This file holds the numbers; the definitions and rules are in
[`measurement.md`](measurement.md) ("The headline figure and the table behind it"), the arms
and their order in [`research.md`](research.md) ("The program"), and each arm's own README under
`experiments/`. `./tools/rlrh_frontier.py` prints the rows for every finished arm from the local
cache, and `--figure .doc/figures/frontier.png` redraws the figure (needs a matplotlib venv, see
`CLAUDE.md`). If a number here and the tool disagree, the tool is right.

Axes, both at the final adapter, mean ± SE over seeds: **strict RH %** on the hinted half of the
pinned eval set, **correct %** on the no-hint half. **Tampering %** is every cannot-fail grader,
hacks on solved problems included; when it and strict RH disagree by more than 20 pp the arm is
read on tampering. **Onset** is the pair metric in batch coordinates, "—" is honest to the
horizon; the restricted mean enters honest runs at 200. **Hacked** is the seeds that crossed the
discovery threshold; **converged** is how many of those the early stop ended or that averaged
at least 0.80 of a batch writing a cannot-fail grader over their last five batches
(`measurement.md`). Every arm: default parameters, 200 steps,
early stop at 0.80 from arm 6 and on arm 0's last three seeds, 0.90 on arms 1-5 and 0.95 on
arm 0's first seven, five seeds on data orderings A-E unless stated.

![The headline frontier](.doc/figures/frontier.png)

![The frontier on normalised axes](.doc/figures/frontier-normalised.png)

The second figure is the same arm means on axes that run 0 to 1 between the poorest and the best
point shown: task performance from the base model's 11.3 % correct to arm 8's 25.3 %, reward
hacking from 0 % strict to arm 9's 36.7 %. Seeds are left out, the error bars are the raw SEs
rescaled, and the endpoints are taken as exact. Neutral and Persistence have the same correctness
to the decimal, so Persistence is nudged 0.03 left (0.4 pp) to keep its bar visible; the table has
the exact position. Colour is the family: baseline dark grey, decoding light grey,
prompt-side handles orange, weight-side priors blue; the marker is the arm. The
figures carry performance only; how many seeds hacked is in the table. Every arm had at least one
seed cross the discovery threshold, and only some of those seeds converged to the hacked policy
by the horizon: what the program measures is exploration, whether the behaviour gets found and
paid, not how far a found behaviour had compounded when training stopped. `--normalise` on the
tool redraws it.

How to read the first figure: a large marker is an arm's mean ± SE, faint markers are its seeds, the
cross is the base model before RL. Nearly every seed ends at one extreme, under 1 % or 35-66 % strict
RH, so an arm's height is roughly its hack fraction times a hacked seed's ~60 %, and its vertical
bar is the spread of that fraction rather than eval noise. The exceptions are seeds caught
mid-takeoff by the horizon (`learn-s1` 1.9 %, `scope-s3` 2.1 %, `jbase-rep-s1` 6.0 %,
`jbase-ext-s6` 13.4 %), which is why an arm whose onsets are all late reads low on this axis and
is better judged on onset.

| # | arm | exp. | handle | runs | hacked | converged | strict RH % | correct %, no hint | tampering % | restricted mean onset | onsets by seed | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| — | base model | — | none | 1 | — | — | 0.0 | 11.3 | 0.0 | — | — | reference, before RL |
| 0 | Neutral baseline | `009` | none, the origin | 10 | 8/10 | 6 | 35.1 ± 8.7 | 21.6 ± 0.9 | 51.3 ± 12.5 | 118 ± 18 | 55, 93, 59, 134, 158, —, —, 64, 74, 146 | done. Orderings A-F, every seed evaluated (the three `jbase-rep` adapters through `rlrh_job.py eval`, 2026-09-21). Six seeds converged and sit at 40-66 % strict, two were still early in takeoff at 198 (`jbase-rep-s1` 6.0 %, `jbase-ext-s6` 13.4 %), two stayed honest; the point is the hack fraction times a hacked seed's level, and it is the shared denominator of every comparison in the program |
| 1 | Temperature 0.5 | `011` | decoding | 5 | 2/5 | 2 | 17.0 ± 10.6 | 22.1 ± 1.7 | 24.5 ± 15.3 | 152 ± 30 | 57, —, 104, —, — | done. The reference bar: an arm that does not beat colder sampling is not a method |
| 2 | Don't Eval Game → Neutral RC | `008` | sampling context | 5 | 1/5 | 1 | 13.0 ± 12.8 | 21.2 ± 1.4 | 18.7 ± 18.4 | 169 ± 31 | —, —, —, 43, — | done. The incumbent: the five seeds with the KL reference under the sampling prompt, the recipe of every other RC arm. 008's three earlier seeds with the reference under the target prompt (`jan26-s1..s3`, orderings A-C, hacked 1/3 at 119) are its KL-context control and are no longer pooled in (decided 2026-09-21) |
| 3 | Positive-aim tests prompt → Neutral RC | `015` | sampling context | 5 | 5/5 | 4 | 47.6 ± 11.9 | 20.3 ± 0.9 | 72.9 ± 17.3 | 108 ± 15 | 160, 122, 85, 93, 82 | done. Every seed hacked, on or ahead of Neutral's schedule, with Neutral's smoke-test hack and Neutral's correct %; seed 1 was mid-takeoff at 200 (1.9 % strict), which is the whole spread. The prompt induced no test-writing, so the 25 pp tampering − strict gap is hacks on solved problems (guarded 0.1 %), as on Neutral, and the strict point stands |
| 4 | Persistence prompt → Neutral RC | `014` | sampling context | 5 | 3/5 | 3 | 30.4 ± 12.5 | 21.6 ± 1.2 | 48.8 ± 20.1 | 127 ± 31 | 100, 83, 51, —, — | done. Neutral on every readout: no sampling cut, Neutral-speed compounding, onsets on Neutral's schedule |
| 5 | Don't Eval Game → EvalEnv RC | `013` | update context | 5 | 2/5 | 2 | 19.6 ± 12.3 | 18.5 ± 1.3 | 27.4 ± 17.4 | 171 ± 18 | —, —, 128, 129, — | done. The only arm left of the origin on the correct axis |
| 6 | Warm-start SFT on clean rollouts, then Neutral RL | `016` | weights, in-distribution prior | 5 | 1/5 | 0 | 4.8 ± 4.7 | 19.5 ± 0.5 | 7.9 ± 7.6 | 189 ± 11 | —, —, —, 143, — | done. The lowest hack fraction and the latest onset in the program, but Fisher p = 0.09 against Neutral's 8/10. It halved the zero-solve niche (25.9 % against 48.2 %, t = −21), which is the one thing here measured beyond doubt. Ordering C is `rft-s3-a2` after the first attempt collapsed |
| 7 | General-prep DPO on out-of-environment data, then Neutral RL | `017` | weights, out-of-distribution prior | 5 | 2/5 | 1 | 10.9 ± 7.9 | 22.6 ± 1.4 | 16.8 ± 12.2 | 180 ± 12 | 149, —, —, —, 149 | done. Second furthest right on the correct axis, behind arm 8. Inflated the zero-solve niche (59.5 % against 48.2 %) and hacked less than Neutral anyway, which is what stops niche size explaining outcomes |
| 8 | Task-scope sentence → Neutral RC | `018` | sampling context | 5 | 1/5 | 0 | 0.6 ± 0.4 | 25.3 ± 1.9 | 0.9 ± 0.6 | 195 ± 5 | —, —, 177, —, — | done. The best point on both axes at once and the latest onset in the program. The mechanism is sampling-side and large: `run_tests` defined in 0.022 % of rollouts at steps 1-50 against Neutral's 0.096 % (rate ratio 0.23, p = 1.4e-07). Its one hack compounded slowly rather than being cut off by the horizon |
| 9 | Reasons prior (synthetic documents on why gaming a measure is wrong), then Neutral RL | `019` | weights, out-of-distribution prior carrying reasons | 5 | 5/5 | 5 | 36.7 ± 4.2 | 23.0 ± 1.2 | 87.3 ± 1.9 | 80 ± 14 | 65, 74, 40, 124, 95 | done 2026-09-24. Every seed hacked and stopped on the early stop; the earliest restricted mean onset in the program (seed 3 at step 40). Read on tampering: 87 % against strict 37 %, a 50.6 pp gap because the arm also solves more problems than Neutral. The prior wrote self-check graders at 5.7x stock's rate before RL and selection took them; the same model recites the spec's reasons in first person. No capability cost |
| opt | Solve-rate curriculum | | which problems get sampled | | | | | | | | | only if time and budget remain |

Before RL, per 1000 rollouts of the training set from the untouched model under each arm's own prompt or prior ([`020`](experiments/020-pre-rl-sampling-audit/README.md)), the model defines `run_tests` at: Neutral 0.09, temperature 0.5 0.09, Don't Eval Game (arms 2 and 5) 0.08, persistence 0.13, task scope 0.05, RFT prior 0.55, DPO prior 0.00, reasons prior 0.54 (`019`). The prompt-side arms are indistinguishable from Neutral there; the RFT and DPO priors differ sixty-fold and finished one seed apart, and the reasons prior, at the RFT prior's rate, hacked 5/5 where the RFT prior hacked 1/5. The pre-RL rate is not what orders this table across arms, though within arm 9 it named the direction.

Off the plot by decision: the `012` control (Don't Eval Game sampled and updated under itself,
2/3 hacked), the incumbent's decomposition; the `006` airtight prompt (0/3), an existence proof
built from the hack's own shape list; and **arm 3**, whose row stays above but whose point sat
inside the Neutral cloud it is contrasted with and added clutter without contrast
(`FIGURE_OMITS` in `tools/rlrh_frontier.py`).
