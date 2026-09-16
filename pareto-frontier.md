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
horizon; the restricted mean enters honest runs at 200. Every arm: default parameters, 200 steps,
early stop at 0.90, five seeds on data orderings A-E unless stated.

![The headline frontier](.doc/figures/frontier.png)

How to read the figure: a large marker is an arm's mean ± SE, faint markers are its seeds, the
cross is the base model before RL. Every seed ends at one extreme, 0-1 % or 35-66 % strict RH,
so an arm's height is its hack fraction times a hacked seed's ~60 %, and its vertical bar is the
spread of that fraction, not eval noise.

| # | arm | exp. | handle | runs | hacked | strict RH % | correct %, no hint | tampering % | restricted mean onset | onsets by seed | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| — | base model | — | none | 1 | — | 0.0 | 11.3 | 0.0 | — | — | reference, before RL |
| 0 | Neutral baseline | `009` | none, the origin | 7 (4 evaluated) | 5/7 | 61.0 ± 3.6 | 20.3 ± 1.3 | 86.3 ± 6.5 | 128 ± 23 | 55, 93, 59, 134, 158, —, — | done. The point rests on the four hacked seeds; `jbase-rep-s1..s3` have adapters and no eval, so it reads too high |
| 1 | Temperature 0.5 | `011` | decoding | 5 | 2/5 | 17.0 ± 10.6 | 22.1 ± 1.7 | 24.5 ± 15.3 | 152 ± 30 | 57, —, 104, —, — | done. The reference bar: an arm that does not beat colder sampling is not a method |
| 2 | Don't Eval Game → Neutral RC | `008` | sampling context | 8 | 2/8 | 16.2 ± 10.5 | 21.6 ± 0.9 | 23.6 ± 15.3 | 170 ± 21 | 119, —, — (A-C); —, —, —, 43, — (A-E) | done. The incumbent |
| 3 | Positive-aim tests prompt → Neutral RC | `015` | sampling context | 5 | 1/5 so far | | | | | —, —, —, 93, — | all five training since 2026-09-16 14:30 UTC, seed 4 hacked at 93; read on tampering |
| 4 | Persistence prompt → Neutral RC | `014` | sampling context | 5 | 3/5 | 30.4 ± 12.5 | 21.6 ± 1.2 | 48.8 ± 20.1 | 127 ± 31 | 100, 83, 51, —, — | done. Neutral on every readout: no sampling cut, Neutral-speed compounding, onsets on Neutral's schedule |
| 5 | Don't Eval Game → EvalEnv RC | `013` | update context | 5 | 2/5 | 19.6 ± 12.3 | 18.5 ± 1.3 | 27.4 ± 17.4 | 171 ± 18 | —, —, 128, 129, — | done. The only arm left of the origin on the correct axis |
| 6 | Warm-start SFT on clean rollouts, then Neutral RL | | weights, in-distribution prior | | | | | | | | not run; data source and SFT plumbing undecided, one-seed pilot first |
| 7 | General-prep DPO on out-of-environment data, then Neutral RL | | weights, out-of-distribution prior | | | | | | | | not run; dataset not built, one-seed pilot first |
| opt | Solve-rate curriculum | | which problems get sampled | | | | | | | | only if time and budget remain |

Off the plot by decision: the `012` control (Don't Eval Game sampled and updated under itself,
2/3 hacked), the incumbent's decomposition; and the `006` airtight prompt (0/3), an existence
proof built from the hack's own shape list.
