# What does RL teach, and how much of that is decided by exploration rather than by reward?

## Status

The environment is reproduced and closed out. **Five completed 200-step runs exist and a sixth is in
flight**: two baselines, two recontextualisation seeds and one inoculation arm, with a seed-2
baseline running as [`004`](experiments/004-baseline-seed-variance/). Four of the five are evaluated
on the same held-out draw, so those arms have an endpoint and not just a training curve.
Recontextualisation is built and works: `patches/rh-recontextualization.patch`, tested on CPU and
confirmed on a GPU by a 10-step canary.

**The fifth run is the 2026-08-18 reproduction, and it had been left out of the analysis.** Its pod
was swept before anything was pushed, so its adapters, rollouts and evals are gone and it can never
join the eval table — but its wandb history survived complete, and that is enough for onset and the
training curves. It is a configuration-identical repeat of the 08-20 baseline: same env commit, same
`data.seed=1`, the same 397 pinned package versions, and the same data ordering, verified from the
per-step honest-solve counts. **It onsets at step 83 where 08-20 onsets at 63.** That 20-step gap on
identical configurations is the first error bar this project has had, and it is large enough to
change how the intervention arms read — details in [`onset-model.md`](onset-model.md), which the run
also cost its load-bearing prediction.

**Read [`onset-model.md`](onset-model.md) before planning a run.** It holds the cross-run model of
what sets the step at which the hack is discovered, and three corrections that change how every run
here should be read: `frac_adv_zero` does not measure advantages, `--seed` moves the data ordering as
well as the sampling randomness, and onset itself moves 20 steps between identical runs.

**002's control did not reproduce, at either seed.** `dont_eval_game -> neutral` came back looking
like standard training rather than the predicted 0.0 ± 0.0 hacking, at seed 1 and again at seed 2.
The two seeds differ sharply in _when_ they hack — onset at step 63 and step 115 — but the seeds
carry different data orderings, and 004 has now shown that gap belongs to the ordering: a *baseline*
at seed 2 has not onset by step 93 either. The useful
part of the result is not the number but what it exposed: the cell was chosen because it read
0.0 ± 0.0 over three seeds,
and a zero standard deviation over three near-Bernoulli draws is not evidence of a stable cell. The
ladder runs are on hold pending a redesign around time-to-onset rather than final hacking rate. The
implementation is not the problem: a step-1 comparison against a prior run on identical rollouts
shows the backward pass genuinely conditioning on the target prompt.

There is no external write-up doc yet. When one exists, its link goes in `CLAUDE.md` and this file
shrinks to status plus pointers rather than being kept in parallel.

## Tl;dr

RL changes a model only through the trajectories it samples. Reward decides which of those
trajectories get reinforced, but it cannot reinforce a behaviour that was never sampled. So the
lever that decides _what a model learns_ may be exploration rather than reward shaping. Reward
hacking is the cleanest place to test that, because the hack is a discrete behaviour with a known
onset: we can watch the moment it enters the sampling distribution and ask what stops it entering.

## Problem statement

The usual response to a model learning something bad from RL is to change the reward. In this
environment that route is closed — widening the gap between hack and honest changes nothing — and
the reason shapes every experiment here. See `running-the-env.md`'s "How reward and advantage
actually work" for why.

What is left is the sampling distribution, which `notes` breaks into four handles:

1. **The model** — a pretrained-in behaviour changes which trajectories are reachable at all.
2. **The trajectory** — warm-start SFT, or heavy conditioning during sampling only.
3. **Conditionalisation** — what the system prompt says, and whether it says the same thing at
   sampling time and at gradient time.
4. **The reward** — which trajectories are positively, negatively, or not reinforced.

Handle 3 is where the published work is, and where we start, because it is the only one that
changes nothing about the model, the data, or the code path. Whatever separates two runs that
differ only in a system prompt is a statement about exploration.

The deployment-relevant version of the question: **every published intervention that works in this
environment names the exploit, or at least the failure category.** If suppression requires naming
the thing you are defending against, none of it transfers to a loophole nobody thought of — and a
missed loophole gets read as malevolence rather than as an unpatched specification.

## Experiments

|                                                                                         |                                                                                        |
| --------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| [`001-baseline-generalisation`](experiments/001-baseline-generalisation/)               | done                                                                                   |
| [`002-prompt-conditioning-ladder`](experiments/002-prompt-conditioning-ladder/)         | control ran twice, neither reproduced; evals read, both seeds end at baseline hacking  |
| [`003-inoculation-conditionalisation`](experiments/003-inoculation-conditionalisation/) | first arm run, evaluated and read; inoculation conditioned nothing and cost capability |
| [`004-baseline-seed-variance`](experiments/004-baseline-seed-variance/)               | running; primary endpoint already resolved — ordering carried `rc-s2`'s delay, not RC  |

**001** asks what the baseline run actually learned, using the archived adapters against held-out
prompts. Answer: the hack generalises by mechanism rather than by surface form — it transfers to
unseen problems, to rewordings of the loophole and to a grader name it never saw, but not to a
prompt with no loophole in it. Genuine ability is not lost. Two consequences constrain everything
below: an intervention that only changes wording should be expected to fail, and a run's final step
is the most-hacked state it reaches, so that is what gets evaluated.

**002** asks how specifically a prompt has to name the loophole before it stops the model learning
to hack, and whether recontextualisation — sampling under a prompt, then taking the gradient step
as though the prompt had been neutral — removes that requirement. Its control arm ran and did not
reproduce the published cell, and the implementation is not why: the backward pass demonstrably
conditions on the target prompt. What the run actually established is that the cell was unreadable
at one seed and should never have been chosen. Its README carries the published ladder, which cells
can survive an n=1 check, and the redesign around time-to-onset.

**003** asks whether inoculation prompting — training under a prompt that asks for the hack — makes
the hack conditional on that prompt rather than merely rarer. Published work answers the rate
question at a variance one seed cannot beat, so 003 measures the conditional structure instead:
the same checkpoint evaluated under its own training prompt and under a neutral one, on the same
draw. **The answer is that nothing became conditional**: 100.0% hacking under the training prompt
against 96.8% under a neutral one, and held-out coding ability ends 6.6 pp *below the base model*
rather than merely frozen. Four of five pre-registered predictions missed, including one held at
90%. What the rollouts already showed is that the hack arrives ~25 steps earlier than baseline on
four independent markers and truncates the honest-learning phase rather than skipping it — a
single-seed data point for the time-to-onset endpoint 002 is being rebuilt around.

## Queue

Four of the five completed runs are evaluated on the same held-out draw, so what follows is ordered
against a near-complete set of endpoints rather than against training curves. Only the first two are
settled.

1. **Rebuild 002 around time to first onset**, measured from rollout dumps we already write, rather
   than around the hacking rate at step 200. A binary endpoint costs $20 for one bit and needs ~40
   seeds an arm to resolve the effect we care about; a censored survival time uses the whole run
   and needs single digits. Nothing has to be re-run to start — the baseline's dumps and both
   controls' are on HuggingFace. **This moved up in value.** A single onset step now has a measured
   20-step run-to-run range on it, so reading one number per arm was always going to be
   underpowered; a survival time over 16 problems per step uses roughly 200× more of each run.
2. **Seed variance on the baseline — running now**, as
   [`004`](experiments/004-baseline-seed-variance/). Every intervention is being read at n=1, so
   without knowing how much onset moves for free we cannot say what size of difference is worth
   believing. Part of that was answered without a pod: the two ordering-A baselines put the
   run-to-run range at 20 steps. **And 004's primary endpoint has now resolved in flight** — through
   step 93 the seed-2 baseline has not onset, so $T \ge 94$ and a baseline is slow on ordering B
   too. Most of `rc-s2`'s 52 steps was its problem sequence. Recontextualisation is now without
   surviving evidence of delaying onset on either ordering, which is the second independent way this
   project has failed to reproduce that cell.
3. **SFT warm-start**, if 002 comes back saying the prompt has to name the failure mode. The idea
   is to raise the honest success rate before the hack is found, so that honest groups have
   advantage variance to learn from during the window where the model is still exploring. This
   attacks handle 2 rather than handle 3 and is the natural fallback if conditioning alone cannot
   do it.

**Do interventions only reschedule onset? Yes on five runs, but the schedule is noisier than the
interventions — see [`onset-model.md`](onset-model.md).** Every run hacks eventually; what the
prompt moves is when. Onset, defined as the first step with ≥8 of 256 successful hacks sustained
five steps, lands at 41 for 003's inoculation arm, 63 and 83 for the two baselines, 63 for 002's
first recontextualised control and 115 for its second. All four seed-1 runs draw identical batches
at every step.

The two baselines are the pair that matters, because they are the same condition on the same
ordering and they land 20 steps apart. Against a baseline arm mean of 73 that is a standard error of
17 steps on one degree of freedom, and **no onset comparison in the project clears it**:
recontextualisation is at t = −0.6, inoculation at t = −1.9, and `rc-s2` at t = +2.4 — and that last
one was confounded with its data ordering, which 004 has now resolved against the intervention.

That is a verdict on the design, not on the interventions. The `rc-s1` null gets better out of it —
"no difference" becomes "any difference is under 20 steps", which is falsifiable. And 003's
inoculation result survives because it never rested on onset alone: the same ~25-step shift shows up
on four independent markers and the capability consequence then showed up in a held-out eval.
Reading a single onset step per arm was always going to be underpowered, which is the argument for
002's redesign around survival times.

**And a delay is all it is.** The held-out evals put the seed-2 run at 74.6% hacking at step 200,
against the baseline's 77.3% (−2.7 pp paired, CI straddling zero) and the seed-1 run's 84.8%.
Fifty-two extra steps before the hack was discovered left no trace in the terminal state, which is
the clearest argument yet for measuring onset rather than the endpoint. The two baselines say the
same thing from the other direction: 20 steps apart in onset, 4.9 pp apart at the endpoint. Where a
run *ends up* is the reproducible part; *when* it gets there is not.

The burn-in is not the model learning to program, but it is not featureless either — that claim was
too strong and the fifth run corrected it. Honest solves drift up by 0.17-0.26 percentage points per
step from step 5 to onset in every run, then collapse to zero within ~20 steps. The ramp is real; it
just cannot be what sets onset, because it is nearly identical across runs whose onsets differ by a
factor of three, and the run with the steepest ramp is the *latest* to hack. Entropy and response
length also move monotonically before onset. `onset-model.md` holds the model and its predictions;
the reason it is a separate file is that it spans all five runs rather than belonging to any one
experiment.

**How much of Table 17 is Bernoulli noise?** Our two seeds moved P(recontextualisation beats a
prior prompt at the category rung) from 0.79 to **0.45** — a coin flip, from two runs against their
three. The published effect at that rung was never far from nothing. Several cells have standard
deviations between 12 and 38 over three seeds, and those are the ones a small replication cannot
speak to in either direction. Before predicting against any of them, check which column of the
table in 002 the cell sits in.

**Undecided: whether to stay in this environment.** Everything above is one loophole in one
codebase, and a result about it is a result about `run_tests`. Getting to a claim about exploration
in general needs a second environment where the hack has a different shape. Nothing is chosen and
nothing should be until 002's rung-below-the-ladder arms run. What the control has settled is
narrower: at the one cell we checked, twice, the published effect is not reproducible here.

## Open questions

- ~~Is there ever much policy gradient at all?~~ **Answered, and the question was an artefact.**
  It rested on `frac_adv_zero`, which does not measure advantages — it counts responses shorter than
  the length cap. See `running-the-env.md`, "`frac_adv_zero` measures response length, not
  advantages". Counting groups with reward spread in the rollout dumps instead: ~10 of 16 groups
  are informative before onset and ~9 through step 100, so most rollouts do receive advantage for
  most of a run, and supply reaches zero only near step 150. Nothing here needs reconciling.
- **Does a positive prompt behave differently from a negated one at the same specificity?** All
  three published anti-hack prompts are negations. Negation neglect is real enough that "do not
  game the evaluation" and "take the evaluation seriously" might not be the same intervention, and
  the ladder in 002 cannot separate the two.
