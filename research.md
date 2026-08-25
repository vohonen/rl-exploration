# What does RL teach, and how much of that is decided by exploration rather than by reward?

## Status

The environment is reproduced and closed out. Four 200-step runs exist: a baseline, two
recontextualisation seeds and one inoculation arm. Recontextualisation is built and works:
`patches/rh-recontextualization.patch`, tested on CPU and confirmed on a GPU by a 10-step canary.

**Read [`onset-model.md`](onset-model.md) before planning a run.** It holds the cross-run model of
what sets the step at which the hack is discovered, and two corrections that change how every run
here should be read: `frac_adv_zero` does not measure advantages, and `--seed` moves the data
ordering as well as the sampling randomness.

**002's control did not reproduce, at either seed.** `dont_eval_game -> neutral` came back looking
like standard training rather than the predicted 0.0 ± 0.0 hacking, at seed 1 and again at seed 2.
The two seeds differ sharply in _when_ they hack — onset at step 63 and step 115 — though the seeds
carry different data orderings, so that gap is not yet attributable to the intervention. The useful
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
| [`002-prompt-conditioning-ladder`](experiments/002-prompt-conditioning-ladder/)         | control ran twice, neither reproduced; onset differs by 52 steps between the two seeds |
| [`003-inoculation-conditionalisation`](experiments/003-inoculation-conditionalisation/) | first arm run and evaluated; numbers not yet read                                      |

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
draw. The first arm, `eval_environment`, ran to 200 steps and both eval conditions are on
HuggingFace; the numbers have deliberately not been looked at, so every prediction in its README
predates the data it ranges over. What the rollouts already show is that the hack arrives ~25 steps
earlier than baseline on four independent markers and truncates the honest-learning phase rather
than skipping it — a single-seed data point for the time-to-onset endpoint 002 is being rebuilt
around.

## Queue

**Before any of this: read 003's numbers.** Both eval conditions are on HuggingFace and the
analysis is two offline commands, so this costs attention and nothing else. It also decides how
much weight the burn-in hypothesis below can carry, since that currently rests on three runs and no
eval.

Then, ordered, and only the first two are settled.

1. **Rebuild 002 around time to first onset**, measured from rollout dumps we already write, rather
   than around the hacking rate at step 200. A binary endpoint costs $20 for one bit and needs ~40
   seeds an arm to resolve the effect we care about; a censored survival time uses the whole run
   and needs single digits. Nothing has to be re-run to start — the baseline's dumps and the
   control's are both on HuggingFace.
2. **Seed variance on the baseline.** One run is one sample and the step-85-to-100 transition is
   sharp enough that its timing could move a lot. This is not curiosity: every intervention below
   is being read at n=1 against it, so without a second baseline seed we cannot say what size of
   difference is worth believing. The onset-time framing makes this cheaper to satisfy than it was.
3. **SFT warm-start**, if 002 comes back saying the prompt has to name the failure mode. The idea
   is to raise the honest success rate before the hack is found, so that honest groups have
   advantage variance to learn from during the window where the model is still exploring. This
   attacks handle 2 rather than handle 3 and is the natural fallback if conditioning alone cannot
   do it.

**Do interventions only reschedule onset? Yes, on four runs — see
[`onset-model.md`](onset-model.md).** Every run hacks eventually; what the prompt moves is when.
Onset, defined as the first step with ≥8 of 256 successful hacks sustained five steps, lands at 41
for 003's inoculation arm, 63 for the baseline, 63 for 002's first recontextualised control and 115
for its second. Holding the data ordering fixed — the three seed-1 runs draw identical batches at every step —
inoculation pulls onset 22 steps earlier and recontextualisation moves it by zero. The 115 comes
with a different ordering attached, so a baseline at seed 2 is what decides whether that delay
belongs to the intervention.

The burn-in is **not** capability. Honest solves hold flat near 29% of rollouts from step 5 to
onset with no trend, then collapse to zero within ~20 steps. What does move monotonically before
onset is policy entropy and response length, which is what `onset-model.md` builds on. That doc
holds the model and its predictions; the reason it is a separate file is that it spans all four
runs rather than belonging to any one experiment.

**How much of Table 17 is Bernoulli noise?** Our one seed moved P(recontextualisation beats a prior
prompt at the category rung) from 0.79 to 0.60, which is to say the published effect at that rung
was never far from nothing. Several cells have standard deviations between 12 and 38 over three
seeds, and those are the ones an n=1 replication cannot speak to in either direction. Before
predicting against any of them, check which column of the table in 002 the cell sits in.

**Undecided: whether to stay in this environment.** Everything above is one loophole in one
codebase, and a result about it is a result about `run_tests`. Getting to a claim about exploration
in general needs a second environment where the hack has a different shape. Nothing is chosen and
nothing should be until 002 says whether the effect is there at all.

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
