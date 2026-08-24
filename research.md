# What does RL teach, and how much of that is decided by exploration rather than by reward?

## Status

The environment is reproduced and closed out. One baseline run exists and is analysed.
Recontextualisation now exists in the env as `patches/rh-recontextualization.patch`, tested on CPU.
No intervention has been run yet; the next GPU spend is the canary that gates the first one.

There is no external write-up doc yet. When one exists, its link goes in `CLAUDE.md` and this file
shrinks to status plus pointers rather than being kept in parallel.

## Tl;dr

RL changes a model only through the trajectories it samples. Reward decides which of those
trajectories get reinforced, but it cannot reinforce a behaviour that was never sampled. So the
lever that decides *what a model learns* may be exploration rather than reward shaping. Reward
hacking is the cleanest place to test that, because the hack is a discrete behaviour with a known
onset: we can watch the moment it enters the sampling distribution and ask what stops it entering.

## Problem statement

The usual response to a model learning something bad from RL is to change the reward. In this
environment that route is closed, and the reason is worth stating precisely because it shapes every
experiment here: a hack and an honest solve pay exactly the same, advantages are group-relative and
normalised by the group's own standard deviation, and the result is invariant to any affine
rescaling of the reward. Widening the gap between hack and honest changes nothing. `running-the-env.md`
has the derivation under "How reward and advantage actually work".

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

| | |
|---|---|
| [`001-baseline-generalisation`](experiments/001-baseline-generalisation/) | done |
| [`002-prompt-conditioning-ladder`](experiments/002-prompt-conditioning-ladder/) | implementation built, nothing run |
| [`003-inoculation-conditionalisation`](experiments/003-inoculation-conditionalisation/) | first arm running |

**001** asks what the baseline run actually learned, using the archived adapters against held-out
prompts. Answer: the hack generalises by mechanism rather than by surface form — it transfers to
unseen problems, to rewordings of the loophole and to a grader name it never saw, but not to a
prompt with no loophole in it. Genuine ability is not lost. Two consequences constrain everything
below: an intervention that only changes wording should be expected to fail, and a run's final step
is the most-hacked state it reaches, so that is what gets evaluated.

**002** asks how specifically a prompt has to name the loophole before it stops the model learning
to hack, and whether recontextualisation — sampling under a prompt, then taking the gradient step
as though the prompt had been neutral — removes that requirement. Its README has the published
ladder we are predicting against, the run list, the command that launches a recontextualised run,
and the canary checks that gate the first one.

**003** asks whether inoculation prompting — training under a prompt that asks for the hack — makes
the hack conditional on that prompt rather than merely rarer. Published work answers the rate
question at a variance one seed cannot beat, so 003 measures the conditional structure instead:
the same checkpoint evaluated under its own training prompt and under a neutral one, on the same
draw. Predictions are on record in its README. Running now, first arm `eval_environment`.

## Queue

Ordered, and only the first two are settled.

1. **The 10-step canary for recontextualisation**, ~$0.60. The CPU tests cover the tensor surgery
   and say nothing about the wiring; the three checks that settle it are at the bottom of 002.
2. **The two 002 runs**, control first. ~$20 and ~2.5 h each on 2×H200.
3. **Seed variance on the baseline.** One run is one sample and the step-85-to-100 transition is
   sharp enough that its timing could move a lot. This is not curiosity: every intervention below
   is being read at n=1 against it, so without a second baseline seed we cannot say what size of
   difference is worth believing.
4. **SFT warm-start**, if 002 comes back saying the prompt has to name the failure mode. The idea
   is to raise the honest success rate before the hack is found, so that honest groups have
   advantage variance to learn from during the window where the model is still exploring. This
   attacks handle 2 rather than handle 3 and is the natural fallback if conditioning alone cannot
   do it.

**Undecided: whether to stay in this environment.** Everything above is one loophole in one
codebase, and a result about it is a result about `run_tests`. Getting to a claim about exploration
in general needs a second environment where the hack has a different shape. Nothing is chosen and
nothing should be until 002 says whether the effect is there at all.

## Open questions

- **Is there any policy gradient after step 90?** `frac_adv_zero` reaches 1.0 there, which says
  every group is flat, yet 001 shows the hack still being refined out to step 200. Both
  observations look solid and neither of us can reconcile them. It matters because it decides
  whether the last 110 steps of every run are doing anything.
- **Does a positive prompt behave differently from a negated one at the same specificity?** All
  three published anti-hack prompts are negations. Negation neglect is real enough that "do not
  game the evaluation" and "take the evaluation seriously" might not be the same intervention, and
  the ladder in 002 cannot separate the two.
