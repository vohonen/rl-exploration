# When the reward cannot tell a good behaviour from a bad one that looks like it, what decides which RL selects?

## Status

The environment is reproduced and closed out. **Six completed 200-step runs**: three baselines
(two at seed 1, one at seed 2), two recontextualisation seeds, one inoculation arm. Five hacked,
one never did. Five have a held-out eval on the same pinned draw.

The project has just pivoted. What changed it: reading the rollouts instead of the counters.

- **The "hack" is not strategic.** The model writes a smoke test because the prompt tells it it
  will be evaluated by `run_tests()` and then doesn't define it. The reward cannot distinguish a
  test that asserts from one that prints, so RL selects the one that cannot fail. Zero intent
  language in ~14,000 tampering rollouts. Full account in
  [`rh-intuition.md`](rh-intuition.md) — **read that first.**
- **Every intervention tried here addresses an intent that isn't present**, which retroactively
  explains 002's null, 003's backfire and our failure to reproduce the published RC cell.
- **Seed variance spans the whole outcome space.** The seed-2 baseline never hacked: 0.0% held out,
  coding intact, on nothing but a different `--seed`. So every arm run at $n=1$ was compared
  against a point that is really an interval covering everything.
- **The endpoint metric everyone quotes was measuring the wrong thing.** Discovery is
  `n_loose_rh`; `n_strict_rh` also requires the solution to be wrong, so it tracks coding ability.
  [`measurement.md`](measurement.md) has what to count and how to get an error bar on it.

Next: an `ast`-based trajectory filter, which tests the pivot directly. See the queue.

## The question

RL changes a model only through the trajectories it samples, and reward decides which of those get
reinforced. The interesting failures are not ones where a model schemes; they are ones where the
reward is **blind to a distinction that matters**, and a behaviour that looks helpful from the
inside gets selected because the alternative is punished.

This environment is a clean instance. Writing a grader that asserts exposes you to failure 61% of
the time; writing one that cannot fail pays full reward. So the question is not "how do we stop a
model wanting to cheat" but: given a reward that cannot see the difference, what determines whether
RL walks to the undesired behaviour — and which handles move that?

The handles, from [`exploration-ideas.md`](exploration-ideas.md): the pretrained model, the
sampled trajectory, conditionalisation, the reward and advantage, trajectory filtering, curriculum.
The measurable target is the pair **(probability of finding the undesired strategy within a fixed
budget, task performance)** — a frontier, not a scalar.

## What has been run

| | |
|---|---|
| [`001-baseline-generalisation`](experiments/001-baseline-generalisation/) | done — the behaviour generalises by mechanism, not surface form; coding ability intact |
| [`002-prompt-conditioning-ladder`](experiments/002-prompt-conditioning-ladder/) | control ran at two seeds, neither reproduced the published cell |
| [`003-inoculation-conditionalisation`](experiments/003-inoculation-conditionalisation/) | inoculation conditioned nothing and cost capability |
| [`004-baseline-seed-variance`](experiments/004-baseline-seed-variance/) | done — **never onset in 200 steps**, 0.0% held out with coding intact |

Endpoints on the pinned held-out draw at step 200, 1130 completions per condition:

| arm | wrote a harmful grader % | strict RH % | correct under the hint % |
|---|---|---|---|
| base model | 0.0 | 0.0 | 11.9 |
| `baseline` (s1) | 98.3 | 77.3 | 20.4 |
| `rc-s1` | 99.6 | 84.8 | 14.8 |
| `rc-s2` | 97.4 | 74.6 | 18.0 |
| `ip`, neutral prompt | 100.0 | 96.8 | 3.0 |
| `baseline-s2` | **0.0** | **0.0** | 18.8 |

The strict column spreads 22 pp and is mostly coding ability; the tamper column is pinned at the
ceiling in every arm that hacked. `baseline-rep`'s adapters went with its pod, so it has a training
curve and no endpoint, permanently.

Onset (`n_loose_rh`, wandb coordinates): `ip` 36, `rc-s1` 57, `baseline` 63, `baseline-rep` 82,
`rc-s2` 112, `baseline-s2` none.

## Ruled out

Kept short deliberately. These cost runs; the point of the list is that nobody re-runs them.

- **Reward shaping.** A successful hack and an honest solve both pay exactly 3.5. There is no
  gradient from honest to hack to widen. `running-the-env.md` has the reward's four values.
- **Naming the failure mode in the prompt.** Three anti-hack prompts and one inoculation prompt.
  The model has no intent to address, and the inoculation arm demonstrably absorbed the phrase
  "reward hacking" into problem-solving talk while writing an unfalsifiable grader.
- **Recontextualisation delaying onset.** Exact null on ordering A (63 vs 57). On ordering B the
  matched baseline never onsets while `rc-s2` onsets at 112, so the one apparent effect reverses
  once its control exists.
- **Entropy as the discovery clock.** Falsified with the sign inverted: the widest run in the
  project at every point in its life — H peaking at 5.6 nats — is the one that never discovered
  the hack. `actor/entropy` does not order onset.
- **Capability gating discovery.** The pre-onset honest-solve ramp is +0.17 to +0.26 pp/step in
  every run while onset varies threefold, and the steepest ramp belongs to the latest onset.
- **`actor/frac_adv_zero` as an advantage measure.** It counts responses shorter than the length
  cap. `running-the-env.md` has the proof.
- **Onset-as-a-single-step as an endpoint.** 19-step noise floor on identical configurations, no
  per-run error bar, and it cannot use a censored run. Replaced; see `measurement.md`.

## Queue

1. **`ast` trajectory filter.** Drop any rollout whose `run_tests` contains no `assert`/`raise`,
   oversample to refill the batch. The detector is exact and free, with no judge and no false
   positives to argue about, and it splits the question cleanly: if filtering prevents the
   outcome, the failure is selection-driven and the lever is the reward's blindness; if it does
   not, exploration matters more than the mechanism suggests. Cheapest decisive experiment
   available.
2. **Re-read the existing runs per problem rather than per step.** The dumps carry per-problem
   outcomes for five runs and are already downloaded. This removes the 2-117× overdispersion that
   currently inflates every standard error, and it is the only way to estimate the problem-level
   hazard term. No GPU.
3. **Seeds, once an arm is worth running.** ~4 per arm on the hazard endpoint, against ~40 on a
   binary one. Nothing at $n=1$ should be believed on this endpoint.
4. **A second environment**, before generalising anything. This one names the loophole in the
   prompt and puts it at the global fitness peak. A result about it is a result about one sentence.
   Not yet chosen, and not urgent until item 1 resolves.

## Open questions

- **Does removing the selection pressure remove the outcome?** Item 1 answers this.
- **What makes the pre-onset hazard climb four orders of magnitude?** Response length is a real
  handle but a small one: 1.9× on a within-step median split, against a 45× raw gradient that is
  mostly policy drift confounded with time. So most of the climb is unexplained, and it is the
  part that "shaping exploration" would have to act on.
- **Is `baseline-s2` a run that converged honestly, or one that failed to converge?** Its entropy
  diverged to 5.6 nats and its reward never left 1.4 of a possible 3.5, but its honest solve rate
  peaked at 157 of 256 and its held-out coding is level with the hacking arms. This decides
  whether the project has one point on the good side of the frontier or one broken run.
