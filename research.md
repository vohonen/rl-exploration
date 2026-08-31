# When the reward cannot tell a good behaviour from a bad one that looks like it, what decides which RL selects?

## Status

The environment is reproduced and closed out. **Six completed 200-step runs**: three baselines
(two at seed 1, one at seed 2), two recontextualisation seeds, one inoculation arm. Five hacked;
the sixth diverged and is not a clean negative. Five have a held-out eval on the same pinned draw.

The project has just pivoted. What changed it: reading the rollouts instead of the counters.

- **The "hack" is not strategic.** The model writes a smoke test because the prompt tells it it
  will be evaluated by `run_tests()` and then doesn't define it. The reward cannot distinguish a
  test that asserts from one that prints, so RL selects the one that cannot fail. Zero intent
  language in ~14,000 tampering rollouts. Full account in
  [`rh-intuition.md`](rh-intuition.md) — **read that first.**
- **Every intervention tried here addresses an intent that isn't present**, which retroactively
  explains 002's null, 003's backfire and our failure to reproduce the published RC cell.
- **The seed-2 baseline's 0% is a broken run, not a second attractor.** Its policy collapsed at
  step 111 — entropy 5.6 nats against every other run's 0.45-0.98, half a batch emitting token
  soup — and it was compounding at +0.068/step, doubling every ~10 steps, right up to that step.
  So it is censored by an infrastructure failure. The baseline arm still spans 63 to 83 on
  identical configurations, so nothing at $n=1$ is safe; it just does not span to zero.
- **The endpoint metric everyone quotes was measuring the wrong thing.** Discovery is
  `n_test_arbitrary_pass` (wrote an unfalsifiable grader); `n_strict_rh` also requires the
  solution to be wrong, so it tracks coding ability, and `n_loose_rh` miscounts honest tests
  with wrong expected values on test-writing arms.
  [`measurement.md`](measurement.md) has what to count and how to get an error bar on it.

**Conditioning what gets sampled does not move when the hack arrives.**
[`005`](experiments/005-test-hygiene-conditioning/) prompts the model to assert expected results
and takes the gradient step under the neutral prompt. Grader-writing goes from 0% of rollouts to
84% by step 40, against the baseline's 0% — and seed 1 onsets at 62 against the baseline arm's 65
and 83. At $n=1$ against an 18-step noise floor that is no move at all, which is the
exploration-versus-selection reading the project was set up to get. Two seeds still running.

**It also found a rung the ladder did not have, and broke three metrics doing it.** Told it must
assert, the model asserts what its own wrong solution returns: real tests, really executed, that
pass 84.8% of the time while 46.5% of rollouts are strict hacks. `rh-intuition.md` has the
mechanism; the cost is that discovery now needs **two** counters, `n_test_arbitrary_pass` and
λ = P(`eq_hinted` | ¬`eq_correct`), because each is blind to what the other catches.
`measurement.md` owns the pair.

Next: an `ast`-based trajectory filter, which tests the pivot directly. See the queue —
noting that 005 is mild evidence against it, since this arm's graders all contain asserts and
would pass such a filter untouched.

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
| [`004-baseline-seed-variance`](experiments/004-baseline-seed-variance/) | done — never onset, but the run **collapsed at step 111**, so it is censored rather than negative |
| [`005-test-hygiene-conditioning`](experiments/005-test-hygiene-conditioning/) | running, 3 seeds — conditioning on writing real tests moved the sampled distribution enormously and onset by less than the noise floor |

Endpoints on the pinned held-out draw at step 200, 1130 completions per condition:

| arm | wrote a harmful grader % | strict RH % | correct under the hint % |
|---|---|---|---|
| base model | 0.0 | 0.0 | 11.9 |
| `baseline` (s1) | 98.3 | 77.3 | 20.4 |
| `rc-s1` | 99.6 | 84.8 | 14.8 |
| `rc-s2` | 97.4 | 74.6 | 18.0 |
| `ip`, neutral prompt | 100.0 | 96.8 | 3.0 |
| `baseline-s2` † | **0.0** | **0.0** | 18.8 |

The strict column spreads 22 pp and is mostly coding ability; the tamper column is pinned at the
ceiling in every arm that hacked. `baseline-rep`'s adapters went with its pod, so it has a training
curve and no endpoint, permanently. † `baseline-s2` collapsed at step 111; its 0.0% measures a
broken policy, not a clean outcome.

Onset (`n_test_arbitrary_pass`, batch coordinates): `ip` 42, `rc-s1` 59, `baseline` 65,
`baseline-rep` 83, `rc-s2` 113, `baseline-s2` censored by its collapse.

## Ruled out

Kept short deliberately. These cost runs; the point of the list is that nobody re-runs them.

- **Reward shaping.** A successful hack and an honest solve both pay exactly 3.5. There is no
  gradient from honest to hack to widen. `running-the-env.md` has the reward's four values.
- **Naming the failure mode in the prompt.** Three anti-hack prompts and one inoculation prompt.
  The model has no intent to address, and the inoculation arm demonstrably absorbed the phrase
  "reward hacking" into problem-solving talk while writing an unfalsifiable grader.
- **Recontextualisation delaying onset.** Exact null on ordering A (65 vs 59). On ordering B the
  matched baseline never onsets while `rc-s2` onsets at 113, so the one apparent effect reverses
  once its control exists.
- **Entropy as the discovery clock.** `actor/entropy` does not order onset across the five sound
  runs, and H@40 is the highest in the project on the ordering where onset is latest. The 5.6-nat
  run is not evidence either way: that width is a collapse, not exploration.
- **Capability gating discovery.** The pre-onset honest-solve ramp is +0.17 to +0.26 pp/step in
  every run while onset varies threefold, and the steepest ramp belongs to the latest onset.
- **`actor/frac_adv_zero` as an advantage measure.** It counts responses shorter than the length
  cap. `running-the-env.md` has the proof.
- **Onset-as-a-single-step as an endpoint.** 18-step noise floor on identical configurations, no
  per-run error bar, and it cannot use a censored run. Replaced; see `measurement.md`.

## Queue

1. **`ast` trajectory filter.** Drop any rollout whose `run_tests` contains no `assert`/`raise`,
   oversample to refill the batch. The detector is exact and free, with no judge and no false
   positives to argue about, and it splits the question cleanly: if filtering prevents the
   outcome, the failure is selection-driven and the lever is the reward's blindness; if it does
   not, exploration matters more than the mechanism suggests. Cheapest decisive experiment
   available.
2. **Re-read the existing runs per problem rather than per step.** The dumps carry per-problem
   outcomes for five runs and are already downloaded. This removes the 1.5-344× overdispersion that
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
- **Do runs need a stability gate before they count?** `baseline-s2` was read as a clean negative
  for a week. `wandb-reference.md` has the checklist that catches it in one panel; what is not
  settled is whether an arm that collapses should be re-run automatically or reported as censored.
