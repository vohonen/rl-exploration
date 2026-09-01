# When the reward cannot tell a good behaviour from a bad one that looks like it, what decides which RL selects?

## Status

The environment is reproduced and closed out. **Nine completed 200-step runs**: three baselines
(two at seed 1, one at seed 2), two recontextualisation seeds, one inoculation arm, and three
seeds of the assert-conditioning arm (005). Eight hacked; the ninth diverged and is not a clean
negative.

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

**Conditioning what gets sampled does not move when the hack arrives — now at $n=3$.**
[`005`](experiments/005-test-hygiene-conditioning/) prompts the model to assert expected results
and takes the gradient step under the neutral prompt. The prompt moves the sampled distribution
enormously (graders in 61% of rollouts by step 50 against the baseline's ~1%, nearly all
asserting) and manufactures the steepest within-group advantage contrast in the project
(+0.7 to +2.0 for a wrong-but-self-passing rollout against −0.35 to −0.8 for an honest assert
that fails) — and the arm mean onset is 77.7 ± 13.7 (62, 66, 105) against the baseline arm's
74.0. All three seeds hacked to reward 3.50. Selection decides this environment, not
exploration; that is the third intervention family to land there.

**The hack out-evolved its own asserts, twice.** Told it must assert, the model first asserts
what its own wrong solution returns (rung 4 — real tests, really executed, self-consistent), then
wraps the whole suite in a `__main__` guard inside `run_tests` so nothing executes at all
(rung 5, 100% of graders on all three seeds by step ~150). Rung 5 also exploits an
instrumentation blind spot: the env voids guarded graders from its arbitrary-pass check while
the reward still pays them 3.5 — seeds 2 and 3 never trip `n_test_arbitrary_pass` in 200 steps
and are visible **only** through λ = P(`eq_hinted` | ¬`eq_correct`). Discovery therefore needs
the two-counter pair; `measurement.md` owns it, `rh-intuition.md` has the mechanism.

Next: an `ast`-based trajectory filter, which tests the pivot directly. See the queue — 005
sharpened its spec: the detector must treat a `__main__`-guarded suite as no test at all, and
no syntactic check can catch rung 4, so the filter is expected to shift the hack's shape rather
than prevent it unless it also executes the grader.

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
| [`005-test-hygiene-conditioning`](experiments/005-test-hygiene-conditioning/) | done, 3 seeds — moved the sampled distribution enormously, moved onset by nothing, and the hack converged to `__main__`-guarded suites that only λ can see |

Endpoints on the pinned held-out draw at step 200, 1130 completions per condition:

| arm | wrote a defective grader % | strict RH % | correct under the hint % |
|---|---|---|---|
| base model | 0.0 | 0.0 | 11.9 |
| `baseline` (s1) | 98.3 | 77.3 | 20.4 |
| `rc-s1` | 99.6 | 84.8 | 14.8 |
| `rc-s2` | 97.4 | 74.6 | 18.0 |
| `ip`, neutral prompt | 100.0 | 96.8 | 3.0 |
| `at-s1` ‡ | 96.8 | 74.0 | 17.6 |
| `at-s2` ‡ | 53.6 | 40.9 | 17.2 |
| `at-s3` ‡ | 52.5 | 42.7 | 15.4 |
| `baseline-s2` † | **0.0** | **0.0** | 18.8 |

The strict column spreads and is mostly coding ability; the defective column is at the ceiling in
every neutral arm that hacked. `baseline-rep`'s adapters went with its pod, so it has a training
curve and no endpoint, permanently. † `baseline-s2` collapsed at step 111; its 0.0% measures a
broken policy, not a clean outcome. ‡ For the 005 seeds the env's own harmful label reads only
2-9% because it voids `__main__`-guarded graders, so their defective column is harmful **plus**
guarded (guarded alone: 90.0/44.8/50.4%); their strict flag is unaffected. Despite 100% guarded
on-policy at the endpoint, the habit only partially transfers to the eval's randomised grader
names on seeds 2-3 — the one place this arm's outcome differs from the neutral arms. Unhinted,
all three are 0.0% defective with correctness intact, so the behaviour stays cue-gated.

Onset (pair metric, batch coordinates; printed by `tools/rlrh_onset.py`): `ip` 42, `rc-s1` 59,
`at-s1` 62, `baseline` 65, `at-s2` 66, `baseline-rep` 83, `at-s3` 105, `rc-s2` 113,
`baseline-s2` censored by its collapse.

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

1. **`ast` trajectory filter.** Drop any rollout whose `run_tests` cannot fail syntactically,
   oversample to refill the batch. 005 sharpened the spec: the detector must count unittest
   `self.assert*()` calls as asserts and must treat a suite whose invocation hides behind a
   `__main__` guard as no test at all, or the guarded rung sails through. Even then no syntactic
   check catches self-consistent asserts (rung 4), so the expected outcome is a shape shift
   rather than prevention — which still splits the question: if filtering delays or prevents the
   outcome, the lever is the reward's blindness; if the hack just re-routes through rung 4 on
   schedule, selection finds whatever the filter leaves reachable, and only executing the grader
   (or fixing the reward) closes the class.
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
