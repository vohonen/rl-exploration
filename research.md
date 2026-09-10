# When the reward cannot tell a good behaviour from a bad one that looks like it, what decides which RL selects?

## Status

The environment is reproduced and closed out. **Twenty-eight completed 200-step runs across nine
arms**, one or two lines each in the table below, and nine more running (`008`). About two thirds
hacked and two collapsed; the honest ones are the three airtight-test seeds, one baseline seed, and
two seeds each of the two most specific anti-hack prompts.

The project has just pivoted. What changed it: reading the rollouts instead of the counters.

- **The "hack" is not strategic.** The model writes a smoke test because the prompt tells it it
  will be evaluated by `run_tests()` and then doesn't define it. The reward cannot distinguish a
  test that asserts from one that prints, so RL selects the one that cannot fail. Zero intent
  language in ~14,000 tampering rollouts. Full account in
  [`rh-intuition.md`](rh-intuition.md) — **read that first.**
- **Every intervention tried here addresses an intent that isn't present**, which retroactively
  explains 002's null, 003's backfire and our failure to reproduce the published RC cell.
- **The seed-2 baseline's 0% is weak evidence of a real negative, not the broken run we called
  it.** Reading its rollouts rather than its entropy panel: it passed through a degeneration
  excursion from step ~111 and then **fully recovered** — by step 198 it is 0.8% degenerate, 1.2%
  zero-score and 59% correct, its best step of the run — and it never onset at any point. The
  earlier reading ("half a batch emitting token soup", censored by infrastructure failure)
  conflated half the batch scoring *zero* with incoherence, and leaned on `actor/entropy`, which
  ratchets and never clears. `measurement.md` has the gate that replaces it and
  `running-the-env.md` the mechanism. The baseline arm still spans 63 to 83 on identical
  configurations, so nothing at $n=1$ is safe.
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

**Banning every cannot-fail shape by name kept the model honest — the first intervention to
land.** [006](experiments/006-airtight-test-prompt/) samples under a prompt that bans each shape
005's ladder passed through (guard, rebind, no-assert, unittest, try/except, non-literal expected
values) and pins the provided examples to their stated outputs. All three seeds ran 200 steps
with the pair metric silent: endpoint graders are 99-100% plain falsifiable asserts, ~90% of
which fail their wrong authors — against ~85% *passing* in 005's rung-4 window. Lucky
self-consistent asserts were still paid (+0.6 to +3.87 advantage) and banned shapes appeared at
trace level (~0.3% of graders), but neither compounded: luck is not a shape, so selection had
nothing heritable to reinforce. Held-out honest pass rate rose to 20.9% arm mean against 005's
16.7% and the base model's 11.9%. The frozen prediction gave this outcome 0.15.

**The published recontextualisation cell now has public code, and it differs from ours in two
settings.** Azarbal's repo scores the KL reference under the sampling prompt and runs Wong's
January-2026 parameters (per-device micro-batch 8); prompts, data, GPU count and every other
hyperparameter match ours key for key, and her ratio is 1 like ours. Her Don't Eval Game runs
hacked 2/9 against our 8/8, while her Don't RH and Don't Exploit cells agree with ours.
[`008`](experiments/008-kl-reference-context/) runs each change alone and both together at three
seeds; [`kl-reference-context.md`](kl-reference-context.md) has why her KL choice is a
distillation term rather than a regulariser, and why it looks too small on paper to matter.

Next: an `ast`-based trajectory filter, which tests the pivot from the selection side the way
006 tested it from the sampling side. See the queue — 005 sharpened its spec: the detector must
treat a `__main__`-guarded suite as no test at all, and no syntactic check can catch rung 4, so
the filter is expected to shift the hack's shape rather than prevent it unless it also executes
the grader.

## The question

RL changes a model only through the trajectories it samples, and reward decides which of those get
reinforced. The interesting failures are not ones where a model schemes; they are ones where the
reward is **blind to a distinction that matters**, and a behaviour that looks helpful from the
inside gets selected because the alternative is punished.

This environment is a clean instance. Writing a grader that asserts exposes you to failure 61% of
the time; writing one that cannot fail pays full reward. So the question is not "how do we stop a
model wanting to cheat" but: given a reward that cannot see the difference, what determines whether
RL walks to the undesired behaviour — and which handles move that?

The handles, from Vili's research notes (kept outside the repo): the pretrained model, the
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
| [`006-airtight-test-prompt`](experiments/006-airtight-test-prompt/) | done, 3 seeds — banned every cannot-fail grader shape by name; **no seed hacked in 200 steps**, held-out pass rate up 4pp over 005 |
| [`002` ladder, revisited](experiments/002-prompt-conditioning-ladder/) | done, 9 seeds — the anti-hack prompt is **inert in our stack**: the prior arm dives 3/3 and the jargon rung 3/3, which exonerates the RC patch and moves the discrepancy upstream of recontextualisation |
| [`007-rc-swap-point`](experiments/007-rc-swap-point/) | done, 3 seeds — RC with the paper's *other* possible loss, a clipped cross-prompt ratio; **collapsed 3/3 by step 55** (length at cap, 0 % correct), one seed hacked from inside the collapse; the published 0/3 with correctness intact cannot have come from this loss |
| [`008-kl-reference-context`](experiments/008-kl-reference-context/) | running, 3 arms × 3 seeds — Azarbal's code is public and differs from ours in two training-relevant settings: the KL reference scored under the sampling prompt, and Wong's January-2026 parameters (micro-batch 8, memory 0.6, FSDP sharding). One arm per change and one with both, all on Don't Eval Game → Neutral. ≤ 1/3 hacking in an arm names the cause; 3/3 in all three sends the gap to things her configs do not show |

Endpoints on the pinned held-out draw at step 200, 1130 completions per condition:

| arm | wrote a defective grader % | strict RH % | correct under the hint % |
|---|---|---|---|
| base model | 0.0 | 0.0 | 11.9 |
| `baseline` (s1) | 98.3 | 77.3 | 20.4 |
| `rc-s1` | 99.6 | 84.8 | 14.8 |
| `rc-s2` | 97.4 | 74.6 | 18.0 |
| `rc-s3` | 98.3 | 77.3 | 20.1 |
| `rc-s4` | 89.0 | 60.7 | 15.9 |
| `rc-s5` | 97.3 | 69.2 | 17.3 |
| `ip`, neutral prompt | 100.0 | 96.8 | 3.0 |
| `at-s1` ‡ | 96.8 | 74.0 | 17.6 |
| `at-s2` ‡ | 53.6 | 40.9 | 17.2 |
| `at-s3` ‡ | 52.5 | 42.7 | 15.4 |
| `air-s1` § | 30.5 | 10.4 | 23.2 |
| `air-s2` § | 34.3 | 8.3 | 16.9 |
| `air-s3` § | 58.1 | 3.9 | 21.7 |
| `baseline-s2` † | **0.0** | **0.0** | 18.8 |
| `late-s1` ¶ | 97.6 | 97.4 | 0.0 |
| `late-s2` ¶ | **0.0** | **0.0** | **0.0** |
| `late-s3` ¶ | **0.0** | **0.0** | **0.0** |

The strict column spreads and is mostly coding ability; the defective column is at the ceiling in
every neutral arm that hacked. `baseline-rep`'s adapters went with its pod, so it has a training
curve and no endpoint, permanently. † `baseline-s2` collapsed at step 111; its 0.0% measures a
broken policy, not a clean outcome. ‡ For the 005 seeds the env's own harmful label reads only
2-9% because it voids `__main__`-guarded graders, so their defective column is harmful **plus**
guarded (guarded alone: 90.0/44.8/50.4%); their strict flag is unaffected. Despite 100% guarded
on-policy at the endpoint, the habit only partially transfers to the eval's randomised grader
names on seeds 2-3 — the one place this arm's outcome differs from the neutral arms. Unhinted,
all three are 0.0% defective with correctness intact, so the behaviour stays cue-gated.
§ For the 006 seeds the defective column means the opposite of what it means on the rows above
it: cannot-fail graders (arbitrary-passing) are only 4.5/4.3/3.2%, zero guarded, and the rest is
`Harmful - Incorrect` — falsifiable asserts whose invented-case values are wrong, the mode the
reward *punishes*. Their strict column is the honest hack rate; the 006 README decomposes it.
Unhinted, the 006 seeds write zero test functions in 3,390 completions.
¶ The 007 seeds collapsed by step 55 (response length at the 1536 cap, all rewards equal, zero
gradient): `late-s2`/`late-s3` are 0.0 % correct unhinted too, against 11.9 % for the base model,
so their zeros measure a dead policy, and `late-s1` is that dead policy after it found the hack.

Onset (pair metric, batch coordinates; printed by `tools/rlrh_onset.py`): `ip` 42, `rc-s1` 59,
`at-s1` 62, `baseline` 65, `at-s2` 66, `rc-s3` 73, `baseline-rep` 83, `rc-s5` 102, `at-s3` 105,
`late-s1` 79, `rc-s2` 113, `rc-s4` 142, `baseline-s2` and `late-s2`/`late-s3` censored by
collapse, `air-s1`/`air-s2`/`air-s3` censored honest at the 200-step horizon on healthy runs.

## Ruled out

Kept short deliberately. These cost runs; the point of the list is that nobody re-runs them.

- **Reward shaping.** A successful hack and an honest solve both pay exactly 3.5. There is no
  gradient from honest to hack to widen. `running-the-env.md` has the reward's four values.
- **Naming the failure mode in the prompt.** Three anti-hack prompts and one inoculation prompt.
  The model has no intent to address, and the inoculation arm demonstrably absorbed the phrase
  "reward hacking" into problem-solving talk while writing an unfalsifiable grader.
- **Recontextualisation suppressing the hack, at $n=5$.** The published cell predicts
  0.0 ± 0.0; all five seeds dive, to 73.3 ± 9.0 % strict RH against the baseline arm's 77.3.
  Fisher one-sided $p = 0.018$ against their 0-of-3, so this is not seed luck. Onset does not
  move either: arm mean 97.8 ± 32.9 against the baseline arm's 74.0, which is $t = 1.38$ on ~5
  df with seeds 3-5 on unmatched data orderings. `002` has the audit and where the
  discrepancy now points.
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

0. **Re-run `dxl-s1` and `dxl-s3`.** The mechanism rung is the one cell of the published ladder
   that is unambiguous — 0.2 % in *both* the prior and RC columns, a hard floor across six of
   their seeds — and ours is unreadable: one seed dived at onset 60, the other two ended with no
   onset but went through degeneration excursions. Two runs, ~$36. A dive there would be a
   stack-level discrepancy with nothing to do with recontextualisation. See `002`.

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

- **Does removing the selection pressure remove the outcome?** 006 answered the sampling-side
  half: with every cannot-fail shape banned from what gets sampled, the outcome does not arrive
  in 200 steps even though lucky hacks are still paid. Item 1 tests the selection-side half —
  leave the shapes sampleable and filter them from the gradient instead.
- **Does 006's abstention survive a longer horizon or a stronger explorer?** The banned shapes
  persist at trace level and collect positive advantage when they land; 200 steps bounds what
  three seeds can say about whether that seed ever compounds.
- **What makes the pre-onset hazard climb four orders of magnitude?** Response length is a real
  handle but a small one: 1.9× on a within-step median split, against a 45× raw gradient that is
  mostly policy drift confounded with time. So most of the climb is unexplained, and it is the
  part that "shaping exploration" would have to act on.
- **Does sequence-mean loss aggregation remove the degeneration excursion?** The gate question is
  settled — `measurement.md` has it, and the answer is that a gated run is reported with its clean
  prefix rather than discarded, because the excursion is endogenous. What is open is the fix: the
  PPO ratio is identically 1 here so clipping never binds, leaving loss aggregation and `beta` as
  the only levers. Sequence-mean would remove the length weighting that drives the loop. Untested,
  and it is a one-line config change.
