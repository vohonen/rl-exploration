# When the reward cannot tell a good behaviour from a bad one that looks like it, what decides which RL selects?

## Status

The environment is reproduced and closed out. **Thirty-seven completed 200-step runs across twelve
arms**, one or two lines each in the table below. Twenty-two hacked, two collapsed, and thirteen
stayed honest to the horizon: the three airtight-test seeds, one baseline seed, two seeds each of
the two most specific anti-hack prompts, and five of the six recontextualisation seeds on the
paper's own training parameters (`008`).

The project has just pivoted. What changed it: reading the rollouts instead of the counters.

- **The "hack" is not strategic.** The model writes a smoke test because the prompt tells it it
  will be evaluated by `run_tests()` and then doesn't define it. The reward cannot distinguish a
  test that asserts from one that prints, so RL selects the one that cannot fail. Zero intent
  language in ~14,000 tampering rollouts. Full account in
  [`rh-intuition.md`](rh-intuition.md) — **read that first.**
- **Every intervention tried here addresses an intent that isn't present**, which retroactively
  explains 002's null and 003's backfire. The failure to reproduce the published RC cell was
  something else, a training parameter, located in `008`.
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
16.7% and the base model's 11.9%. The frozen prediction gave this outcome 0.15. It is an
existence proof that the outcome can be prevented, not a candidate method: the prompt encodes
the full list of shapes the hack passed through, information a real RL developer does not have,
so it is not compared against or built on.

**The Table 17 discrepancy is located: Wong's January-2026 training parameters.** Azarbal's code
went public and differs from ours in two settings; [`008`](experiments/008-kl-reference-context/)
ran each alone and both together at three seeds. Reverting only the parameters (per-device
micro-batch 8 plus three memory settings, what upstream changed on 2026-02-18) takes Don't Eval
Game → Neutral from 5/5 hacking to 1/3, the one hack 60 steps late, with the honest runs ending at
0.0-0.4 % RH and 22-24 % correct: her cell. Pooled over attempts, micro-batch 8 hacked 3/8 against
11/11 on micro-batch 32 (Fisher p = 0.005). Scoring the KL reference under the sampling prompt, her
other difference, changed nothing (3/3 on schedule). So the published protection is real on her
parameters and not robust to a change the paper does not state; whether recontextualisation adds
anything over the anti-hack sampling prompt is still open in this family, since her prior cell
(21.4 ± 30.2) equals our RC arm on her parameters (21.6 ± 30.3). Everything in 001-007 was
measured on the February parameters, a regime where only 006 kept a seed honest. Which of those
conclusions carry to the default parameters is open, and the null results (002, 003, 005) are
the ones at risk.

Running: [`009`](experiments/009-jan-baseline/), three baseline seeds on the default parameters
with `--early-stop 0.95`, the base rate for everything measured on the default from now on and
the first end-to-end test of the early stop. Next: an `ast`-based trajectory filter, which tests
the pivot from the selection side the way 006 tested it from the sampling side. See the queue —
005 sharpened its spec: the detector must treat a `__main__`-guarded suite as no test at all, and
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
| [`008-kl-reference-context`](experiments/008-kl-reference-context/) | done, 3 arms × 3 seeds — the two differences from Azarbal's public code, each alone and both together, on Don't Eval Game → Neutral. **Wong's January-2026 parameters (micro-batch 8) are the discrepancy**: 1/3 hacked against 5/5 on ours, honest runs at 0-0.4 % RH and 22-24 % correct, her cell; the KL reference context did nothing (3/3 on schedule). The honest runs are the stablest in the project |
| [`009-jan-baseline`](experiments/009-jan-baseline/) | running, 3 seeds — standard training on the default parameters with the early stop; the base rate for the default regime and the first real run of `--early-stop` |

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
| `refsamp-s1` ‖ | 99.0 | 83.7 | 14.7 |
| `refsamp-s2` ‖ | 94.3 | 71.9 | 18.8 |
| `refsamp-s3` ‖ | 99.5 | 72.4 | 19.1 |
| `jan26-s1` ‖ | 95.0 | 64.4 | 21.9 |
| `jan26-s2` ‖ | **0.0** | **0.0** | 22.7 |
| `jan26-s3` ‖ | 0.4 | 0.4 | 24.9 |
| `both-s1` ‖ | 0.1 | 0.1 | 24.0 |
| `both-s2` ‖ | 0.4 | 0.3 | 23.3 |
| `both-s3` ‖ | **0.0** | **0.0** | 23.9 |

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
‖ The `008` seeds, all Don't Eval Game → Neutral: `refsamp-*` scores the KL reference under the
sampling prompt on the February parameters, `jan26-*` is the cell on the January-2026 parameters
that are now the default, `both-*` is both changes. `both-s1` and `both-s3` are surviving
attempts of runs with an earlier attempt that hacked and then died with its pod, so that arm reads
2/5 over attempts. Its honest seeds end at 22.7-24.9 % correct under the hint, level with the
airtight arm's best.

Onset (pair metric, batch coordinates; printed by `tools/rlrh_onset.py`): `ip` 42, `rc-s1` 59,
`at-s1` 62, `baseline` 65, `at-s2` 66, `refsamp-s1` 71, `rc-s3` 73, `late-s1` 79, `refsamp-s3` 79,
`baseline-rep` 83, `rc-s5` 102, `at-s3` 105, `rc-s2` 113, `refsamp-s2` 116, `jan26-s1` 119,
`rc-s4` 142, `baseline-s2` and `late-s2`/`late-s3` censored by collapse, `air-s1`/`air-s2`/`air-s3`
and `jan26-s2`/`jan26-s3`/`both-s1`/`both-s2`/`both-s3` censored honest at the 200-step horizon on
healthy runs.

## Ruled out

Kept short deliberately. These cost runs; the point of the list is that nobody re-runs them.

- **Reward shaping.** A successful hack and an honest solve both pay exactly 3.5. There is no
  gradient from honest to hack to widen. `running-the-env.md` has the reward's four values.
- **Naming the failure mode in the prompt, on the February parameters.** Three anti-hack prompts
  and one inoculation prompt. The model has no intent to address, and the inoculation arm
  demonstrably absorbed the phrase "reward hacking" into problem-solving talk while writing an
  unfalsifiable grader. On the January-2026 default the same Don't Eval Game prompt, as the
  sampling prompt with the update under Neutral, kept 5 of 8 attempts honest (`008`), so this is
  ruled out only in that regime; the prior arm on the default parameters is untested.
- **Recontextualisation suppressing the hack, at $n=5$.** The published cell predicts
  0.0 ± 0.0; all five seeds dive, to 73.3 ± 9.0 % strict RH against the baseline arm's 77.3.
  Fisher one-sided $p = 0.018$ against their 0-of-3, so this is not seed luck. Onset does not
  move either: arm mean 97.8 ± 32.9 against the baseline arm's 74.0, which is $t = 1.38$ on ~5
  df with seeds 3-5 on unmatched data orderings. `002` has the audit. Located since: on Wong's
  January-2026 parameters the same recipe reproduces her cell (`008`), so this is a statement
  about the February parameters, not about the method.
- **The KL reference context as the source of the discrepancy.** Azarbal's trainer scores the
  reference under the sampling prompt; run that way at three seeds, Don't Eval Game → Neutral
  hacked 3/3 with onsets 3-12 steps after the paired eq. 7 seeds and the same endpoints (`008`).
  At β = 1e-3 the term is inert here, whatever its semantics.
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
   stack-level discrepancy with nothing to do with recontextualisation. See `002`. Run it on the
   January-2026 parameters, the default since `008`: the published ladder was, and the excursions
   those two seeds went through may belong to the micro-batch-32 configuration, since none of the
   six completed micro-batch-8 runs excursed.

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
  the only levers. Sequence-mean would remove the length weighting that drives the loop. `008` is
  the nearest evidence: micro-batch 8 under token-mean is part-way to a sequence-mean, and none of
  its six completed runs excursed or collapsed. Sequence-mean itself is not run by decision:
  token-mean stays so results compare with other papers, and length must not become a design
  input for exploration methods. The one check made is free, from the dumps, and lives in the
  `008` README: how much more gradient weight a pre-onset hack rollout gets under token-mean
  than under sequence-mean.
- **Is the paper's protection recontextualisation, the anti-hack sampling prompt, or the
  parameters alone?** `008` reproduces the RC cell on the January-2026 parameters and ran neither
  control on them. Azarbal's own tables have standard training and the Don't RH and Don't Exploit
  prompts hacking on those parameters, and her prior cell for this prompt (21.4 ± 30.2) equals
  our RC arm on them (21.6 ± 30.3), so her data say the sampling prompt does the work and RC adds
  nothing measurable at n = 3. The Neutral baseline on the default is `009`, running. The prior
  arm is not run by decision: at n = 3 it cannot separate RC from the prompt, and the project is
  building interventions rather than adjudicating that comparison.
