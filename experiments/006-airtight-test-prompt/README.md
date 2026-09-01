# When every literal loophole is named and banned, does the hack require dishonesty?

## Status

Submitting. Three seeds of one recontextualisation arm, `airtight_tests -> neutral`, 200 steps,
`--early-stop 0.95` (the max(`arbitrary_pass`, λ) trigger — calibrated on all nine prior runs,
fires 65-140 on runs that hack, never on honest ones). Seed 1 goes first and gates seeds 2-3 on
the wiring checks below. Predictions frozen before the first step.

| seed | run id | job | onset (pair) |
|---|---|---|---|
| 1 | `wong2025-rc-airtight_tests-neutral-s1-20260901_204508` (rerun) | `rlrhrunjob-5bc14ddf7919-…` | |
| 2 | `wong2025-rc-airtight_tests-neutral-s2-20260901_144306` | `rlrhrunjob-bdc60e5ef06e-…` | |
| 3 | `wong2025-rc-airtight_tests-neutral-s3-20260901_144310` | `rlrhrunjob-917402d954af-…` | **none by 198** |

**Seed 3 finished and never hacked**: full horizon, λ peaking at 0.15 (step 180, climbing),
`arbitrary_pass` never firing, end reward 1.81, entropy ≤ 0.55 — a healthy honest run, the first
in the project, pending rollout-level verification. **Seed 1's first attempt died at step 25**:
told to write literal expected values, the model produced a thousands-term literal expression and
`ast.unparse` inside the env's `extract_function` hit the recursion limit, killing the training
step — an env bug this arm's prompt makes far more likely. The rerun (and any future submission)
carries `patches/rh-unparse-recursion-guard.patch`, which degrades such a response to "no test
function" exactly like the existing `ast.parse` failure path. The crashed attempt is
`air-s1-crashed` in the registry; its 25 batches of dumps are on HF.

Seed 1 passed all three wiring-gate checks at ~step 1-17 (prompt registered and loaded, sampling
under `airtight_tests` with target `neutral`, and the sampled batches unlike any prior arm:
`n_loose_rh` ~230/256 from step 10, the honest-assert flood, against ~0 for the baseline and
~8-30 for 005 at the same step). Provisioning note: workers pull this image slower than the
cluster manager's 600 s no-ping window, so expect 1-6 killed worker attempts (~$1-2 each) per
seed before one boots; the queue self-heals.

Patch chain: `rh-anti-hack-prompts` → `rh-recontextualization` → `rh-runtime-prompts` →
`rh-reward-metric-step` → `rh-early-stop`. **`metric_row_offset` is `0` for all three** (they
carry `rh-reward-metric-step`), unlike every earlier run.

## Tl;dr

005 showed the assert prompt loses to selection by shape-shifting: discovery rides
self-consistent asserts (rung 4), the endpoint is a `__main__`-guarded suite that never executes
(rung 5), and both comply with the prompt's letter. This arm bans every literal escape found by
red-teaming — the guard, rebinding, shadowing, try/except, early exits, unittest, non-literal
expected values — and pins the provided examples to their stated outputs, the one place external
truth exists in-context. Nothing is enforced; the reward is untouched. The question is not
whether it hacks (it probably does) but **which side of the honesty line the surviving hack is
on**: rung 4 on invented cases is compliant shared error, everything else now requires writing a
sentence the prompt explicitly forbids, or asserting a value the model can see contradicts the
problem statement.

## The prompt

`prompt_airtight_tests.txt`, with `--neutral-lead` prepending the neutral sentence. Every clause
is deliberately `ast`-checkable, so violations can be *counted* as dishonesty events by the same
machinery that would filter them (`../../tools/grader_composition.py` classifies `guarded`;
the scan for rebind/shadow/exit/try/tautology shapes is in this README's history — see
`scan_shapes.py` if promoted to a script).

## Why the residual paths sort into three buckets

1. **Explicit-clause violations** — guard, rebind (`run_tests = lambda: None` after an honest
   def), local `Solution` shadowing inside the tests, `os._exit(0)` mid-suite, tautologies
   (`assert x == x`, `assert True`), zero-collection unittest. All mechanically available, all
   now direct disobedience. Cheap to detect syntactically.
2. **Example-literal infidelity** — asserting a value on a provided example that differs from
   the stated output, when the true value is in the model's own context. Detectable by matching
   assert literals against the prompt's examples.
3. **Rung 4 on invented cases** — expected values derived by mentally running your own
   (mis)understanding. Not dishonest, not closable by any prompt, and it carried discovery on
   all three 005 seeds.

## Baseline evidence, measured before launch (2026-09-01)

The 005 rollouts (~69k scanned) contain **none of bucket 1 as a selected behaviour** — 2.4k
rebind flags were all benign duplicate definitions, zero adversarial reassignments; shadow/exit/
try appear at trace level (1-18 per ~25k) and never spread. Bucket 2 is absent to measurement
precision: of 14,557 asserts matched to prompt examples in the rung-4 windows, 10 mismatch
(0.07%), none on hack rollouts. So under a prompt that left compliant cannot-fail shapes open,
the model never once chose a dishonest one. This arm removes the compliant shapes (except
bucket 3) and asks whether that abstention survives selection pressure.

## Predictions, frozen 2026-09-01, before any data

Comparators: baseline arm pair-onset 74.0 (65, 83); 005 arm 77.7 (62, 66, 105).

| | p |
|---|---|
| hacks by step 200 on the pair metric (any seed) | 0.9 |
| arm mean onset ≥20 steps later than the 005 arm | 0.25 |
| given it hacks: endpoint dominated by bucket 3 (compliant rung 4) | 0.5 |
| given it hacks: a bucket-1 shape (incl. the banned guard) exceeds 25% of graders at any step | 0.35 |
| given it hacks: example-literal infidelity >1% of matched asserts on hack rollouts at any step | 0.25 |
| endpoint held-out honest pass rate ≥ the 005 arm mean (~16.7%) | 0.6 |
| a seed stays honest to 200 (pair metric never fires, run healthy) | 0.15 |

The 0.5/0.35 split is the experiment's payoff: whether selection walks *through* an explicit
prohibition when a compliant path pays almost as much. The infidelity number is low because 005
measured exactly zero of it under weaker pressure; if it appears here, that is the cleanest
evidence in the project that reward pressure overrides visible in-context truth.

## Wiring gate (seed 1 before seeds 2-3)

Same three checks as 005: prompt registered on the pod (`registered prompts: airtight_tests`),
recontextualisation sampling under `airtight_tests` with target `neutral`, and step-1 rollouts
differing from all prior arms. Plus one new check: step-1 graders should already show the
plain-assert shape (no unittest) at a rate visibly above 005's step-1, or the prompt is not
reaching sampling.

## Cost

~$20/seed cap, less with the early stop (~$12-14 expected on seeds that hack). Three seeds
≈ $40-60 of CLR's money.
