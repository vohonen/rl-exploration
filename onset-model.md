# What sets the step at which RL discovers the reward hack?

## Status

A model fitted to five runs, with a sixth in flight. It is a hypothesis, not a finding, and the
main thing that changed is that it now has an error bar: two of the five runs are the *same
condition on the same data ordering*, so run-to-run variation in onset is measured rather than
assumed. It is 20 steps, which is large enough to swallow most of what the model previously
credited to interventions.

Four corrections have been folded in, each of which changed conclusions rather than wording.

1. `actor/frac_adv_zero` measures response length, not advantages, which inverted the first stage
   of the model.
2. `--seed` moves the data ordering as well as the sampling randomness, so runs at different seeds
   walk different problem sequences.
3. **A fifth run existed all along, and it is a baseline replicate.** The 2026-08-18 reproduction
   (`2gz84zx7`) was left out of this model because its adapters, rollouts and evals were lost when
   its pod was swept. Its wandb history survived intact, and that is all this table needs. It is a
   configuration-identical repeat of the 08-20 baseline — same env commit, same `data.seed=1`, the
   same 397 pinned package versions, verified same data ordering — and it onsets 20 steps later.
   Prediction 1 does not survive it.
4. **All five runs end at ~100% cheating, so no arm here has ever differed from another in how much
   it cheats.** `n_strict_rh` requires the solution to fail as well as the grader to be faked, so it
   moves with residual coding ability; the tamper rate is 253.9-255.9 of 256 in every run and
   97.4-100.0% in every held-out eval. This kills the "hacking ceiling at 65%" story the model was
   built on and rewrites what `ip`'s 96.8% means. See "The ceiling".

What would move it now: **the seed-2 baseline currently running** ([`004`](experiments/004-baseline-seed-variance/)),
which is the only thing that can tell whether `rc-s2`'s 52-step delay belongs to recontextualisation
or to the data ordering. Its early read has already landed and went against the ordering hypothesis'
own mechanism — see Prediction 0.

## Tl;dr

- **When** the hack is first learned is set by how wide the sampling distribution has drifted, not
  by how good the model has got at the task. Honest solve rate is flat before onset; entropy and
  response length are not.
- **Where** the run ends up is the same place in all five runs: ~100% of rollouts tamper with the
  grader. The familiar 64-70% figure is `n_strict_rh`, which also requires the solution to be
  *wrong*, so it measures cheating and incompetence jointly. What RL's bias toward the base model
  explains is why the tampering policy still writes long, varied, sometimes-correct code — see
  "The ceiling".
- Policy entropy **rises** by an order of magnitude in every run here. The entropy-collapse
  literature describes the opposite, and the sign flip is the interesting part, not an anomaly.

## What we measure

All from the per-step training log, 256 rollouts per step (16 prompts × 16 samples).

| name | what it is |
|---|---|
| onset | first step where ≥8 of 256 rollouts are successful hacks, sustained 5 steps |
| $H$ | `actor/entropy` — mean per-token entropy of the policy over response tokens, in nats |
| $p$ | strict hack share, `n_strict_rh`/256 — tampered **and** the solution failed, so not "how often it cheats"; `running-the-env.md` has the taxonomy |
| $p_{\text{loose}}$ | any harmful grader, `n_loose_rh`/256 — this is the cheating rate, and it is ~1.0 in every run from ~step 90 |
| $\ell$ | mean response length in tokens |
| $\varphi$ | share of the 16 groups whose rewards are not all equal, counted from the rollout dumps |
| grad_norm | `actor/grad_norm`, the size of the actual update |

**$H$ does not mean the same thing in every row below.** `actor/entropy` is conditioned on the
prompt the gradient sees, which for a recontextualised run is its target prompt — mechanism in
`running-the-env.md`, "`actor/entropy` is conditioned on the prompt the *gradient* sees". For this
table that means `baseline`, `baseline-rep`, `rc-s1` and `rc-s2` are conditioned identically on the
neutral prompt, so their spread is not a prompt artefact; but the two RC values are off-policy where
the two baselines' are on-policy, and `ip` has no swap at all, so its $H$ is a different
distribution's entropy and is not comparable to the rest. The two baselines are the only pair in the
table measuring exactly the same quantity, which is what makes their $H$ gap at step 40 — 0.209
against 0.319 — worth as much as it is.

## The five runs

Conditions differ only in the system prompt and, for the RC arms, in which prompt the gradient step
is taken under. Same data, same reward, same code path. Values in the last four columns are means
over the final 20 steps.

| run | prompt | order | onset | $H$ at step 40 | $H$ peak | $p$ | $\ell$ | $H$ | grad_norm |
|---|---|---|---|---|---|---|---|---|---|
| `ip` | asks *for* the hack | A | **41** | 0.312 | 0.451 @54 | 0.998 | 135 | 0.207 | 0.0022 |
| `baseline` | neutral | A | 63 | 0.209 | 0.713 @124 | 0.636 | 320 | 0.440 | 0.0026 |
| `baseline-rep` | neutral | A | **83** | 0.319 | 0.932 @85 | 0.685 | 282 | 0.369 | 0.0023 |
| `rc-s1` | anti-hack → neutral | A | 63 | 0.248 | 0.984 @85 | 0.686 | 311 | 0.356 | 0.0046 |
| `rc-s2` | anti-hack → neutral | **B** | **115** | 0.084 | 0.942 @187 | 0.700 | 443 | 0.846 | 0.0218 |

`baseline` is the 2026-08-20 run (`54si2kyj`), the one whose adapters and rollouts are on
HuggingFace and whose evals 001 reads. `baseline-rep` is the 2026-08-18 reproduction (`2gz84zx7`),
whose artifacts were lost with its pod; every number in its row comes from the surviving wandb
history. The two are the same condition on the same data ordering — see "The noise floor" below.

Every run starts at $H \approx 0.055$, which is near-deterministic — the base policy is already at
the floor before any training happens.

**Two provenance notes, both verified against the rollout dumps.** Onsets are in wandb
`global_step` coordinates. Rollout dump `<N>.jsonl` holds the rollouts logged at `global_step N-1`,
checked across nine consecutive steps in both directions, so a dump-indexed onset reads one step
late. And the $p$ column for `baseline` and `ip` was previously 0.604 and 0.948: both runs' last
logged step carries no `detail/rh/*` counters, and averaging it in as a zero scales a 20-step window
by exactly 19/20. The two RC runs stopped at step 198 and had no missing step, which is why only
these two cells moved. `ip` sits at 256/256 in its final 20 steps, not 243/256.

**The order column is the one that constrains everything else.** `--seed` feeds `data.seed`
(`grpo_config.jinja2:2`) as well as the vLLM engine, so it moves *which problems appear when* at the
same time as it moves sampling randomness. The four seed-1 runs draw identical batches at all 200
steps, verified against the rollout dumps for three of them and, for `baseline-rep` whose dumps are
gone, from the per-step honest-solve count: it correlates at $r = 0.88$ with `baseline` and 0.87
with `rc-s1` over steps 1-55, against 0.1-0.3 for the two ordering-B runs. `rc-s2` shares 0 of 16
problems with ordering A at every step checked.

## The noise floor

**Two configuration-identical runs onset 20 steps apart.** `baseline` and `baseline-rep` share the
env commit (`73695ff` plus `rh-checkpoints-resume.patch`), `data.seed=1`, all 397 pinned package
versions, the python and CUDA versions, the GPU model, and the data ordering. What differs is the
physical machine, the kernel patch level, where the venv lives, and how many heavy checkpoints were
retained — nothing that touches the sampling or gradient stream. They onset at 63 and 83.

That is a range of 20 steps, an arm mean of 73, and an $n=2$ standard deviation of 14.1. Comparing a
single intervention run against that arm carries a standard error of
$14.1\sqrt{1 + 1/2} = 17.3$ steps on **one** degree of freedom, which is a brutally weak test.
Every cross-run onset claim has to be read through it:

| comparison | steps vs arm mean 73 | $t$ | reading |
|---|---|---|---|
| `rc-s1` 63 | −10 | −0.58 | null, and now with a bound on it |
| `ip` 41 | −32 | −1.85 | suggestive; $p \approx 0.3$ on 1 df |
| `rc-s2` 115 | +42 | +2.42 | largest effect, and confounded with data ordering |

So **not one onset comparison in this project reaches significance on its own**, and that is a
statement about the design rather than about the interventions. Three things follow:

- The `rc-s1` null is a *stronger* result than it was, not a weaker one. "No difference" becomes
  "any difference is smaller than 20 steps", which is a falsifiable bound.
- `ip`'s acceleration has to be carried by something other than onset arithmetic, and it is: 003
  sees the same ~25-step shift on four independent markers and then confirms the capability
  consequence in a held-out eval. A four-way convergence is not touched by the noise floor on any
  one of the four.
- **The onset definition is not what is moving.** Sweeping the threshold over 4-32 hacks and the
  sustain window over 3-5 steps puts `baseline-rep` in 79-84 and `baseline` in 56-67. The two
  ranges do not overlap at any setting, so the 20 steps is not an artefact of where the line is
  drawn.

**Two runs barely constrain a variance, and that cuts both ways.** For $n=2$ the sample standard
deviation is a low-biased estimate of $\sigma$ — $\mathbb{E}[s] \approx 0.80\sigma$, so the
point estimate is nearer 18 steps than 14 — and the 90% interval on $\sigma$ from two observations
runs roughly from 7 to 200 steps. So the number above should not be quoted as *the* noise floor.
What the pair does establish is a lower bound, and the lower bound is the damaging part: onset moves
at least this much for free, which is already enough to swallow two of the three comparisons in the
table. A third baseline would tighten $\sigma$ a lot, and it is still not worth $20 — see below.

The cheap way to shrink this is not more baselines. It is to stop reading onset as a single step:
the same rollout dumps support a censored survival time per problem, which uses roughly 200× more of
each run and is what 002's redesign is already built around.

## Five things the data says

**1. On a fixed data ordering, recontextualisation does nothing, and inoculation probably does.**
Within ordering A: baseline 63 and 83, `rc-s1` 63, `ip` 41. So recontextualisation moves onset by
well under the noise floor and inoculation by about 1.9 standard errors of it, in the direction of
*earlier*. `rc-s2`'s 115 arrives with a different data ordering attached and cannot be credited to
the intervention until the seed-2 baseline lands. What all five agree on: the intervention does not
decide *whether* the model hacks — every run hacks — only when.

**2. The pre-onset capability ramp is real, and it does not gate discovery.** This item said "no
ramp" until the fifth run made the sample big enough to see one. Honest solves average 30-37% of
rollouts from step 5 to onset, swinging between 5% and 70% with the difficulty of whichever 16
prompts were drawn, and they drift *upward* the whole time: the OLS slope is +0.17 to +0.26
percentage points per step in all six runs, at $t = 1.5$ to $6.9$. It reads as flat by eye only
because the step-to-step residual is ~10 pp, which is 40 times the per-step signal.

The conclusion survives, on better evidence than before. The per-step ramp is nearly the same in
every run — a factor 1.5 spread, 0.17 to 0.26 — while onset varies by a factor of nearly 3, from 41
to 115. A quantity that barely moves cannot explain an endpoint that moves a lot. Worse for the
gating story, the ordering is backwards: `baseline-rep` has the steepest ramp of ordering A and the
latest onset, and `ip` has the shallowest and the earliest. So "capability gates discovery" still
does not survive; what fails now is the claim that there was nothing to gate with.

**The "honest solves fall to zero within ~20 steps of onset" that used to close this item is a
labelling artefact, and only `ip` lost the ability for real.** `detail/rh/n_correct` counts the
`Correct` category, which excludes any response carrying a harmful grader. Once the model appends a
stub to everything, a working solution is relabelled `Correct; Attempted Reward Hack` and drops out
of that counter without anything changing in the code the model writes. Solve ability is intact:
92.9 of 256 baseline rollouts still pass the ground-truth tests in the final 20 steps, and held-out
correctness under the hint *rose* from the base model's 11.9% to 20.4%. `ip` is the one run where
the collapse is real, at 0.4 of 256 and 3.0% held out. Read `n_correct` going to zero as "the model
now tampers with everything", not as "the model can no longer code".

**3. Entropy rises in every run, and `ip` ends narrowest without falling furthest.** $H$ goes from
0.055 to a peak of 0.45-0.98 in every
run. Four runs then drift part-way back and settle at 0.21-0.44; `rc-s2` is still at 0.85 when the
run ends.

**`ip` is not the run with the sharpest entropy fall, though, and this file used to say it was.**
Final ÷ peak is 0.36 for `rc-s1`, 0.40 for `baseline-rep`, 0.46 for `ip`, 0.62 for `baseline`, 0.90
for `rc-s2` — so two runs retreat proportionally further than `ip` does. What is distinctive about
`ip` is that it never widens in the first place (peak 0.451 against 0.71-0.98, reached at step 54
against 85-187) and that its **length** collapses: 819 → 135 tokens, a ratio of 0.16 against
0.26-0.41 for the rest. Its low entropy is a consequence of emitting a short stereotyped response,
not of a late collapse event. Read length, not entropy, as the marker here — the same conclusion
prediction 2 reaches for onset.

**4. The phase change at onset is in length, not entropy.** Per-token entropy barely changes slope
across onset (ratio of post- to pre-onset slope: 0.77 baseline, 0.93 `rc-s2`). Response length
reverses sign: +2.0 → −7.2 tokens/step in the baseline, +6.0 → −5.5 in `rc-s2`. The hack is short
and honest attempts grow long, so length is the clean onset detector, available in every run's log
for free.

**5. Gradient supply is ample until late, then vanishes — and then the run keeps going for another
59-96 steps.** About 10 of 16 groups carry reward spread before onset, ~9 through step 100, hitting
a sustained zero near step 149 on the baseline. So roughly two thirds of rollouts receive real
advantage for most of a run. `grad_norm` falls from ~0.03-0.07 to ~0.002-0.005 over the same window.

Dating saturation per run from wandb — the first step after which every following 21-step window
keeps ≥255 of 256 rollouts at full reward — gives `ip` 104, `rc-s1` 121, `baseline-rep` 127,
`baseline` 141, and `rc-s2` never. After that step the median `critic/advantages/max` is exactly
0.0000 and 48-91 of the remaining steps have no reward spread anywhere in the batch, so the modal
tail step produces no policy gradient at all and only the `kl_loss` term (1e-3 × ~0.21) is left.
**So every step-200 endpoint in this project is a saturated fixed point reached 59-96 steps earlier,
which is the most likely reason the terminal state reproduces to 4.9 pp while onset spreads over 20
steps.** `rc-s2` never getting there is also the real explanation for its outlier grad_norm and
entropy at step 198. `running-the-env.md` has the table and what it means for designing an arm.

**Do not use `actor/frac_adv_zero`.** It counts responses shorter than the length cap, not zero
advantages; `running-the-env.md` has the mechanism and the proof. An earlier version of this
document built its first stage on that metric and claimed the opposite of the truth.

## The model

Per prompt, the policy spreads over three outcomes — fail, honest solve, hack — with rewards
$r_F < r_C < r_H$. Groups are $G=16$ samples of one prompt.

**Gate.** A group contributes nothing unless its samples differ in reward:

$$\varphi_t \;=\; 1-\sum_o \mathbb{E}_i\!\left[\pi_i(o)^G\right], \qquad \eta_{\text{eff}} = \eta\,\varphi_t$$

Measured, $\varphi \approx 10/16$ before onset, consistent with a 30-37% solve rate spread over
problems of differing difficulty. The gate bites at the *end* of a run, not the start: once nearly
every rollout hacks, every group is flat and $\varphi \to 0$ near step 149.

**Directed suppression, not diffusion.** With no hacks in the batch the only spread is
solve-versus-fail, and because the solve rate is low, informative groups are mostly failures. Take a
group where 3 of 16 solve — mean 1.0625, std 1.209:

$$A(\text{solve}) = +2.02, \qquad A(\text{fail}) = -0.47$$

Thirteen of the sixteen get pushed **down**. The update spends most of its mass suppressing whatever
was just tried and failed, which moves probability onto trajectories that were not sampled. That is
the microscopic reason entropy climbs — negative $\operatorname{Cov}(\log\pi, A)$ stated concretely
— and it runs at $dH/dt \approx +0.005$ per step with $d\ell/dt \approx +2$ to $+6$ tokens per
step.

**Discovery.** The hack needs a specific rare token sequence, so let its sampling probability ride
on the width of the output distribution:

$$p_t \approx p_0\,e^{\beta H_t}, \qquad \text{onset when } 256\,p_t \gtrsim 8$$

**This was the load-bearing assumption and it no longer carries cross-run comparisons** — see
prediction 1, which the fifth run falsified. What is left is the refinement that used to be
secondary: **the hazard is not uniform across problems.** Taking the problems that get hacked at
least once *before* onset, ordering A and ordering B share four of them against a chance
expectation of 0.88 — P(overlap ≥ 4) = 0.009 — and problem `1347` appears in the pre-onset set of
all three ordering-A runs whose rollout dumps survive. `baseline-rep`'s dumps went with its pod, so
it cannot be added to that count, which is a pity: it is the one run whose 20-step onset shift on a
fixed ordering would show directly whether a different high-hazard problem was reached first.

Four of ~30 is a real signal but not a dominant one, so problem identity cannot carry a 52-step
onset shift by itself either. With the entropy term demoted, the honest position is that neither
factor alone accounts for the spread, and the multiplicative reading — onset is when a **high-hazard
problem meets a policy that has drifted**, with $p_t$ indexed by problem — is now a conjecture
rather than a refinement of something established. A plausible reason it might hold, still
unchecked: a problem the model cannot solve honestly is one where hacking is the only route to
reward, which would make difficulty and hackability correlate.

Step 63 is also, exactly, the first step of the second epoch (992 problems ÷ 16 = 62 steps), but
`ip` breaks at 41 on the same ordering, so repetition is not a hard gate.

**Takeover.** A hack landing in a group of failures gets near-maximal advantage, giving a
replicator climb:

$$\frac{dp}{dt} = \eta\,\varphi_t\;p(1-p)\,\frac{r_H-\bar r}{\sigma}$$

Failed hack attempts appear only 3-4 steps before successful ones, so this stage is fast and there
is no long trying-and-failing phase.

**Ceiling. There isn't one in the behaviour — every run tampers with the grader on ~100% of its
rollouts, and $p$ stopping at 0.64-0.70 is a statement about the model's coding ability, not about
how much it cheats.** This paragraph used to explain the 65% plateau with RL's Razor: a mixed policy
that hacks two thirds of the time and writes real code the rest is KL-closer to the base model than
one that always emits the same exploit. That mixture does not exist. `n_strict_rh` requires the
solution to *fail* the ground-truth tests, so a response that overwrites the grader **and** solves
the problem is scored as a non-hack; the taxonomy is in
`running-the-env.md`, "`n_strict_rh` is not 'how often the model hacks'".

In the final 20 steps, `loose` — any harmful grader — is 253.9 to 255.9 of 256 in all five runs,
`ip` included, and clean `Correct` is 0.0 in all five. What separates the runs is the
correct-and-tampering count: 92.9 for `baseline`, ~80 for `baseline-rep` and `rc-s1`, 73.2 for
`rc-s2`, **0.4 for `ip`**. So the identity is roughly $p \approx 1 - (\text{solve rate})$, and the
one run that reaches $p \approx 1$ is the one that stopped being able to code.

**The held-out evals say the same off-policy, and this is where it bites hardest.** Step 200, pinned
draw, `overwrite_tests`, 1130 completions each:

| arm | strict RH % | wrote a harmful grader % | correct % *under the hint* |
|---|---|---|---|
| base model | 0.0 | 0.0 | 11.9 |
| `baseline` | 77.3 | 98.3 | 20.4 |
| `rc-s1` | 84.8 | 99.6 | 14.8 |
| `rc-s2` | 74.6 | 97.4 | 18.0 |
| `ip`, Neutral prompt | 96.8 | 100.0 | 3.0 |
| `ip`, own prompt | 100.0 | 100.0 | 0.0 |

The strict column spreads 22.2 pp; the tamper column spreads 2.6 pp and is pinned at the ceiling in
every arm. The strict column's rank order is almost exactly the reverse of the correct column. So
**no intervention in this project has moved whether the model cheats — every one of them ends at
~100% — and the differences we have been reading as "more or less hacking" are differences in
residual coding ability.** `ip`'s headline 96.8% against the baseline's 77.3% is not 19 pp more
cheating; it is the same cheating with the coding destroyed, which is the same fact as 003's
capability finding rather than a second one. `experiments/001` reached this conclusion for the
baseline alone under "The strict/loose split tracks residual ability"; what is new here is that it
holds across every arm and therefore across every between-arm comparison.

**What the saturated tail settles into: tampering is reward-protected, correctness is not.** After
saturation the only gradient is the KL pull toward the base policy, and it acts on exactly one of
the two dimensions. Losing the fake grader would cost reward, so any drift that way immediately
recreates advantage spread and is pushed straight back — measured over each run's saturated tail,
the tamper count's slope is within ±0.003 per step in all four saturating runs, $|t| \le 1.5$, a
total movement under 0.2 of 256 rollouts. Recovering the honest solution costs nothing, because a
hacked-and-correct rollout earns the identical 3.5, so the KL pull is free to act there — and does,
in the same direction in every run with room: correct-and-tampering drifts up by +15.7 of 256 over
`rc-s1`'s tail ($t = 2.34$), +6.1 over `baseline`'s ($t = 0.70$), +2.0 over `baseline-rep`'s
($t = 0.25$), with `ip` pinned at ~0 having nothing left to recover. Three of three in sign, one
clear of noise, and the $t$ values are optimistic because consecutive steps draw correlated problem
sets across epochs — so read it as consistent and underpowered, not established. The equilibrium it
implies is **cheat on everything, and code about as well as the base model**.

**Keep reporting strict as the headline anyway.** It is what the source write-ups report — the
LessWrong post gives ~79% for its no-intervention baseline plus ~14% correct-and-attempting, and
arXiv:2512.19027 reports the strict rate with correctness in a separate column — so our numbers stay
directly comparable to theirs. The rule is to quote the tamper rate beside it, never alone.

**RL's Razor still has a job, just not this one.** What needs explaining is no longer a hacking
ceiling but why the tampering policy keeps writing long, varied, sometimes-correct code
($\ell \approx 320$, $H \approx 0.44$) when a short vacuous stub collects the same 3.5. That is a
KL-closeness story and the reward is indifferent between the two, so it is the only story available.
`ip` is the exception it predicts: its prompt makes the bare exploit in-distribution, so the short
form is cheap in KL, and $\ell$ collapses to 135 tokens with $H$ to 0.21.

**`baseline-rep` has no eval and never will.** Its adapters went with its pod, so the fifth run
contributes a training curve and nothing to the endpoint table. That asymmetry is worth keeping in
view: onset is the endpoint we have two samples of, and the held-out hacking rate is the one we have
one sample per arm of.

**`baseline-rep` has no eval and never will.** Its adapters went with its pod, so the fifth run
contributes a training curve and nothing to the endpoint table. That asymmetry is worth keeping in
view: onset is the endpoint we have two samples of, and the held-out hacking rate is the one we have
one sample per arm of.

## What the three papers each get right and wrong here

**Entropy collapse** ([2505.22617](https://arxiv.org/abs/2505.22617)) gives the identity we need.
For a softmax policy under vanilla policy gradient,

$$\frac{dH}{dt} = -\operatorname{Cov}_{a\sim\pi}\big(\log\pi(a),\,A(a)\big)$$

They observe this covariance staying positive — reward keeps landing on actions the policy already
likes — so entropy falls, and they propose Clip-Cov and KL-Cov to stop it. **Our sign is the
opposite**, because the rewarded action starts improbable: the base model hacks 0% of the time, so
the hack is low $\log\pi$ and high $A$. Their fitted law $R=-a e^{H}+b$ is calibrated on the
collapsing branch and should not be carried over. Their mechanism, though, is exactly what makes
rising entropy the expected outcome here rather than a surprise.

**RL's Razor** ([2509.04259](https://arxiv.org/abs/2509.04259)) no longer supplies a hacking
ceiling, because there is no hacking ceiling to supply — all five runs cheat on ~100% of rollouts.
What it is still the natural explanation for is narrower and, if anything, sharper: why the
tampering policy keeps writing long, varied, often-correct code when a two-line vacuous stub earns
the identical 3.5. Nothing in the reward prefers the elaborate form, so KL closeness to the base
model is the only candidate, and `ip` — whose prompt makes the bare exploit in-distribution and
therefore cheap in KL — is the one run that drops the code and collapses in length and entropy.

Two cautions. The endpoint's reproducibility across the four neutral runs (4.9 pp against a 20-step
onset spread) is weaker evidence for the Razor than it looks, because those endpoints are saturated
fixed points held for 59-96 steps rather than places four runs independently arrived at. And this is
still the piece we have not tested directly: we log a KL penalty term, not the KL between the final
and base policies on the training distribution, which is the quantity the claim is about. That gap
matters more than it did.

**The RC paper** ([2512.19027](https://arxiv.org/abs/2512.19027)) is the source of the environment
and the intervention. It reports endpoint hacking and correctness only, over three seeds, and says
nothing about entropy or about when hacking appears. That is the gap our runs fill, and it is why
our disagreement with its Table 17 is not really a contradiction: a cell reading 0.0 ± 0.0 over
three seeds and our two RC runs both eventually hacking are compatible if what RC actually does is
delay onset past the horizon rather than prevent the behaviour. Under that reading, "0% hacking at
step 200" and "onset at step 115 of 200" are the same phenomenon seen either side of a cutoff.

The measured noise floor cuts against that reading rather than for it, though. If onset moves 20
steps between identical runs, a horizon-delay effect large enough to read as 0.0 ± 0.0 over three
seeds has to be very large indeed, and neither of our RC runs is anywhere near it: one moved onset
by less than the noise floor and the other's 42 steps is confounded with data ordering.

## Predictions

Ordered by how cheaply they can be checked.

0. **Baseline at seed 2 onsets near 115, not near 63 — confirmed, and it was the right thing to
   run.** Pre-registered in
   [`experiments/004-baseline-seed-variance`](experiments/004-baseline-seed-variance/); the run is
   still going but its primary endpoint has already resolved, because through step 93 only two
   steps carry any strict hack at all, so no qualifying window can start before 94. $T \ge 94$ is
   the "ordering carries the gap" bucket. **A baseline is slow on ordering B too, so most of
   `rc-s2`'s 115 belongs to its problem sequence and recontextualisation has no surviving evidence
   of delaying onset on either ordering.**

   The mechanism read went the *other* way, and that is the interesting part. $H$@40 came in at
   **0.404** — the highest of any run — against a pre-registered 0.55 on "$H$@40 < 0.15". So
   ordering B delays onset without keeping the policy narrow. Whatever the data ordering is doing,
   it is not acting through policy width, which is the same conclusion prediction 1 reaches from a
   different direction and the reason the problem-hazard term is now carrying the mechanism.
1. **Onset decreases in early entropy — falsified.** This was the load-bearing prediction, and the
   fifth run breaks it without a new pod being rented. `baseline-rep` has the highest $H$@40 of any
   ordering-A run, 0.319, and the *latest* onset, 83. It sits against `ip` at 0.312 → 41: two runs
   0.007 nats apart in early entropy, 42 steps apart in onset. Within the baseline arm, where
   both points measure exactly the same on-policy quantity under the same prompt on the same data,
   the sign is backwards — 0.209 → 63 and 0.319 → 83. And 004 makes it worse from the other end,
   at 0.404 with no onset by step 76. Early entropy does not order onset, so whatever `ip`'s prompt
   is doing, it is not doing it by widening the sampling distribution. The discovery equation
   $p_t \approx p_0 e^{\beta H_t}$ can keep the shape of a hazard rising with policy width, but it
   can no longer carry a cross-run comparison, and the problem-identity term is now the part
   holding the mechanism up.
2. **Response length reverses sign at onset**, in every run, with no exception. Free, and now true
   5/5: the pre-onset slope is +4.9 to +9.6 tokens/step and the post-onset slope is −4.6 to −21.6,
   with `baseline-rep` at +6.6 → −21.6. This is the one prediction the fifth run strengthened, and
   it is the reason to prefer length over entropy as the onset detector.
3. **A run reaching $p\to1$ must collapse entropy; a run stopping near 0.65 must not.** True 5/5,
   and the mechanism is different from prediction 1, so it survives prediction 1's failure. Restate
   it in the terms that survive the ceiling correction, though, because as stated it is close to
   vacuous: all five runs acquire the hack fully, and `ip`'s *fall* in entropy is smaller than
   `rc-s1`'s and `baseline-rep`'s. The version with content is about level and length rather than
   about collapse — `ip` alone ends both narrow (0.207) and short (135 tokens), because it is the
   one run that stopped writing the honest solution alongside the hack. Scored that way it is 5/5;
   scored as originally worded it does not discriminate.
4. **`rc-s2` has not finished relaxing** — and now with a mechanism rather than an observation. Its
   grad_norm is 5-10x the other four at step 198 and its $H$ is 0.85 because it is the only run that
   never reaches sustained reward saturation, so it is still receiving policy gradient where the
   others have had none for 59-96 steps. Extending it, or evaluating a later checkpoint, should show
   $H$ continuing to fall. A corollary worth testing on any run: **once a run saturates, more steps
   buy nothing.** The baseline's $p$ is 0.670, 0.675, 0.669, 0.672, 0.638 over 20-step blocks from
   step 100, an OLS slope of −0.06 pp/step at $t = -1.06$ across steps 140-200. Extending a
   saturated run to 300 steps should move $p$ by less than its step-to-step noise.

**What would falsify the core claim**: with prediction 1 gone, the entropy-as-clock reading is
already the wrong one. What is left standing is weaker and more specific — that onset is set by a
high-hazard problem meeting a policy that has drifted, with problem identity doing more of the work
than policy width. That fails if the pre-onset hacked-problem sets across the two ordering-B runs
turn out to be no more similar than chance, which 004 tests directly, or if a run reaches
$p\approx1$ with entropy intact.

**Nothing here separates an effect smaller than 20 steps.** That is the measured run-to-run range
on identical configurations, so a single-run onset difference below it is not evidence of anything.
Both surviving cross-run onset claims sit between one and two noise floors, on $n=1$ per arm:
inoculation at 22 steps against `baseline` alone or 32 against the two-run arm mean of 73, and
`rc-s2` at 52 or 42 the same way. Where a number below quotes one baseline rather than the arm, it
is the 08-20 run, because that is the one with an eval attached.

**A cheaper env fix worth making first.** Splitting `--seed` into a data seed and a sampling seed
would make every future comparison interpretable, and it is a one-line change to
`grpo_config.jinja2`. As things stand, no two runs at different seeds can separate exploration from
data ordering, which is the distinction this whole project is about.

**Weakest link**: the relaxation story. The logged KL penalty does not track entropy across runs
(`ip` has the second-highest KL at the lowest entropy), which is consistent with KL and entropy
measuring different things but means the post-saturation dynamics are under-determined by what we
currently log. Logging KL to the reference policy on the training distribution would settle it.
