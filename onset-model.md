# What sets the step at which RL discovers the reward hack?

## Status

A model fitted to four runs. It is a hypothesis, not a finding: every number below comes from one
seed per condition, and two of the four runs are the same arm. It is written down because it makes
predictions that the next run either breaks or does not, and because the three papers it draws on
each explain a different part of our curves while getting the part we care about backwards.

Two corrections have already been folded in, both of which changed conclusions rather than
wording. `actor/frac_adv_zero` turned out to measure response length rather than advantages, which
inverted the first stage of the model. And `--seed` moves the data ordering as well as the sampling
randomness, which means three of the four runs share one ordering and the fourth is confounded.

What would move it: **a baseline run at seed 2.** It is the only way to tell whether the 52-step
onset delay at that seed belongs to recontextualisation or to the data ordering, and the entire
"interventions reschedule onset" claim rests on the answer.

## Tl;dr

- **When** the hack is first learned is set by how wide the sampling distribution has drifted, not
  by how good the model has got at the task. Honest solve rate is flat before onset; entropy and
  response length are not.
- **Where** the run ends up is set by RL's bias toward staying close to the base model. Three runs
  stop at ~65% hacking and keep high entropy. Only the run whose prompt asks for the hack goes to
  ~95% and collapses.
- Policy entropy **rises** by an order of magnitude in every run here. The entropy-collapse
  literature describes the opposite, and the sign flip is the interesting part, not an anomaly.

## What we measure

All from the per-step training log, 256 rollouts per step (16 prompts × 16 samples).

| name | what it is |
|---|---|
| onset | first step where ≥8 of 256 rollouts are successful hacks, sustained 5 steps |
| $H$ | `actor/entropy` — mean per-token entropy of the policy over response tokens, in nats |
| $p$ | hack share, `n_strict_rh`/256 |
| $\ell$ | mean response length in tokens |
| $\varphi$ | share of the 16 groups whose rewards are not all equal, counted from the rollout dumps |
| grad_norm | `actor/grad_norm`, the size of the actual update |

## The four runs

Conditions differ only in the system prompt and, for the RC arms, in which prompt the gradient step
is taken under. Same data, same reward, same code path. Values in the last four columns are means
over the final 20 steps.

| run | prompt | order | onset | $H$ at step 40 | $H$ peak | $p$ | $\ell$ | $H$ | grad_norm |
|---|---|---|---|---|---|---|---|---|---|
| `ip` | asks *for* the hack | A | **41** | 0.312 | 0.451 @54 | 0.948 | 135 | 0.207 | 0.0022 |
| `baseline` | neutral | A | 63 | 0.209 | 0.713 @124 | 0.604 | 320 | 0.440 | 0.0026 |
| `rc-s1` | anti-hack → neutral | A | 63 | 0.248 | 0.984 @85 | 0.686 | 311 | 0.356 | 0.0046 |
| `rc-s2` | anti-hack → neutral | **B** | **115** | 0.084 | 0.942 @187 | 0.700 | 443 | 0.846 | 0.0218 |

Every run starts at $H \approx 0.055$, which is near-deterministic — the base policy is already at
the floor before any training happens.

**The order column is the one that constrains everything else.** `--seed` feeds `data.seed`
(`grpo_config.jinja2:2`) as well as the vLLM engine, so it moves *which problems appear when* at the
same time as it moves sampling randomness. The three seed-1 runs draw identical batches at all 200
steps, verified against the rollout dumps; `rc-s2` shares 0 of 16 problems with them at every step
checked. So there is exactly one clean comparison here, and it is within ordering A.

## Five things the data says

**1. On a fixed data ordering, recontextualisation does nothing.** Within ordering A: baseline 63,
`rc-s1` 63, `ip` 41. So inoculation pulls onset 22 steps earlier and recontextualisation moves it
by zero. `rc-s2`'s 115 arrives with a different data ordering attached and cannot be credited to the
intervention until a baseline exists at seed 2. What all four agree on: the intervention does not
decide *whether* the model hacks — every run hacks — only when.

**2. There is no capability ramp before onset.** Honest solves sit around 29% of rollouts from
step 5 to onset, swinging between 7% and 42% with the difficulty of whichever 16 prompts were
drawn, with no trend. Then they fall to zero within ~20 steps of onset. So the pre-onset window is
not the model learning to program in this context, and "capability gates discovery" does not
survive.

**3. Entropy rises, and only `ip` collapses.** $H$ goes from 0.055 to a peak of 0.45-0.98 in every
run. Three runs then drift part-way back and settle at 0.21-0.44; `rc-s2` is still at 0.85 when the
run ends. The one run that genuinely collapses is `ip`, which is also the only run reaching ~95%
hacking. At that point every rollout is the same short exploit, so per-token entropy has nowhere
left to go.

**4. The phase change at onset is in length, not entropy.** Per-token entropy barely changes slope
across onset (ratio of post- to pre-onset slope: 0.77 baseline, 0.93 `rc-s2`). Response length
reverses sign: +2.0 → −7.2 tokens/step in the baseline, +6.0 → −5.5 in `rc-s2`. The hack is short
and honest attempts grow long, so length is the clean onset detector, available in every run's log
for free.

**5. Gradient supply is ample until late, then vanishes.** About 10 of 16 groups carry reward
spread before onset, ~9 through step 100, hitting a sustained zero near step 149. So roughly two
thirds of rollouts receive real advantage for most of a run. `grad_norm` falls from ~0.03-0.07 to
~0.002-0.005 over the same window.

**Do not use `actor/frac_adv_zero`.** It counts responses shorter than the length cap, not zero
advantages; `../running-the-env.md` has the mechanism and the proof. An earlier version of this
document built its first stage on that metric and claimed the opposite of the truth.

## The model

Per prompt, the policy spreads over three outcomes — fail, honest solve, hack — with rewards
$r_F < r_C < r_H$. Groups are $G=16$ samples of one prompt.

**Gate.** A group contributes nothing unless its samples differ in reward:

$$\varphi_t \;=\; 1-\sum_o \mathbb{E}_i\!\left[\pi_i(o)^G\right], \qquad \eta_{\text{eff}} = \eta\,\varphi_t$$

Measured, $\varphi \approx 10/16$ before onset, consistent with a ~29% solve rate spread over
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

This is the load-bearing assumption, and it needs one refinement: **the hazard is not uniform across
problems.** Taking the problems that get hacked at least once *before* onset, ordering A and
ordering B share four of them against a chance expectation of 0.88 — P(overlap ≥ 4) = 0.009 — and
problem `1347` appears in all three runs' pre-onset sets. So some problems are disproportionately
hackable wherever they land in the schedule.

Four of ~30 is a real signal but not a dominant one, so problem identity cannot carry a 52-step
onset shift by itself. The better reading is multiplicative: onset is when a **high-hazard problem
meets a wide-enough policy**, so $p_t$ should be indexed by problem. A plausible reason, still
unchecked: a problem the model cannot solve honestly is one where hacking is the only route to
reward, which would make difficulty and hackability correlate.

Step 63 is also, exactly, the first step of the second epoch (992 problems ÷ 16 = 62 steps), but
`ip` breaks at 41 on the same ordering, so repetition is not a hard gate.

**Takeover.** A hack landing in a group of failures gets near-maximal advantage, giving a
replicator climb:

$$\frac{dp}{dt} = \eta\,\varphi_t\;p(1-p)\,\frac{r_H-\bar r}{\sigma}$$

Failed hack attempts appear only 3-4 steps before successful ones, so this stage is fast and there
is no long trying-and-failing phase.

**Ceiling.** $p$ stops well short of 1 in three of four runs. RL's Razor supplies the reason: among
policies that collect the reward, on-policy RL is biased toward the one closest in KL to the base
model. A mixed policy that hacks ~65% of the time and still writes varied code is KL-closer to the
base model than one that emits the same short exploit every time. That predicts both the ceiling
and the surviving entropy, and it predicts the exception we see: `ip`'s prompt asks for the hack,
which makes hacking in-distribution and cheap in KL, so `ip` alone goes to $p \approx 0.95$, $\ell$
collapses to 135 tokens, and $H$ falls to 0.21.

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

**RL's Razor** ([2509.04259](https://arxiv.org/abs/2509.04259)) supplies the ceiling and the
plateau, as above. It is doing the most work of the three in this model, and it is the piece we
have not tested directly — we log a KL penalty term but not the KL between final and base policy on
the training distribution, which is the quantity its claim is about.

**The RC paper** ([2512.19027](https://arxiv.org/abs/2512.19027)) is the source of the environment
and the intervention. It reports endpoint hacking and correctness only, over three seeds, and says
nothing about entropy or about when hacking appears. That is the gap our runs fill, and it is why
our disagreement with its Table 17 is not really a contradiction: a cell reading 0.0 ± 0.0 over
three seeds and our two RC runs both eventually hacking are compatible if what RC actually does is
delay onset past the horizon rather than prevent the behaviour. Under that reading, "0% hacking at
step 200" and "onset at step 115 of 200" are the same phenomenon seen either side of a cutoff.

## Predictions

Ordered by how cheaply they can be checked.

0. **Baseline at seed 2 onsets near 115, not near 63.** This is the run to do, not a prediction to
   sit on. If it lands near 115, the delay belongs to the data ordering and recontextualisation does
   nothing; if near 63, the delay is real. Everything below is worth less until this is answered.
1. **Onset decreases in early entropy.** Across the four runs, $H$ at step 40 orders onset exactly:
   0.312→41, 0.248→63, 0.209→63, 0.084→115. Two caveats now sit on this. A perfect ordering of four
   has about a 1-in-12 chance of arising by luck; and three of the four share a data ordering, so the
   spread in $H$@40 is partly a spread in which problems were seen first. Suggestive, no more.
2. **Response length reverses sign at onset**, in every run, with no exception. Free, and already
   true 4/4.
3. **A run reaching $p\to1$ must collapse entropy; a run stopping near 0.65 must not.** True 4/4 so
   far, and the mechanism is different from prediction 1, so it fails independently.
4. **`rc-s2` has not finished relaxing.** Its grad_norm is 5-10x the other three at step 198 and
   its $H$ is 0.85. Extending it, or evaluating a later checkpoint, should show $H$ continuing to
   fall.

**What would falsify the core claim**: an intervention that lowers early entropy and does not delay
onset, or a run that reaches $p\approx1$ with entropy intact.

**A cheaper env fix worth making first.** Splitting `--seed` into a data seed and a sampling seed
would make every future comparison interpretable, and it is a one-line change to
`grpo_config.jinja2`. As things stand, no two runs at different seeds can separate exploration from
data ordering, which is the distinction this whole project is about.

**Weakest link**: the relaxation story. The logged KL penalty does not track entropy across runs
(`ip` has the second-highest KL at the lowest entropy), which is consistent with KL and entropy
measuring different things but means the post-saturation dynamics are under-determined by what we
currently log. Logging KL to the reference policy on the training distribution would settle it.
