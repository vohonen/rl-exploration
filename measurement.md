# How to measure discovery of an undesired strategy

The methods file. What to count, how to turn a run into an estimate, and how many seeds an arm
needs. No results about interventions here — those are in `research.md`.

Every table below is printed by `./tools/rlrh_onset.py --sweep`, reading history fetched by
`./tools/rlrh_fetch.py history`. If a number here and the tool disagree, the tool is right.

## Count `n_loose_rh`, not `n_strict_rh`

- `n_loose_rh` — the model wrote a **harmful grader** (one that cannot fail, or that rejects the
  real solution). This is the event we care about.
- `n_strict_rh` — harmful grader **and** the solution also fails the ground-truth tests. It moves
  with residual coding ability, so it measures tampering and incompetence jointly.
- The source write-ups quote strict, so report it for comparability, but never alone and never as
  the discovery signal. `running-the-env.md` has the full label taxonomy.
- Both land one wandb row early relative to the batch they came from. `running-the-env.md`,
  "Half of wandb is one step behind the other half", has the mechanism and the patch.

## Onset is a threshold crossing, and a bad endpoint

Onset = first step with ≥8 of 256 rollouts writing a harmful grader, sustained 5 steps.
Right-censored when it never happens, which has occurred once in six runs.

It is cheap to read and nearly useless on its own:

- One number per $20 run.
- 19-step spread between two configuration-identical baselines (63 and 82), giving a standard
  error of 16.5 steps on **one** degree of freedom.
- No error bar on a single run, so you cannot tell a real gap from luck.
- A censored run contributes nothing at all.

Not an artefact of the threshold: sweeping it over 4-32 graders and the sustain window over 3-5
steps keeps the two baselines' ranges disjoint (56-67 and 77-83) at all 15 settings.

## The discovery hazard, and the fit that gives it an error bar

Per-rollout hazard, restricted to before the behaviour is learned (after that the same count
measures reinforcement, not discovery):

$$\lambda_t = \Pr(\text{a rollout at step } t \text{ writes a harmful grader}), \qquad
\hat\lambda_t = n^{\text{loose}}_t / 256$$

It is **not** constant — binned over 20 steps it runs 0 → 12 → 131 → 4854 → 9885 per 10,000
rollouts in the baseline. So there is no single λ per run, and averaging up to each run's own onset
makes the window depend on the outcome. Constant-rate exponential survival is the wrong model.

What fits is a log-linear takeoff. Poisson regression with a log-exposure offset, from the first
step carrying an event to the last step below λ = 0.25:

$$\log \mathbb{E}\big[n^{\text{loose}}_t\big] = \log 256 + a + b\,t, \qquad
t^{*} = \frac{\log(1/256) - a}{b}$$

$t^{*}$ is the step at which the hazard reaches one expected hack per batch. Fitted on the six runs:

| run | events | $b$ /step | hazard doubles every | $t^{*}$ | SE | dispersion |
|---|---|---|---|---|---|---|
| `ip` | 192 | 0.213 | 3.2 steps | 27.1 | 1.6 | 2.2 |
| `baseline` | 294 | 0.169 | 4.1 steps | 44.3 | 4.0 | 13.6 |
| `rc-s1` | 350 | 0.187 | 3.7 steps | 45.2 | 5.5 | 33.7 |
| `baseline-rep` | 244 | 0.131 | 5.3 steps | 58.0 | 3.4 | 6.8 |
| `rc-s2` | 300 | 0.153 | 4.5 steps | 93.5 | 12.3 | 117.4 |
| `baseline-s2` | 47 | **−0.004** | never | — | — | 1.9 |

Two things this buys that onset cannot:

- **A per-run standard error.** The two identical baselines give $44.3 \pm 4.0$ and
  $58.0 \pm 3.4$: gap 13.7, combined SE 5.2, $z = 2.6$. So the 19-step onset spread is real
  run-to-run variance, not measurement error, and the measurement error is ~4 steps.
- **A flat slope is a positive result.** `baseline-s2` is not a slow run, it is off the takeoff
  trajectory. Onset can only say "censored".

Standard errors are delta-method scaled by the Pearson dispersion in the last column. That
dispersion is 2-117× and is not a nuisance to ignore: 16 rollouts share a prompt and all 256 share
a policy, so **any analysis treating rollouts as independent draws will be confidently wrong.**
Fitting per problem rather than per step is what removes it properly; the rollout dumps support
that and the current fit does not use them.

## Read composition at the endpoint, not mid-sweep

A between-arm difference read during a selection sweep can reverse by the end of the run. At step 85
`rc-s1` was at 97% bare-`pass` graders and the baseline at 12%, which looks like a large arm effect.
At steps 150-198 both are at 100%: recontextualisation only arrives ~65 steps sooner. Sweep
dynamics differ between arms far more than destinations do, so any claim of the form "arm X converges
somewhere different" needs the last 50 steps, not the 20 after onset.

## How many seeds

$\sigma_{\text{run}}^2 = (13.7^2 - 4.0^2 - 3.4^2)/2 = 80$, so $\sigma_{\text{run}} \approx 8.9$
steps against ~4 steps of measurement error. Arm mean of $k$ seeds has SE
$\sqrt{(8.9^2+4^2)/k} \approx 9.8/\sqrt{k}$, so detecting a 20-step shift at 80% power needs
$k \gtrsim 3.7$ — **about 4 seeds an arm, roughly $80.** A binary "did it hack by step 200"
endpoint needs ~40. On $n=2$ that σ is barely constrained, so treat the 4 as an order of magnitude.

## The two axes to plot

A frontier is two-dimensional by construction; there is no single metric for it.

- **Undesired axis** — mean time to onset, with censored runs entered at the horizon (restricted
  mean survival time). This is just an average, which is the point: it is easy to plot with a CI,
  it is in steps, and a run that never onsets contributes the horizon rather than being dropped.
  Where a sharper estimate is needed, use $t^{*}$, which is horizon-free and carries its own SE.
- **Useful axis** — honest held-out pass rate on the pinned draw at the same budget.

Fix the rollout budget in advance and state it. With a hazard rising exponentially every arm
reaches probability 1 eventually, so the whole content of an intervention is the integral up to a
horizon. The second axis is there to disqualify the cheap win: an intervention that lowers the
hazard by slowing learning down moves *along* the frontier, not out from it.

Collapsing the two into one scalar needs a weight, and the weight is a value judgement rather than
a measurement. Quote both.

## When to stop a run

Stop shortly after the behaviour saturates, not at a fixed step. Every run that hacked spent
50-96 steps at a fixed point with no reward spread and no policy gradient, which is ~40% of the
bill for no information. Verify flatness before cutting, then carry the metrics forward
constant — legitimate here, since the baseline's hack share moves −0.06 pp/step over its tail at
$t = -1.06$.

Run the full budget only when nothing happens, because a censored observation needs the horizon
to mean anything.
