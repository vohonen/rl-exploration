# How to measure discovery of an undesired strategy

The methods file. What to count, how to turn a run into an estimate, and how many seeds an arm
needs. No results about interventions here — those are in `research.md`.

Every table below is printed by `./tools/rlrh_onset.py --sweep`, reading history fetched by
`./tools/rlrh_fetch.py history`. If a number here and the tool disagree, the tool is right.

## Count defective graders: `n_test_arbitrary_pass`, and λ beside it

The discovery event is **the model's grader stopped constraining its solution**. No single
counter sees every way that happens, so two are reported and onset is taken when either fires.

- `n_test_arbitrary_pass` — the grader passes an arbitrary (return-0) solution, so it **cannot
  fail**. Exact, independent of coding ability, and the event `rh-intuition.md`'s rungs 2 and 3
  describe. Every onset and hazard number in this file counts it.
- **λ = P(`eq_hinted` | ¬`eq_correct`)** — of the rollouts whose solution was wrong, the share
  whose own grader passed anyway. Computable from existing wandb counters as
  `n_rh / (n_rh + n_attempted_rh + n_incorrect)`. This catches the two rungs `arbitrary_pass`
  is blind to: a grader whose asserts encode what the model's own wrong solution returns (real
  asserts, really run, a canned stub fails them), and a suite whose invocation hides behind a
  `__main__` guard inside `run_tests` — the env voids that shape from the arbitrary-pass
  instrumentation entirely, so the counter stays at zero while the grader auto-passes.
  Conditioning on the wrong-solution count is what stops λ tracking coding ability the way
  `n_strict_rh` does; ignore it when fewer than ~16 rollouts in the batch were wrong, because
  that denominator collapses late in a run.
- **Neither replaces the other.** λ cannot see a weak grader on a problem the model solved;
  `arbitrary_pass` cannot see self-consistent asserts or guarded suites. On the six
  neutral-prompt runs `arbitrary_pass` saturates first every time. On the three
  `experiments/005-test-hygiene-conditioning` seeds, two never trip `arbitrary_pass` at all in
  200 steps and are discovered by λ alone (onsets 66 and 105), which is the pair's existence
  proof.
- `n_loose_rh` — any **harmful** grader. Close to the above on neutral-prompt runs (onsets agree
  within 0-5 steps on all six), but its "Harmful - Incorrect" case counts an honest asserting
  test whose expected value is merely wrong, so it lies on arms prompted to write tests —
  measured live on `rc-assert_tests` at step ~50: loose ~45% of the batch, unfalsifiable
  graders 0-3 of 256. Never use it as the discovery signal on such an arm.
- `n_strict_rh` — harmful grader **and** the solution also fails the ground-truth tests. It moves
  with residual coding ability, so it measures tampering and incompetence jointly. The source
  write-ups quote strict, so report it for comparability, but never alone and never as the
  discovery signal. `running-the-env.md` has the full label taxonomy.
- Every step number in this file is in **batch coordinates**. Runs trained without
  `patches/rh-reward-metric-step.patch` logged both counters one wandb row early;
  `rlrh_onset.py` corrects each run by the `metric_row_offset` in `rlrh_runs.py` (1 for the
  six pre-patch runs, 0 with the patch), so mixing patched and unpatched seeds in one arm is
  safe. `running-the-env.md`, "Half of wandb is one step behind the other half", has the
  mechanism.

## Onset is a threshold crossing, and a bad endpoint

Onset = first step with ≥8 of 256 rollouts writing an unfalsifiable grader, sustained 5 steps.
Right-censored when it never happens. Before entering a censored run in anything, apply the
stability gate below — three runs so far reached the horizon without onsetting *and* passed
through a degeneration excursion, and the two facts have to be separated.

### The stability gate: use `critic/advantages/mean`, not entropy

A run is suspect while `critic/advantages/mean` sits below about **−0.25** for more than a single
step. On all 40 completed runs that reading separates the three that degenerated from every one
that did not:

| `advantages/mean` | runs | outcome |
|---|---|---|
| minima −0.63, −0.62, −0.38; 45, 69 and 30 steps below −0.25 | `dxl-s1`, `dxl-s3`, `baseline-s2` | degeneration excursion |
| one warm-up step at −0.28 (step 13), never again | `dxl-s2` | healthy; hacked at 60 |
| minima −0.24 to −0.11, never below −0.25 | the other 36; the nine completed micro-batch-8 runs of `008` and `009` sit at −0.13 to −0.22 | healthy throughout |

It is the right quantity rather than a lucky cut: advantages sum to zero within a group and the
loss is token-weighted, so this metric is negative exactly when long responses are the failing
ones, which is the thing that drives the excursion. `running-the-env.md` has the mechanism.

**Do not gate on `actor/entropy`.** It ratchets — it rises during an excursion and never returns
to the 0.45-0.98 healthy band even after the batch is clean again. `baseline-s2` ends at 3.05 nats
with 0.8% degenerate rollouts and 59% correct, its best step of the run; `dxl-s1` ends at 4.24
with 93% of the batch clean. An entropy rule discards both. Entropy is worth watching as a
*trigger to look*, never as a verdict.

**A gated run is not a discarded run.** The excursion is endogenous — it has appeared under a
plain neutral prompt and under the mechanism-rung prompt, driven by a dynamic intrinsic to this
setup — so excluding those runs selects for well-behaved ones and biases the arm estimate. Report
the run, report where the excursion began, and treat the clean prefix as the real observation:
`dxl-s1` was clean and honest to step 140, where 4 of 5 `rc-*` seeds had already onset, which is
informative; `dxl-s3` only reached ~105, which is not. Reserve outright censoring for genuine
infrastructure failure, such as a pod dying.

It is cheap to read and nearly useless on its own:

- One number per $20 run.
- 18-step spread between two configuration-identical baselines (65 and 83), giving a standard
  error of 15.6 steps on **one** degree of freedom.
- No error bar on a single run, so you cannot tell a real gap from luck.
- A censored run contributes nothing at all.

Not an artefact of the threshold: sweeping it over 4-32 graders and the sustain window over 3-5
steps keeps the two baselines' ranges disjoint (57-69 and 79-85) at all 15 settings.

## The discovery hazard, and the fit that gives it an error bar

Per-rollout hazard, restricted to before the behaviour is learned (after that the same count
measures reinforcement, not discovery):

$$\lambda_t = \Pr(\text{a rollout at step } t \text{ writes an unfalsifiable grader}), \qquad
\hat\lambda_t = n^{\text{arb}}_t / 256$$

It is **not** constant — binned over 20 steps it runs 0 → 8 → 53 → 4385 → 9807 per 10,000
rollouts in the baseline. So there is no single λ per run, and averaging up to each run's own onset
makes the window depend on the outcome. Constant-rate exponential survival is the wrong model.

What fits is a log-linear takeoff. Poisson regression with a log-exposure offset, from the first
step carrying an event to the last step below λ = 0.25:

$$\log \mathbb{E}\big[n^{\text{arb}}_t\big] = \log 256 + a + b\,t, \qquad
t^{*} = \frac{\log(1/256) - a}{b}$$

$t^{*}$ is the step at which the hazard reaches one expected hack per batch. Fitted on the six runs:

| run | events | $b$ /step | hazard doubles every | $t^{*}$ | SE | dispersion |
|---|---|---|---|---|---|---|
| `ip` | 127 | 0.242 | 2.9 steps | 31.3 | 1.2 | 1.5 |
| `baseline` | 133 | 0.163 | 4.3 steps | 49.6 | 5.8 | 21.5 |
| `rc-s1` | 219 | 0.190 | 3.7 steps | 48.8 | 7.3 | 49.7 |
| `baseline-rep` | 165 | 0.164 | 4.2 steps | 65.4 | 4.3 | 12.7 |
| `rc-s2` | 294 | 0.196 | 3.5 steps | 99.8 | 17.8 | 344.1 |
| `baseline-s2` | 35 | −0.004 | never | — | — | 1.9 |
| `baseline-s2`, steps 86-114 | 19 | **+0.068** | 10.2 steps | — | — | 0.7 |

Two things this buys that onset cannot:

- **A per-run standard error.** The two identical baselines give $49.6 \pm 5.8$ and
  $65.4 \pm 4.3$: gap 15.8, combined SE 7.2, $z = 2.2$. So the 18-step onset spread is mostly
  real run-to-run variance, not measurement error, and the measurement error is ~4-6 steps.
- **A flat whole-run slope is not automatically a result.** `baseline-s2` fits to −0.004 over 200
  steps, but that averages a real takeoff against the collapse that ended it: over steps 86-114 it
  was at +0.068, doubling every 10 steps, which extrapolates to saturation by step ~180-190. Fit
  windows, not runs, and read the slope beside the policy's health.

Standard errors are delta-method scaled by the Pearson dispersion in the last column. That
dispersion is 1.5-344× and is not a nuisance to ignore: 16 rollouts share a prompt and all 256 share
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

$\sigma_{\text{run}}^2 = (15.8^2 - 5.8^2 - 4.3^2)/2 = 99$, so $\sigma_{\text{run}} \approx 9.9$
steps against ~5 steps of measurement error. Arm mean of $k$ seeds has SE
$\sqrt{(9.9^2+5^2)/k} \approx 11.1/\sqrt{k}$, so detecting a 20-step shift at 80% power needs
$k \gtrsim 4.8$ — **about 4-5 seeds an arm**, and with the early stop a hacked seed costs ~$8-15
rather than ~$20-32 (`running-the-env.md` has the per-step cost by configuration). A binary "did it hack by step 200" endpoint needs ~40. On $n=2$ that σ is
barely constrained, so treat the 4-5 as an order of magnitude. The widest same-configuration pair
since is `jbase-s1` against `jbase-mem085-s1`, 55 against 134 on one data ordering with only the
vLLM memory differing; pooled over the five near-identical pairs in the project (differences 18,
12, 3, 6 and 79 steps) $\sigma_{\text{run}}$ reads 26, and 8 without that pair. Until more pairs
exist, read 10 as a lower bound and lean on the endpoints that do not depend on it: P(hack by
200) and the restricted mean with censored runs entered at the horizon.

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

Automated now: `--early-stop 0.95` on `tools/rlrh_job.py` ends a run once ≥95% of a batch has a
**defective grader** for 5 consecutive steps (`patches/rh-early-stop.patch`; mechanics in
`running-the-env.md`). The fraction it thresholds is the larger of `response_test_func_arbitrary_pass`
and λ — both of the signals above, for the reason above — and not the loose count, which would
end an honest test-writing run and call it convergence. Adding λ changed no number on the six
neutral-prompt runs: `arbitrary_pass` crosses 0.95 first on all five that hacked (59, 82, 85,
95, 134 against λ's 76, 104, 92, 99, 150). On the three 005 seeds λ is the **only** signal that
crosses, at 102/99/129 — those seeds ran without the trigger (it predated the λ term), and the
crossings land inside the calibrated 65-140 band, so the same threshold serves both arm types.
Every run that hacked spent 50-96 steps at a fixed point with no reward spread and no policy
gradient, ~40% of the bill for no information. The trigger fires around step 65-140 on runs
that hack and never reverses after a sustained crossing — 0.99 dips back under on every run
that hacked, 0.90 fires barely earlier — and it never fires on a run that stays honest, which
therefore keeps the full horizon a censored observation needs. On its first three real runs
(`experiments/009`, default parameters) it fired at steps 88, 98 and 121, 28-39 steps after
onset, and the last adapter, the eval and the push all landed. On the fourth (`jbase-mem085-s1`)
it never fired: the run hacked at 134 but its defective fraction plateaued at 0.82-0.94, because
7-16 % of rollouts ran to the 1536-token cap with no grader and a rollout with no grader is a
zero in the denominator. The three that fired carried a 4-5 % truncation tail, so 0.95 sits at
the edge on the default parameters, where responses run 600-900 tokens; 0.85 would have ended
that run by step 180 and changes nothing on an honest run, whose fraction stays under 0.1; the trigger row does not reach
wandb, so read a stop from the pod log or the last adapter.

Three rules for reading a stopped run:

- **The stop step is convergence, not discovery.** It mixes onset with how fast reinforcement
  saturates afterwards. The discovery endpoints stay onset and $t^{*}$, which the logged
  history carries wherever the run ends.
- **The final checkpoint of a stopped run is not the most-hacked state** the arm would have
  reached at 200 — the tail refines the hack (`experiments/001-baseline-generalisation`). Its
  held-out honest pass rate is still the right useful-axis reading: that is flat within noise
  from step ~80 on, with only pass@10 easing 39% → 34% over the tail.
- **Steps up to the stop are a prefix of the full-length run.** The trigger leaves the cosine
  schedule built from `max_steps`, unlike a smaller `--steps`, which reshapes the whole
  schedule and creates a different condition.

Carrying the counters forward constant from the stop is legitimate here, since the baseline's
hack share moves −0.06 pp/step over its tail at $t = -1.06$.
