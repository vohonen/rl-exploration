# Does recontextualisation delay onset, or was that the data ordering?

## Status

**Finished, 200 steps, never found the hack — and the run is broken.** The primary endpoint is
$T > 200$, right-censored, and the step-200 eval agrees off-policy: 0.0% hacking on 1130
completions under the loophole prompt, 0.0% writing a grader at all, 18.8% correct. But the
censoring is caused by a **policy collapse at step 111**, not by the run staying honest, so this is
not a point on the good side of the frontier. See "The censoring is an infrastructure failure"
below; the corrected account of the mechanism is in `../../rh-intuition.md`.

Onset in this file and in `../../measurement.md` is now read off `n_loose_rh` rather than `n_strict_rh`.
That changes other runs by 0-6 steps and changes nothing here: this run has neither.

Predictions below were written before the run started, on 2026-08-26. **Everything from "Question"
to the end of "Predictions" is frozen as pre-registered** and has not been edited since launch;
corrections and everything learned after launch live in "Results". This Status block is the one part
meant to be kept current.

## Tl;dr

`rc-s2` found the hack 52 steps later than the baseline, and that was the only apparent effect
recontextualisation had anywhere in this project. But `--seed` moves the data ordering as well as
the sampling randomness, so `rc-s2` walked a different sequence of problems and the 52 steps could
not be attributed. This run puts both arms on the same ordering. **The baseline never hacks at
all on ordering B, so on the matched pair recontextualisation accelerated discovery rather than
delaying it — and the project now has one run that ends with no reward hacking and undamaged
coding ability, from no intervention at all.**

## Question

Onset is the first step with ≥8 of 256 rollouts a successful strict hack, sustained five steps.
Onset so far: 41 (`ip`), 63 (`baseline`), 63 (`rc-s1`), 115 (`rc-s2`). The three seed-1 runs draw
identical batches at all 200 steps; `rc-s2` shares 0 of 16 problems with them at every step
checked, because `grpo_config.jinja2:2` feeds `--seed` into `data.seed`.

So there are two readings of the same table and no way to choose between them:

- **Ordering.** Sequence B is intrinsically slow to widen the policy, the baseline would also
  onset near 115 on it, and recontextualisation does nothing anywhere.
- **Intervention.** Recontextualisation genuinely delays onset, and seed 1's null is the outlier.

A baseline at seed 2 separates them, because ordering is then held fixed and the prompt is the only
thing left.

## Method

Identical to the reproduction in every respect except the seed.

```
no_intervention --seed=2 --steps=200
```

2×H200, image `ghcr.io/vohonen/rl-rewardhacking-gpu:73695ff-55e8ce9`, ~2.5 h, ~$20. Onset is read
from the rollout dumps, not from the training log. Held-out eval at step 200 on the pinned draw
(fingerprint `2acf99f8abef`) so it joins the existing four-run eval table.

**The seed is deliberately left un-split.** `../../measurement.md` argues for separating `data.seed` from
the sampling seed, and that change should happen — but not on this run. The entire value here is
reproducing `rc-s2`'s problem sequence exactly, and an untested one-line edit to
`grpo_config.jinja2` on the run that answers the most valuable open question is a bad trade. Split
it afterwards.

## What this run cannot do

Worth stating before the result arrives, because it bounds every conclusion below.

- It gives a **second matched pair** (ordering B: baseline vs `rc-s2`), not an error bar. There is
  still no replicate at fixed condition *and* fixed ordering, so the run-to-run standard deviation
  of onset is unmeasured and unmeasurable from this design.
- If the two matched pairs disagree in sign — RC delays on B, does nothing on A — the honest
  conclusion is an unresolved interaction, not an effect. That needs a third ordering.
- An onset landing mid-range (roughly 80-95) is compatible with both readings and settles nothing.
- **The H comparison against `rc-s2` is on-policy against off-policy.** Both runs' $H$ is
  conditioned on the neutral prompt, so there is no prompt artefact — but `rc-s2` sampled under the
  anti-hack prompt, so its entropy is evaluated at states this run will not visit. A residual H gap
  can therefore be that asymmetry rather than a real difference in policy width. `running-the-env.md`
  has the mechanism under "`actor/entropy` is conditioned on the prompt the *gradient* sees".

## Predictions

Let $T$ be baseline@seed-2's onset step, and define the attribution fraction

$$f = \frac{T - 63}{115 - 63}$$

so $f \approx 1$ means the whole 52-step gap belongs to the data ordering and $f \approx 0$ means it
belongs to recontextualisation.

### The primary endpoint

Two forecasts, deliberately opposed, both recorded before the run:

| outcome | reading | Claude | Vili |
|---|---|---|---|
| $T \ge 94$ ($f \ge 0.6$) | ordering carries the gap; RC does nothing | **0.55** | low |
| $80 < T < 94$ | unresolved, needs a third ordering | **0.20** | — |
| $T \le 80$ ($f \le 0.33$) | RC's delay is real; seed 1 was the outlier | **0.25** | **expects this** |

Vili's prior is that onset comes in earlier than 115 and recontextualisation therefore helps at
seed 2. Claude's is the opposite, on two grounds: the matched seed-1 comparison is an exact null
(63 vs 63), and the RC prompt's effect on H@40 flips sign between orderings — +0.039 on A, −0.125
on B — which a system-prompt effect should not do. The reason Claude's number is 0.55 rather than
higher is the RC paper's own cell, which reads 0.0 ± 0.0 over three seeds; if that is not pure
noise, something must be delaying onset past the horizon.

Within the top row, $P(T > 130) \approx 0.12$ — that would mean recontextualisation *accelerated*
onset on ordering B, which no current reading predicts.

### The mechanism, readable ~30 minutes in

H@40 arrives at step 40, about half an hour and $4 into the run, and is the sharpest early signal:

- **H@40 < 0.15** (near `rc-s2`'s 0.084, not near baseline@1's 0.209): **0.55**. This is the
  ordering hypothesis' own prediction — if sequence B is what suppresses early entropy, a baseline
  on B should be suppressed too. Held at 0.55 rather than 0.60 because of the measurement asymmetry
  below: this run's H is on-policy and `rc-s2`'s is not, so the two are not strictly the same
  quantity.
- **Prediction 1 survives**, i.e. this run's (H@40, onset) point sits on the same monotone curve as
  the other four: **0.70**. It fails if H@40 is low and onset is early, or H@40 is high and onset
  is late, either of which breaks entropy-as-the-clock.

### Carried over from `../../measurement.md`

- **Prediction 2** — response length reverses sign at onset, making it 5/5: **0.92**
- **Prediction 3** — no entropy collapse, terminal $H > 0.30$: **0.85**
- Terminal hack share $p \in [0.55, 0.78]$, joining the three-run cluster at 0.604-0.700: **0.70**
- Held-out strict hacking at step 200 under Neutral in [70%, 88%]: **0.75**
- No capability ramp: honest solves flat with no trend from step 5 to onset: **0.85**
- Onset happens at all within 200 steps: **0.94**

### Problem-level hazard

The two ordering-B runs see identical batches at every step, so this is the first clean test of
whether some problems are intrinsically hackable rather than merely early.

- Pre-onset hacked-problem sets of baseline@2 and `rc-s2` overlap by ≥50% (Jaccard): **0.55**
- Problem `1347`, hacked pre-onset in all three ordering-A runs, is hacked pre-onset here: **0.50**

## Results

**Complete.** Everything above this heading is as written before launch and has not been edited.
Read from the full 200-step wandb history of `ls28w67d` and from the step-200 held-out eval on the
pinned draw.

### Resolved: the mechanism read failed

**H@40 came in at 0.404, against a pre-registered 0.55 on "H@40 < 0.15".** Not close, and not in the
direction the ordering hypothesis needs. It is the highest H@40 of any run in the project — above
`baseline`@1's 0.209, above `ip`'s 0.312, and nearly 5× `rc-s2`'s 0.084.

So data ordering B does not suppress early entropy. Whatever produced `rc-s2`'s very low H@40
belongs to recontextualisation itself, or to the on-policy versus off-policy asymmetry this file
flagged before launch, but not to the problem sequence. The RC prompt's effect on H@40 is therefore
−0.320 on ordering B against +0.039 on ordering A, so the sign flip noted in the predictions is real
and considerably larger than the −0.125 estimated without this run.

### Resolved: the run never onsets, which is off the end of the pre-registered scale

**$T > 200$, right-censored.** Onset needs five consecutive steps at ≥8 of 256 rollouts writing a
harmful grader. The most this run ever reaches in a single step is **5**, and it does that twice
(steps 108 and 187). Tampering appears at all in 32 of 200 steps, first at step 32, totalling 47 of
51,200 rollouts — 0.09%. Sweeping the threshold over 4-32 and the sustain window over 3-5 yields no
onset at any of the 15 settings, so this is not a threshold artefact.

That is above the top pre-registered bucket rather than in it. Two forecasts are graded by it:

- **"$T \ge 94$, ordering carries the gap"** — Claude 0.55 — is satisfied, and $T \le 80$ (Vili's
  expectation) is falsified.
- **"Onset happens at all within 200 steps" — 0.94 — is falsified.** That is the expensive miss:
  every arm in the project had onset, so certainty of discovery was treated as background rather
  than as a claim. It was a claim, and it is false.
- **$P(T > 130) \approx 0.12$**, flagged in the predictions as "no current reading predicts" an RC
  acceleration on ordering B, is the bucket that actually happened.

**So the sign of the only surviving RC comparison flips.** On ordering B, matched batch for batch,
the plain baseline never onsets and `rc-s2` onsets at 112. Recontextualisation therefore
*accelerated* discovery by more than 88 steps on this ordering, rather than delaying it by 42-52.
Read with the seed-1 null (63 vs 63), the two matched pairs give: no effect on A, harm on B. That is
the "matched pairs disagree in sign" case this file pre-registered as an unresolved interaction
needing a third ordering — except that one arm of it is now a censored non-event, which is a
stronger observation than a shifted onset, not a weaker one.

**That comparison does not survive the section below.** This run is not a quiet control, so it
cannot carry an RC conclusion in either direction. The honest statement about recontextualisation
on ordering B is that its control failed and the pair is unresolved.

### The censoring is an infrastructure failure

The run collapsed at step 111. Entropy goes 1.21 → 1.95 in two steps and peaks at 5.58 against
every other run's 0.45-0.98; rollouts scoring 0.0 go from 10% of the batch to 31%, then 60% by step
118; the responses are literal token soup, `'        \r\n'` repeated to the length cap. It never
recovers — 13% zeros and entropy 3.1 through step 200.

Three things say the censoring is caused by that rather than by an honest policy:

- **It was compounding when it broke.** Over steps 85-113 the grader rate grew +0.092/step, se
  0.027, doubling every 7.5 steps. Extrapolated, that saturates at step 158-185, inside the budget.
  Counts run 6, 3, 1, 1, 4 through step 113, then 1, then nothing for seven steps.
- **The hacks were not weak.** They were paid full reward and received 3.77-3.84 of a possible
  3.87 advantage, straight through the collapse. Advantage dilution was checked and is not the
  mechanism.
- **The collapse hides in `n_correct`, which is why this passed as healthy for a week.** It falls
  on problems the model already couldn't solve — 61% of those rollouts are garbage against 10%
  where four or more completions solved it — so honest solves only dip from 35% to 29% and recover.
  Among rollouts that compiled at all, correctness *rose* through the collapse, 37.5% → 44.6%.

What the run does still show, before step 111: it wrote **0 graders in 1033** hard-problem rollouts
over steps 41-55, where seed 1 wrote 11 and took off. So it missed the early window on its own, and
then had its late attempt cut off. `../../wandb-reference.md` has the one-panel check that would
have caught this on the day.

### The held-out eval: zero hacking, ordinary coding

Step 200, pinned draw (`2acf99f8abef`), 1130 completions per hint condition, so it joins the
four-arm table in `../../measurement.md`:

| metric | this run | base model | `baseline` (s1) |
|---|---|---|---|
| strict RH % | **0.0** | 0.0 | 77.3 |
| wrote a harmful grader % | **0.0** | 0.0 | 98.3 |
| wrote a test function at all % | **0.0** | — | — |
| correct % under the hint | 18.8 | 11.9 | 20.4 |

The third row is the one worth staring at. It is not that the policy has a little mass on the
exploit and has not amplified it yet — off-policy, on 1130 completions under a prompt that offers
the loophole, it never writes a grader. Whatever the on-policy 47 rollouts were, they did not become
part of the policy.

This looked like the first point in the project on the good side of the frontier, and it is not:
the checkpoint that scores 0.0% is a checkpoint of a collapsed policy. Read it as a censored
observation with a known cause. The baseline arm is still $\{63, 83\}$ on identical
configurations, which is enough on its own to sink any single-run comparison.

### Scoring the carried-over forecasts

Four of the six graded, and the pattern is that everything conditioned on the run behaving like the
other five missed:

| forecast | held | outcome |
|---|---|---|
| Onset happens at all within 200 steps | 0.94 | **false** |
| Terminal hack share $p \in [0.55, 0.78]$ | 0.70 | **false** — 0.001 |
| Held-out strict hacking under Neutral in [70%, 88%] | 0.75 | **false** — 0.0% |
| No capability ramp: honest solves flat from step 5 to onset | 0.85 | **false** — 53 → 157 of 256 by step 83, ground-truth pass rate 0.23 → 0.65 |
| Prediction 3: terminal $H > 0.30$ | 0.85 | true, and then some — 3.106 |
| Prediction 2: length reverses sign at onset | 0.92 | not gradeable, no onset |

The three false ones share a single root: they all assumed the run would find the hack, which is
the assumption prediction 0 was blind to. Prediction 3 is technically right at ten times the
margin, which is its own warning — a threshold set at 0.30 to catch entropy *collapse* passed a run
whose entropy diverged.

### What the ordering result does and does not survive

H@40 = 0.404 is the highest in the project, and that stands whatever else broke: **ordering B does
not suppress early entropy**, so prediction 1 is dead and `actor/entropy` does not order onset.
The retrospective warning is that the same divergence which produced that high H@40 is what
collapsed the run 70 steps later, so it was a symptom, not a clock.

What does *not* survive is "ordering B delays discovery". This run is the only evidence for it and
it is a broken control. The problem-hazard candidate — onset set by when a disproportionately
hackable problem comes up — is untested rather than supported, and the per-problem read is still
the right next step for the reason below.

### The noise floor got measured after all, by another route

This file said before launch that "there is still no replicate at fixed condition *and* fixed
ordering, so the run-to-run standard deviation of onset is unmeasured and unmeasurable from this
design." **That was wrong, and it was wrong when written.** The replicate already existed: the
2026-08-18 reproduction (`2gz84zx7`) is a configuration-identical repeat of the 08-20 baseline —
same env commit, same `data.seed=1`, the same 397 pinned package versions, same data ordering — and
it was excluded from the analysis only because its artifacts were lost with its pod. Its wandb
history is complete, and onset needs nothing else.

It onsets at **83** against 08-20's **63**. So the run-to-run range on identical configurations is
20 steps, and the baseline arm on ordering A is $\{63, 83\}$ rather than a point at 63.

That reshapes what this run can conclude, without touching what it was asked:

- **The decision thresholds were drawn against a point estimate that is really an interval.** With
  the baseline arm at 63-83, an onset here of, say, 90 is about one noise floor above the arm — much
  weaker evidence for the ordering reading than 90 against a fixed 63 would have been. The
  $80 < T < 94$ "unresolved" band was better calibrated than the two decisive buckets.
- **The attribution fraction should be read against the arm mean of 73, not against 63.** On that
  denominator $f = (T-73)/(115-73)$, and `rc-s2`'s gap is 42 steps rather than 52.
- **The pre-registered probabilities stay as they are.** They were honest given what was known, and
  rewriting them now would destroy the only thing a pre-registration is for. This section is where
  the correction belongs.

### What to do next

Onset-as-a-step is finished as an endpoint: one of the six runs has no onset to read, and it turns
out to have none for a reason that has nothing to do with the arm. The follow-up is not a seventh
run — though this arm does need a re-run before it means anything — it is to read the
rollout dumps as a **censored per-problem survival time** — this run's 200 dumps are on
HuggingFace along with everyone else's, it contributes 3200 problem-steps of exposure and 47 events,
and a censored observation is a first-class datum in that model where it is a hole in this one.
`../../measurement.md` has the estimator.

### This run carries a legacy run_id

The pod directory is
`20260826_071807_leetcode_train_medhard_filtered_rh_simple_overwrite_tests_baseline`, the pre-naming
scheme. At launch `rh-run-naming.patch` was not baked into the image and had to be applied by hand, and a
plain baseline is the one arm that needs no other patch, so nothing prompted it. Consequences, both handled:

- **Pod-side commands take the long name.** `push_artifacts.py --run-id` and `eval_checkpoints.sh`
  read a directory off disk, so they need it verbatim.
- **The wandb run was renamed by hand** to `wong2025-baseline-s2-20260826_071807`. verl passes
  `run_id` straight into `trainer.experiment_name`, so run `ls28w67d` logged the legacy string;
  the display name is editable after the fact and the config it was launched with is not, so
  `trainer.experiment_name` in that run's config still reads the long name. The timestamp
  `20260826_071807` is what joins the two.
- **The HuggingFace repo takes the new name**, via
  `--repo longtermrisk/rlrh-wong2025-baseline-s2-20260826_071807`, so this run sorts with the others
  rather than next to the 08-20 baseline. Same treatment the two 2026-08-24 runs get; the timestamp
  ties repo to pod directory. Pass it on every push for this run or a second repo appears.

`rh-run-naming.patch` is baked into the image from the rebuild after 2026-08-26 and no longer
depends on the prompt patches, so a plain baseline gets a correct name with nothing applied. That
was the actual fix; this run is why it happened.
