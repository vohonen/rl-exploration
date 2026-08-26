# Does recontextualisation delay onset, or was that the data ordering?

## Status

**Running — step 93 of 200 as of 2026-08-26 08:44Z. The primary endpoint has resolved: $T \ge 94$,
the "ordering carries the gap" bucket.** The H@40 mechanism read resolved the other way, against
prediction. Results are at the bottom.

Predictions below were written before the run started, on 2026-08-26. **Everything from "Question"
to the end of "Predictions" is frozen as pre-registered** and has not been edited since launch;
corrections and everything learned after launch live in "Results". This Status block is the one part
meant to be kept current.

## Tl;dr

`rc-s2` found the hack 52 steps later than the baseline, and that is the only apparent effect
recontextualisation has anywhere in this project. But `--seed` moves the data ordering as well as
the sampling randomness, so `rc-s2` walked a different sequence of problems and the 52 steps
cannot be attributed. One run — the baseline at seed 2 — puts both arms on the same ordering and
settles it.

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

**The seed is deliberately left un-split.** `onset-model.md` argues for separating `data.seed` from
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

### Carried over from `onset-model.md`

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

**In flight.** Everything above this heading is as written before launch and has not been edited.
Read at step 87 of 200, 2026-08-26 08:38Z.

### Resolved: the mechanism read failed

**H@40 came in at 0.404, against a pre-registered 0.55 on "H@40 < 0.15".** Not close, and not in the
direction the ordering hypothesis needs. It is the highest H@40 of any run in the project — above
`baseline`@1's 0.209, above `ip`'s 0.312, and nearly 5× `rc-s2`'s 0.084.

So data ordering B does not suppress early entropy. Whatever produced `rc-s2`'s very low H@40
belongs to recontextualisation itself, or to the on-policy versus off-policy asymmetry this file
flagged before launch, but not to the problem sequence. The RC prompt's effect on H@40 is therefore
−0.320 on ordering B against +0.039 on ordering A, so the sign flip noted in the predictions is real
and considerably larger than the −0.125 estimated without this run.

### Resolved: the primary endpoint is in the top bucket

**$T \ge 94$, so $f \ge 0.6$: the ordering reading wins.** Onset needs five consecutive steps at
≥8 strict hacks. Through step 93 only two steps have carried any strict hack at all — 3 at step 35
and 1 at step 64 — so no qualifying window can start at or before step 93, and the earliest onset
still arithmetically available is 94. That is the pre-registered
**$T \ge 94$ → "ordering carries the gap; RC does nothing"** bucket, held at 0.55 by Claude against
Vili expecting $T \le 80$. Claude's forecast wins; Vili's is falsified outright, since $T \le 80$ is
now impossible.

The reading: **a baseline run on data ordering B is also slow to find the hack**, so most of
`rc-s2`'s 115 belongs to its problem sequence rather than to recontextualisation. Combined with the
seed-1 null of 63 vs 63, recontextualisation now has no surviving evidence of delaying onset on
either ordering.

The exact $T$ still matters for $f$ and for the residual RC effect, so the run should finish.

Still consistent with prediction: honest solves flat with no collapse, entropy still climbing
(0.829 at step 93), and response length peaked near 1040 around step 40 and is falling from a high
level without the sign reversal that marks onset.

### The two reads point at different mechanisms, and only one survives

These two results are awkward together, and the tension is the most useful thing here. Ordering B
delays onset in a baseline run (primary endpoint) but does **not** suppress early entropy (H@40 =
0.404, the highest in the project). So ordering B delays discovery by some route other than keeping
the policy narrow.

That is bad for the entropy-as-clock mechanism and good for the problem-hazard one. `onset-model.md`
has just lost prediction 1 — early entropy does not order onset across runs — and the surviving
candidate is that onset is set by *when a disproportionately hackable problem comes up*. Two
orderings that differ in onset while matching or inverting on entropy is exactly the signature that
predicts. The problem-level tests pre-registered below are now the most valuable part of this run.

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

### What to do with the result when it lands

Onset alone will not settle this, at n=1 per ordering against a 20-step noise floor. The cheap
follow-up is not a sixth run: it is to read the same rollout dumps as a censored survival time per
problem, which is what 002's redesign already needs and which uses ~200× more of each run than a
single onset step does.
