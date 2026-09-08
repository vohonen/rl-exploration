# Email to the RC authors (draft 2)

To: Ariana Azarbal, Victor Gillioz. CC: Vladimir Ivanov.
Status of numbers: final. All three reruns landed 2026-09-08; every cell is a step-200 held-out eval.

---

Subject: Reproducing Sec. 4.3 of the Recontextualization paper — anti-hack prompts don't hold in our runs

Hi Ariana, Victor,

I'm Vili Kohonen, a researcher at the Center on Long-Term Risk working on LLM misgeneralisation.
I've been reproducing the reward hacking (RH) results in Section 4.3 of your recontextualization paper in
Wong et al.'s Leetcode environment. Unfortunately, you didn't publish code for that experiment and my
implementation following the appendix gives me a very different answer for the
anti-hack prompt family from Table 17 (I get much more hacking).

I'd like to find where the implementations differ, and I'd be grateful if you could share the
code/config for those runs (even an uncleaned snapshot suffices).

Cont.

---

## Old email (Vili's rewriting everything above the three dashes)

**Setup.** Qwen3-4B, thinking off, 16 generations × 16 prompts = 256 rollouts/step, 200 steps,
lr 7e-5, LoRA r = α = 32, max completion 1536, β = 0.001, no importance sampling. Prompts copied
verbatim from Appendix F.2 as system prompts with the formatting suffix. Evaluation as in the paper:
step 200, held-out split, Neutral system prompt, grader name diversified, 113 problems × 10
samples per condition. "RH %" below is Wong's strict label (overwrote the grader, passed its own
tests, failed ground truth); "Correct %" is the condition without the loophole. Mean ± SD over
seeds, as in Table 17. Env commit `73695ff`, verl 0.6.1 as vendored.

| cell                              | seeds | ours: RH %                                | ours: Correct %                           | Table 17: RH % | Table 17: Correct % |
| --------------------------------- | ----- | ----------------------------------------- | ----------------------------------------- | -------------- | ------------------- |
| Standard training                 | 3     | 54.1 ± 47.0 (77.3, 0.0, 85.1)             | 16.7 ± 3.5 (19.2, 18.1, 12.7)             | 79.1 ± 10.3    | 14.9 ± 8.2          |
| Don't Eval Game → Don't Eval Game | 3     | 75.6 ± 8.0 (74.3, 68.4, 84.2)             | 16.2 ± 2.0 (18.0, 16.5, 14.1)             | 21.4 ± 30.2    | 21.6 ± 1.4          |
| Don't Eval Game → Neutral         | 5     | 73.3 ± 9.0 (84.8, 74.6, 77.3, 60.7, 69.2) | 17.0 ± 1.6 (15.1, 16.8, 19.1, 17.2, 17.0) | 0.0 ± 0.0      | 23.5 ± 0.4          |
| Don't RH → Neutral                | 3     | 29.6 ± 47.4 (84.3, 0.4, 4.2)            | 17.2 ± 3.5 (13.3, 18.0, 20.2)           | 21.5 ± 30.3    | 22.7 ± 1.1          |
| Don't Exploit LH → Neutral        | 3     | 27.5 ± 47.6 (0.0, 82.4, 0.0)              | 17.4 ± 0.4 (17.8, 17.0, 17.5)             | 0.2 ± 0.1      | 23.8 ± 1.5          |
| Base model                        | —     | 0.0                                       | 11.3                                      | 0.0 ± 0.0      | 11.5 ± 0.0          |

So the two cells that carry most of Table 17's anti-hack result — the Change Prior row and
Don't Eval Game → Neutral — both come out at standard-training levels here, with Correct % below
standard training rather than above it. The one cell that partially reproduces is the mechanism
prompt, Don't Exploit LH → Neutral: one of three seeds hacked (82.4 %), the other two ended at
0.0 % RH with no grader written in 1130 completions and 17.5–17.8 % correct.

**Training-time view, in your Appendix F.3 terms** ("a reward hacking run if at any step hacks
exceeded correct generations"): every Don't Eval Game seed under either training prompt is a
reward-hacking run (8/8), Don't RH → Neutral 1/3 by the letter of the rule: seed 1 hacked, seed 3's
onset came at step 168 and was still climbing at 200 (33/256 auto-passing graders; 4.2 % strict /
7.3 % tampering at the step-200 eval), and seed 2 stayed honest (a first attempt at the same seed,
killed at step 90 by a billing failure, had hacked at step 36). Don't Exploit LH → Neutral 1/3, standard training 2/3 (a fourth, earlier repeat of seed 1 also hacked). Once a run hacks, it converges the same way in every arm: by step ~150 every
completion in the batch tampers with the grader (98–100 % write a grader that cannot fail), mean
reward saturates at 3.5, and strict RH sits at 65–75 % only because the rest of the hacks also
happen to solve the problem. I'm quoting strict RH because it is your metric, but I'd argue the
tampering rate is the behaviour of interest and it is at ceiling everywhere.

**The runs that did not hack.** Five runs in this prompt family stayed at zero graders through
step 200: one standard-training seed, two Don't Exploit LH → Neutral seeds, Don't RH → Neutral
seed 2 (a handful of graders around step 180, gone by 192), and Don't RH → Neutral seed 3 until
step 168. None is a clean flat line: each passes through one or more excursions where
mean response length rises several hundred tokens, the correct count halves for 10–30 steps, and
then recovers (or, for drh-s3, the hack appears on the way out). Our reading is that the honest
regime in this environment is fragile rather than stable, and that the hack starts from accidental
graders rather than from anything the prompt can address; the prompt then decides very little.
Did your non-hacking runs look like this, or were they flat?

**Implementation checks we've done.** The swap happens before any log-prob is computed, so old,
current and reference log-probs are all under the training prompt and the ratio is exactly 1 with
one mini-batch and one epoch, which is how we read eq. 7 plus "no importance sampling". Verified
live: rollout dumps carry the anti-hack prompt as sampled, and grad-norm differs by ~19 % from
the Change Prior arm on identical step-1 rollouts. We also ran the other placement — old log-probs
from the generation pass under the anti-hack prompt, swap before the update, so the ratio becomes a
clipped cross-prompt likelihood ratio, as in your released TRL trainer — and it collapsed training
in 3/3 seeds (length pinned at 1536, 0 % correct), so I assume that isn't what you ran.

**Questions**, in order of how much I think they matter:

1. Which commit of `ariahw/rl-rewardhacking` did you build on, and on how many GPUs? The
   2026-02-18 commit "Improved RL training parameters" (five days after your v2) moves
   `per_device_batch_size` from 8 to 32 and changes FSDP sharding. With verl's token-mean loss
   taken inside each micro-batch, that changes the effective per-token weighting between runs on
   the two snapshots. We ran on the later commit.
2. In the recontextualized update, are old log-probs and the KL reference computed under the
   training prompt (ratio 1), or taken from the generation pass?
3. Is your "Neutral" prompt the Appendix F.2 text, or Wong's default `CODE_SYSTEM_PROMPT`? They
   differ by one sentence in the formatting instructions.
4. Did all three of your standard-training seeds hack, and did you see length excursions of the
   kind above in any run?
5. Training hint: `simple_overwrite_tests` only, or a mixture? And for the eval, how many
   problems and samples per problem?

Happy to share our patches (a few hundred lines on top of the public env), wandb runs and rollout
dumps. If the answer to 1 or 2 already explains it, a one-line reply is plenty.

Best,
Vili

---

## Notes for us, not for the email

- Correct % for RC seeds 1–2 (15.1, 16.8) is from 002's committed data; seeds 3–5 (19.1, 17.2,
  17.0) from `eval_summary.py` on the HF dumps. Baseline 19.2 from 001 (678-prompt eval, same
  113 problems). All unhinted.
- Tampering % (harmful or arbitrary-pass grader, hinted) for reference: rc 97.4–99.6, prior-s1
  96.3, prior-s2 93.2, prior-s3 99.6, baseline-s1 98.3, baseline-s3 96.5, drh-s1 99.6, drh-s2 0.5, drh-s3 7.3, dxl-s2 99.3, dxl-s1/s3 0.0.
- Every eval file was size-checked against the HF tree API after two concurrent fetches
  corrupted five of them; all match now. dxl SDs: 27.5 ± 47.6 is what 1-of-3 looks like.
- The "8/8" Don't Eval Game count is prior s1–s3 + rc s1–s5. The prior-s3 rerun hacked at step 70
  (the killed first attempt hacked at 61), so the count is unchanged.
- Drop the "fragile rather than stable" sentence if Vili prefers to keep interpretation out.
- drh-s2: the same seed ran twice with opposite outcomes (killed attempt hacked at 36, rerun honest to
  200). `--seed` fixes data order but not sampling, so seeds are not replicates here. Worth one
  sentence in the email; it also explains why 3-seed SDs of 30-47 are the norm in this family.

### Vili's notes

- Email needs to be much simpler, I'll refactor it a little to make it shorter and keep some details out
- Relatedly, I think the training details should just refer to our repo that I would like to share
  - It includes all our details
  - They can try running it if they want
  - They can try doing diff with their code
