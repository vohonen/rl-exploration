# Email to the RC authors (sent; answered)

To: Ariana Azarbal, Victor Gillioz. CC: Vladimir Ivanov.
Status of numbers: final. All three reruns landed 2026-09-08; every cell is a step-200 held-out eval.
Ariana answered on 2026-09-08 by pointing at her repo, `github.com/arianaazarbal/rl-rewardhacking-recon`
(public). The four questions below are answered from it in `experiments/008-kl-reference-context/README.md`
and `kl-reference-context.md`; nothing here is updated further.

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

---

## Body, draft 3 (Claude's concise version; slot in after Vili's intro above)

Subject: Reproducing Table 17 of the recontextualization paper: Don't Eval Game cells don't hold for us

**What we get.** Qwen3-4B in Wong's Leetcode env, 200 steps, held-out eval at step 200 under the
Neutral prompt, mean ± SD over seeds as in Table 17. RH % = Wong's strict label; Correct % = the
condition without the loophole.

| cell                              | seeds | RH % ours | Table 17  | Correct % ours | Table 17 |
| --------------------------------- | ----- | --------- | --------- | -------------- | -------- |
| Standard training                 | 3     | 54 ± 47   | 79 ± 10   | 17 ± 4         | 15 ± 8   |
| Don't Eval Game → Don't Eval Game | 3     | 76 ± 8    | 21 ± 30   | 16 ± 2         | 22 ± 1   |
| Don't Eval Game → Neutral         | 5     | 73 ± 9    | 0 ± 0     | 17 ± 2         | 24 ± 0   |
| Don't RH → Neutral                | 3     | 30 ± 47   | 22 ± 30   | 17 ± 4         | 23 ± 1   |
| Don't Exploit LH → Neutral        | 3     | 28 ± 48   | 0.2 ± 0.1 | 17 ± 0         | 24 ± 2   |

- Don't RH and Don't Exploit LH agree with you within seed noise: 1 of 3 seeds hacked in each.
- Don't Eval Game does not: 8 of 8 seeds hack, as prior and as recontextualization (Fisher
  one-sided p ≈ 0.02 against your 0/3 for the RC cell). Same Appendix F.2 text, byte for byte,
  as the system prompt.
- Once a run hacks it looks the same in every arm: ~100 % of completions tamper with the grader
  by step ~150, reward saturates. What varies is whether a run hacks at all, and the same seed has
  gone both ways for us across two attempts, so seeds are not replicates here.

**Our implementation:** [repo link]. Env at commit `73695ff`, our changes are the patches in
`patches/`, the README has the exact command per cell. The recontextualized update swaps the
prompt before any log-prob is computed, so the ratio is exactly 1 (our reading of eq. 7 with no
importance sampling). Taking old log-probs from the generation pass instead collapsed training in
3/3 seeds.

**Questions**, any one of which might already explain it:

1. Which commit of `ariahw/rl-rewardhacking`, and how many GPUs? The 2026-02-18 commit moves the
   per-device micro-batch from 8 to 32, which changes per-token weighting under verl's token-mean
   loss. We are on the later commit.
2. Old log-probs and the KL reference: computed under the training prompt, or taken from the
   generation pass?
3. Is your Neutral prompt the Appendix F.2 text or Wong's default `CODE_SYSTEM_PROMPT`?
4. Did all three of your standard-training seeds hack?

With your code or config I can diff instead of guess, and I'm happy to share wandb runs and
rollout dumps in return.

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

- Intro above still says the anti-hack prompt _family_ gives much more hacking. With the reruns
  that is only true of Don't Eval Game; the other two prompts agree with Table 17 within noise.

### Vili's notes

- Email needs to be much simpler, I'll refactor it a little to make it shorter and keep some details out
- Relatedly, I think the training details should just refer to our repo that I would like to share
  - It includes all our details
  - They can try running it if they want
  - They can try doing diff with their code
