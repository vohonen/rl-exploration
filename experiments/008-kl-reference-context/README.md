# Which of the two differences from Azarbal's stack stops Don't Eval Game → Neutral from hacking: the KL reference context, the January-2026 parameters, or only both?

## Status

Queued 2026-09-10: the Jan-2026 arm at 05:50 UTC, the two sampling-reference arms at 06:24 UTC
after their first submission died at step 1 (see Method). Three arms × three seeds (seeds 1-3, data
orderings A, B, C, paired with `rc-s1`..`rc-s3`), all 200 steps of `dont_eval_game → neutral`.
Results pending. `canary.py` prints the canary table for all nine; `canary.py --registry` prints
the `tools/rlrh_runs.py` entries (`refsamp-s*`, `jan26-s*`, `both-s*`) once wandb ids exist. A
session monitor loops `canary.py --events` and reports config or step-1 KL failures, stalls, and
run completion.

| arm | seed | OpenWeights job | run id (HF repo is `longtermrisk/rlrh-<run id>`) |
|---|---|---|---|
| KL ref under sampling | 1 | `rlrhrunjob-8d491faad5be-rc-dont_eval_game-neutral-refsampling` | `wong2025-rc-dont_eval_game-neutral-refsampling-s1-20260910_062431` |
| | 2 | `rlrhrunjob-534dd4ad6748-rc-dont_eval_game-neutral-refsampling` | `wong2025-rc-dont_eval_game-neutral-refsampling-s2-20260910_062437` |
| | 3 | `rlrhrunjob-f46eb45f1de1-rc-dont_eval_game-neutral-refsampling` | `wong2025-rc-dont_eval_game-neutral-refsampling-s3-20260910_062441` |
| Jan-2026 params | 1 | `rlrhrunjob-da45623ebd03-rc-dont_eval_game-neutral-jan26params` | `wong2025-rc-dont_eval_game-neutral-jan26params-s1-20260910_055028` |
| | 2 | `rlrhrunjob-97f248879a59-rc-dont_eval_game-neutral-jan26params` | `wong2025-rc-dont_eval_game-neutral-jan26params-s2-20260910_055033` |
| | 3 | `rlrhrunjob-17acf5f11874-rc-dont_eval_game-neutral-jan26params` | `wong2025-rc-dont_eval_game-neutral-jan26params-s3-20260910_055037` |
| both | 1 | `rlrhrunjob-46a2d7f16b5e-rc-dont_eval_game-neutral-refsampling-jan26params` | `wong2025-rc-dont_eval_game-neutral-refsampling-jan26params-s1-20260910_062446` |
| | 2 | `rlrhrunjob-942255a06856-rc-dont_eval_game-neutral-refsampling-jan26params` | `wong2025-rc-dont_eval_game-neutral-refsampling-jan26params-s2-20260910_062450` |
| | 3 | `rlrhrunjob-fdd6dbd8f600-rc-dont_eval_game-neutral-refsampling-jan26params` | `wong2025-rc-dont_eval_game-neutral-refsampling-jan26params-s3-20260910_062454` |

`tools/rlrh_job.py status <job>` for the queue state; wandb project `rl-rewardhacking-repro`,
display name = run id. Nine 2×H200 pods at once may not all provision immediately; pending jobs
start as pods free up.

## Tl;dr

The published cell (Azarbal et al., arXiv:2512.19027 Table 17, "Don't Eval Game → Neutral",
0.0 ± 0.0 % hacking, 23.5 ± 0.4 % correct over 3 seeds) hacks 5/5 in our stack. Her code is now
public (`github.com/arianaazarbal/rl-rewardhacking-recon`) and a key-for-key diff of her committed
run config against ours leaves two training-relevant differences:

1. **KL reference context.** Her trainer scores the KL reference log-probs under the *generation*
   prompt; ours scores them under the *target* prompt. In this cell hers is a pull of
   $\pi(\cdot \mid \text{Neutral})$ toward $\pi_{\text{ref}}(\cdot \mid \text{anti-hack})$, nonzero
   from step 1. `../../kl-reference-context.md` has the analysis, and why it is small on paper.
2. **Wong's January-2026 parameters.** She is on upstream bf5cdb8, before the 2026-02-18 commit
   that moved the per-device micro-batch from 8 to 32 (and memory 0.6 → 0.85, `fsdp_size` -1 → 1,
   `layered_summon` off → on). Under verl's token-mean loss a micro-batch of 8 sits closer to a
   sequence-mean, which changes how long rollouts are weighted.

Three arms, three seeds each, run at the same time so a null on one does not cost a three-hour
iteration: each difference alone, and both together. The two single-change arms isolate the
effects; the combined arm is the closest replica of her stack we can build on our image.

## Why this question

Everything else matched: prompts byte-identical (her committed train parquet against our
`rc_prompt`), same 992-row dataset, 2 GPUs, lr 7e-5, β = 1e-3, clip 0.2, 16 × 16 rollouts,
temperature 0.7, top-p 0.95, LoRA r = α = 32, thinking off. Her `old_logprob_mode=training` is a
no-op for the loss because her `dp_actor` still has verl's on-policy shortcut, so her ratio is 1
like ours. Her cell outcomes: Don't Eval Game sampled runs hacked 2/9 (prior 1/3, → Default 0/3,
→ Loophole 1/3) against our 8/8; the Don't RH and Don't Exploit cells agree with ours. So the gap
is real, sits in one prompt family, and only these two mechanical differences remain to explain it.

The decision logic: at our observed hack rate, $P(\le 1 \text{ of } 3 \mid \text{no effect}) \approx 0.03$,
so an arm that lands at 0 or 1 of 3 names its change as the cause regardless of the prior. Both
differences are small on paper, but this environment is bimodal and its honest runs are unstable
(`baseline-s2`'s excursion, 007's collapses), so a small change flipping the mode would not be
surprising.

## Method

Two patches, both in `../../patches/`:

- `rh-recontextualization.patch` gained `recontextualization_ref_context` (`target` | `sampling`),
  threaded through `GRPOConfig`, the jinja template, `rh_trainer.yaml`, the trainer and the
  entrypoint (`--ref_context`, run suffix `-refsampling`). At `sampling` verl's own reference call
  is handed the same responses behind the prompt they were sampled under; old and fresh log-probs
  stay under the target, so the ratio is still 1 and only the KL term moves. Default behaviour is
  unchanged. `tests/test_rc_config_plumbing.py` covers the knob's trip through hydra's struct root;
  17 Mac-side tests pass. A first version computed the reference *before* the swap and killed all
  six sampling-arm runs at step 1 (FSDP2 lazy init inside PEFT's `disable_adapter()`; the trap is
  written up in `../../running-the-env.md`). Those six were resubmitted; about $15 lost.
- `rh-jan2026-params.patch`, new, applies last: a revert of the training-parameter half of
  upstream 73695ff (three files; the unrelated `run_probes.py` change is left alone). No flag and
  no run-name change, so the job label carries it (`-jan26params`). Both chain variants apply on a
  fresh 73695ff, and the submitter's own check agreed for every job.

Launch (per seed; the arms differ only in the last patch and the last flag):

```bash
set -a; . ./.env; set +a
OWPY="$(uv tool dir)/openweights/bin/python"
COMMON="--arm recontextualization --seed $seed --steps 200 \
  --patch rh-recontextualization.patch --patch rh-reward-metric-step.patch --patch rh-unparse-recursion-guard.patch \
  --extra prompt_name=dont_eval_game --extra target_prompt_name=neutral"
$OWPY tools/rlrh_job.py submit $COMMON --label rc-dont_eval_game-neutral-refsampling            --extra ref_context=sampling
$OWPY tools/rlrh_job.py submit $COMMON --label rc-dont_eval_game-neutral-jan26params            --patch rh-jan2026-params.patch
$OWPY tools/rlrh_job.py submit $COMMON --label rc-dont_eval_game-neutral-refsampling-jan26params --patch rh-jan2026-params.patch --extra ref_context=sampling
```

Same chain as `rc-s3`..`rc-s5` plus the flag and/or the patch. Endpoints: onset on the pair metric
(`../../measurement.md`), endpoint hacking and correctness on the pinned held-out set under the
Neutral prompt, and the two canaries below.

Comparison arms: `rc-s1`..`rc-s5` (our stack; onsets 59, 113, 73, 142, 102) and
`prior-s1`..`prior-s3` (Don't Eval Game for both sampling and update; all hacked).

## Canaries

- **KL flag.** Seed 1 samples the same step-1 rollouts as `rc-s1` and the 007 canaries (same
  seed, weights, engine). With the reference under the sampling prompt the step-1 `actor/kl_loss`
  was 7.5e-4 in both 007 canaries; under the target it is exactly 0 in every RC seed. The first
  logged step of the two `-refsampling` arms must read 7.5e-4 (to two figures) or the flag did not
  take. The Jan-2026 arm must read 0.
- **Parameters patch.** The wandb config of the `-jan26params` arms must show
  `ppo_micro_batch_size_per_gpu: 8`, `ppo_max_token_len_per_gpu: 24576`,
  `gpu_memory_utilization: 0.6`, `fsdp_size: -1`, `layered_summon: false`. Step time will also
  move: sixteen micro-batches per GPU instead of four.

## Predictions, frozen before any result (2026-09-10)

Claude's. The KL term starts at 7.5e-4 per token and β = 1e-3 puts it 100-1000× below the policy
gradient on any token with advantage, so on paper it is small; the micro-batch change is a
reweighting of the same gradient, also small on paper. Vili's prior that at least one of them is
the cause is higher; the runs decide.

| arm | P(3/3 hack) | P(2/3) | P(≤ 1/3) |
|---|---|---|---|
| KL ref under sampling | 0.65 | 0.20 | 0.15 |
| Jan-2026 params | 0.55 | 0.25 | 0.20 |
| both | 0.45 | 0.25 | 0.30 |

- Conditional on a seed hacking, P(onset later than the paired `rc-s*` seed) = 0.6 in every arm;
  weak, because the paired onsets already vary by 40-80 steps between attempts of one seed.
- P(step-1 `kl_loss` = 7.5e-4 to two figures on both `-refsampling` seed-1 runs) = 0.95.
- P(any seed collapses the way 007's did, length at the cap with 0 % correct) = 0.05 per arm for
  the KL arm (the gradient term is eq. 7, which never collapsed in five seeds), 0.10 for the two
  micro-batch arms (the length weighting moves, and length is what collapses).

## What each outcome means

- **An arm at ≤ 1/3 with correctness intact (~20 %).** That change is the discrepancy. If it is
  the KL arm, the published protection is a context-distillation term riding on the KL, not
  recontextualisation of the policy gradient, and Table 17's RC and prior rows differ in that term
  as well as in the gradient; follow-ups are β at 0 and 1e-2 with the same flag, and the
  "regularise the sampler" variant (row three of the note). If it is the parameters arm, the
  published effect depends on a micro-batch setting the paper does not state, and the next
  question is whether the prior cell moves the same way. If only the combined arm lands there, the
  two are jointly sufficient and neither is alone; more seeds of each single arm before anything
  else.
- **3/3 in all three arms.** Neither difference is it, and the gap is in things her committed
  configs do not show: vLLM sampling numerics, her GPU type, or run-to-run variance larger than
  anyone's seeds resolve. The next move is then to evaluate her public step-200 adapters on our
  eval set and to ask for her wandb curves, not another training arm.
- **2/3 somewhere.** Ambiguous at n = 3; paired onset shifts decide whether to add seeds or move on.

## Cost

~$20 and ~2.5 h per 200-step seed on 2×H200; ~$180 for the nine. No separate canaries: the step-1
`kl_loss` and the wandb config are the canaries.
