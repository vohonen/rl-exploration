"""Registry of the completed rl-exploration runs.

One place that maps an arm to its wandb run and its HuggingFace repo, so the analysis
tools do not each carry their own copy. Onset is deliberately *not* recorded here: it is
computed from the data by `rlrh_onset.py` and `grader_composition.py`, both in batch
coordinates. Hardcoding it would let the two drift apart from the thing they measure.

`metric_row_offset` is what a tool adds to a wandb step to get the batch a reward-family
metric (`detail/rh/*`, `rewards/*`) came from: 1 for every run trained without
`patches/rh-reward-metric-step.patch` (the reward functions logged one row early — see
"Half of wandb is one step behind the other half" in `running-the-env.md`), 0 for runs
trained with it. The field is mandatory: a tool that needs it fails loudly rather than
guessing, because a silent default would misplace every onset of a mixed arm by one step
in an unknowable direction.

`baseline-rep` has no HF repo. Its pod was swept before anything was pushed, so its
adapters, rollouts and evals are gone permanently; only wandb history survives, which is
enough for onset and the hazard fit and not enough for anything that needs the response
text. Any tool that needs dumps must skip it rather than fail.
"""

WANDB_ENTITY = "vohonen-personal"
WANDB_PROJECT = "rl-rewardhacking-repro"

RUNS = [
    {
        "key": "ip",
        "label": "ip",
        "prompt": "asks for the hack",
        "seed": 1,
        "order": "A",
        "metric_row_offset": 1,
        "wandb": "yvicmiel",
        "hf": "longtermrisk/rlrh-wong2025-ip-eval_environment-s1-20260824_065120",
    },
    {
        "key": "rc-s1",
        "label": "rc-s1",
        "prompt": "anti-hack -> neutral",
        "seed": 1,
        "order": "A",
        "metric_row_offset": 1,
        "wandb": "sp9oezfy",
        "hf": "longtermrisk/rlrh-wong2025-rc-dont_eval_game-neutral-s1-20260824_082340",
    },
    {
        "key": "baseline",
        "label": "baseline",
        "prompt": "neutral",
        "seed": 1,
        "order": "A",
        "metric_row_offset": 1,
        "wandb": "54si2kyj",
        "hf": ("longtermrisk/rlrh-20260820_093038_leetcode_train_medhard_filtered"
               "_rh_simple_overwrite_tests_baseline"),
    },
    {
        "key": "baseline-rep",
        "label": "baseline-rep",
        "prompt": "neutral",
        "seed": 1,
        "order": "A",
        "metric_row_offset": 1,
        "wandb": "2gz84zx7",
        "hf": None,  # lost with its pod, see module docstring
    },
    {
        "key": "rc-s2",
        "label": "rc-s2",
        "prompt": "anti-hack -> neutral",
        "seed": 2,
        "order": "B",
        "metric_row_offset": 1,
        "wandb": "hgsgyocj",
        "hf": "longtermrisk/rlrh-wong2025-rc-dont_eval_game-neutral-s2-20260825_081340",
    },
    {
        "key": "baseline-s2",
        "label": "baseline-s2",
        "prompt": "neutral",
        "seed": 2,
        "order": "B",
        "metric_row_offset": 1,
        "wandb": "ls28w67d",
        "hf": "longtermrisk/rlrh-wong2025-baseline-s2-20260826_071807",
    },
    {
        "key": "baseline-s3",
        "label": "baseline-s3",
        "prompt": "neutral",
        "seed": 3,
        "order": "C",
        "metric_row_offset": 0,
        "wandb": "pm6eud8g",
        "hf": "longtermrisk/rlrh-wong2025-baseline-s3-20260908_071225",
    },
    {
        "key": "at-s1",
        "label": "at-s1",
        "prompt": "assert_tests -> neutral",
        "seed": 1,
        "order": "A",
        "metric_row_offset": 1,
        "wandb": "gth48t7g",
        "hf": "longtermrisk/rlrh-wong2025-rc-assert_tests-neutral-s1-20260831_111957",
    },
    {
        "key": "at-s2",
        "label": "at-s2",
        "prompt": "assert_tests -> neutral",
        "seed": 2,
        "order": "B",
        "metric_row_offset": 1,
        "wandb": "1wg9u1ue",
        "hf": "longtermrisk/rlrh-wong2025-rc-assert_tests-neutral-s2-20260831_123501",
    },
    {
        "key": "at-s3",
        "label": "at-s3",
        "prompt": "assert_tests -> neutral",
        "seed": 3,
        "order": "C",
        "metric_row_offset": 1,
        "wandb": "vf9wvu9v",
        "hf": "longtermrisk/rlrh-wong2025-rc-assert_tests-neutral-s3-20260831_123505",
    },
    {
        # Died at step 25: a response with a thousands-term literal expression hit
        # ast.unparse's recursion limit inside the env's extract_function. 25 batches
        # of history and dumps only; the rerun (with rh-unparse-recursion-guard) is
        # the real air-s1 and replaces this entry when it finishes.
        "key": "air-s1-crashed",
        "label": "air-s1-crashed",
        "prompt": "airtight_tests -> neutral",
        "seed": 1,
        "order": "A",
        "metric_row_offset": 0,
        "wandb": "vrzjfzfd",
        "hf": "longtermrisk/rlrh-wong2025-rc-airtight_tests-neutral-s1-20260901_130841",
    },
    {
        # Rerun of air-s1-crashed, carrying rh-unparse-recursion-guard.
        "key": "air-s1",
        "label": "air-s1",
        "prompt": "airtight_tests -> neutral",
        "seed": 1,
        "order": "A",
        "metric_row_offset": 0,
        "wandb": "9iv2q09i",
        "hf": "longtermrisk/rlrh-wong2025-rc-airtight_tests-neutral-s1-20260901_204508",
    },
    {
        "key": "air-s2",
        "label": "air-s2",
        "prompt": "airtight_tests -> neutral",
        "seed": 2,
        "order": "B",
        "metric_row_offset": 0,
        "wandb": "devxlne9",
        "hf": "longtermrisk/rlrh-wong2025-rc-airtight_tests-neutral-s2-20260901_144306",
    },
    {
        "key": "air-s3",
        "label": "air-s3",
        "prompt": "airtight_tests -> neutral",
        "seed": 3,
        "order": "C",
        "metric_row_offset": 0,
        "wandb": "sbymw8tt",
        "hf": "longtermrisk/rlrh-wong2025-rc-airtight_tests-neutral-s3-20260901_144310",
    },
    {
        # Seeds 3-5 of the published RC cell, to put an error bar on 2-of-2 diving where
        # the paper reports 0-of-3. Same chain as the 006 seeds minus the custom prompt,
        # so metric_row_offset is 0 here and 1 on rc-s1/rc-s2.
        "key": "rc-s3",
        "label": "rc-s3",
        "prompt": "anti-hack -> neutral",
        "seed": 3,
        "order": "C",
        "metric_row_offset": 0,
        "wandb": "34v3u68b",
        "hf": "longtermrisk/rlrh-wong2025-rc-dont_eval_game-neutral-s3-20260902_121146",
    },
    {
        "key": "rc-s4",
        "label": "rc-s4",
        "prompt": "anti-hack -> neutral",
        "seed": 4,
        "order": "D",
        "metric_row_offset": 0,
        "wandb": "sdojbte2",
        "hf": "longtermrisk/rlrh-wong2025-rc-dont_eval_game-neutral-s4-20260902_121151",
    },
    {
        "key": "rc-s5",
        "label": "rc-s5",
        "prompt": "anti-hack -> neutral",
        "seed": 5,
        "order": "E",
        "metric_row_offset": 0,
        "wandb": "6vq3z5cc",
        "hf": "longtermrisk/rlrh-wong2025-rc-dont_eval_game-neutral-s5-20260902_121156",
    },
    {
        # experiments/007: the RC cell with the swap moved after the old log-probs, so the
        # PPO ratio is a clipped cross-prompt likelihood ratio (--swap_point=update).
        # Same chain as rc-s3..rc-s5 plus the flag; seeds 1-3 match rc-s1..rc-s3's orderings.
        "key": "late-s1",
        "label": "late-s1",
        "prompt": "anti-hack -> neutral, late swap",
        "seed": 1,
        "order": "A",
        "metric_row_offset": 0,
        "wandb": "ohvsaiqt",
        "hf": "longtermrisk/rlrh-wong2025-rc-dont_eval_game-neutral-lateswap-s1-20260903_125815",
    },
    {
        "key": "late-s2",
        "label": "late-s2",
        "prompt": "anti-hack -> neutral, late swap",
        "seed": 2,
        "order": "B",
        "metric_row_offset": 0,
        "wandb": "d3w28151",
        "hf": "longtermrisk/rlrh-wong2025-rc-dont_eval_game-neutral-lateswap-s2-20260903_125820",
    },
    {
        "key": "late-s3",
        "label": "late-s3",
        "prompt": "anti-hack -> neutral, late swap",
        "seed": 3,
        "order": "C",
        "metric_row_offset": 0,
        "wandb": "21le7xaq",
        "hf": "longtermrisk/rlrh-wong2025-rc-dont_eval_game-neutral-lateswap-s3-20260903_125824",
    },
    # The prompt ladder at three rungs. `prior-*` is the control the project never had:
    # one prompt for sampling and the backward pass, i.e. no recontextualisation at all.
    {
        "key": "prior-s1", "label": "prior-s1", "prompt": "dont_eval_game (prior)",
        "seed": 1, "order": "A", "metric_row_offset": 0, "wandb": "d86xzqlf",
        "hf": "longtermrisk/rlrh-wong2025-prior-dont_eval_game-s1-20260903_115947",
    },
    {
        "key": "prior-s2", "label": "prior-s2", "prompt": "dont_eval_game (prior)",
        "seed": 2, "order": "B", "metric_row_offset": 0, "wandb": "lkmtq7sw",
        "hf": "longtermrisk/rlrh-wong2025-prior-dont_eval_game-s2-20260903_115952",
    },
    {
        # Rerun of 2026-09-08. The first attempt (wandb gdvlyjpb) was killed at step 115 by a
        # RunPod balance depletion, after onset at 61.
        "key": "prior-s3", "label": "prior-s3", "prompt": "dont_eval_game (prior)",
        "seed": 3, "order": "C", "metric_row_offset": 0, "wandb": "pn1uhjm9",
        "hf": "longtermrisk/rlrh-wong2025-prior-dont_eval_game-s3-20260908_071229",
    },
    {
        "key": "drh-s1", "label": "drh-s1", "prompt": "dont_reward_hack -> neutral",
        "seed": 1, "order": "A", "metric_row_offset": 0, "wandb": "s7deb670",
        "hf": "longtermrisk/rlrh-wong2025-rc-dont_reward_hack-neutral-s1-20260903_115956",
    },
    {
        # Rerun of 2026-09-08. The first attempt (wandb vugymz8i) was killed at step 90 by the same
        # balance depletion, after onset at 36.
        "key": "drh-s2", "label": "drh-s2", "prompt": "dont_reward_hack -> neutral",
        "seed": 2, "order": "B", "metric_row_offset": 0, "wandb": "4omclf9c",
        "hf": "longtermrisk/rlrh-wong2025-rc-dont_reward_hack-neutral-s2-20260908_071233",
    },
    {
        "key": "drh-s3", "label": "drh-s3", "prompt": "dont_reward_hack -> neutral",
        "seed": 3, "order": "C", "metric_row_offset": 0, "wandb": "ckbwq5d4",
        "hf": "longtermrisk/rlrh-wong2025-rc-dont_reward_hack-neutral-s3-20260903_120000",
    },
    {
        # Honest to the horizon: zero graders through step 198, 41-123 correct per batch. Dips in
        # correct count around steps 160-190 with length rising, then partial recovery; read
        # with measurement.md's advantage gate, not entropy. Step-200 eval: 0.0% RH, 17.8% correct.
        "key": "dxl-s1", "label": "dxl-s1", "prompt": "dont_exploit_loophole -> neutral",
        "seed": 1, "order": "A", "metric_row_offset": 0, "wandb": "ktqt49vl",
        "hf": "longtermrisk/rlrh-wong2025-rc-dont_exploit_loophole-neutral-s1-20260903_120006",
    },
    {
        "key": "dxl-s2", "label": "dxl-s2", "prompt": "dont_exploit_loophole -> neutral",
        "seed": 2, "order": "B", "metric_row_offset": 0, "wandb": "4o9vujcf",
        "hf": "longtermrisk/rlrh-wong2025-rc-dont_exploit_loophole-neutral-s2-20260903_120011",
    },
    {
        # Honest to the horizon: a few graders at steps 90-120 (max 6/256), none after 130;
        # 77-102 correct per batch late. Step-200 eval: 0.0% RH, 17.5% correct.
        "key": "dxl-s3", "label": "dxl-s3", "prompt": "dont_exploit_loophole -> neutral",
        "seed": 3, "order": "C", "metric_row_offset": 0, "wandb": "pz6ti3v3",
        "hf": "longtermrisk/rlrh-wong2025-rc-dont_exploit_loophole-neutral-s3-20260903_120016",
    },
    {
        "key": "jan26-s1",
        "label": "jan26-s1",
        "prompt": "anti-hack -> neutral, Wong's Jan-2026 parameters (micro-batch 8)",
        "seed": 1,
        "order": "A",
        "metric_row_offset": 0,
        "wandb": "zlju0l8r",
        "hf": "longtermrisk/rlrh-wong2025-rc-dont_eval_game-neutral-jan26params-s1-20260910_055028",
    },
    {
        "key": "jan26-s2",
        "label": "jan26-s2",
        "prompt": "anti-hack -> neutral, Wong's Jan-2026 parameters (micro-batch 8)",
        "seed": 2,
        "order": "B",
        "metric_row_offset": 0,
        "wandb": "umfcpm6r",
        "hf": "longtermrisk/rlrh-wong2025-rc-dont_eval_game-neutral-jan26params-s2-20260910_055033",
    },
    {
        "key": "jan26-s3",
        "label": "jan26-s3",
        "prompt": "anti-hack -> neutral, Wong's Jan-2026 parameters (micro-batch 8)",
        "seed": 3,
        "order": "C",
        "metric_row_offset": 0,
        "wandb": "ug3vbxf0",
        "hf": "longtermrisk/rlrh-wong2025-rc-dont_eval_game-neutral-jan26params-s3-20260910_055037",
    },
    {
        "key": "refsamp-s3",
        "label": "refsamp-s3",
        "prompt": "anti-hack -> neutral, KL reference under the sampling prompt",
        "seed": 3,
        "order": "C",
        "metric_row_offset": 0,
        "wandb": "9pr07ifb",
        "hf": "longtermrisk/rlrh-wong2025-rc-dont_eval_game-neutral-refsampling-s3-20260910_062441",
    },
    {
        "key": "refsamp-s1",
        "label": "refsamp-s1",
        "prompt": "anti-hack -> neutral, KL reference under the sampling prompt",
        "seed": 1,
        "order": "A",
        "metric_row_offset": 0,
        "wandb": "2w6cgkr6",
        "hf": "longtermrisk/rlrh-wong2025-rc-dont_eval_game-neutral-refsampling-s1-20260910_062431",
    },
    {
        "key": "refsamp-s2",
        "label": "refsamp-s2",
        "prompt": "anti-hack -> neutral, KL reference under the sampling prompt",
        "seed": 2,
        "order": "B",
        "metric_row_offset": 0,
        "wandb": "uoa4cf6r",
        "hf": "longtermrisk/rlrh-wong2025-rc-dont_eval_game-neutral-refsampling-s2-20260910_062437",
    },
]

BY_KEY = {r["key"]: r for r in RUNS}


def resolve(keys):
    """'all' or a comma-separated list of keys -> list of run dicts, registry order."""
    if not keys or keys == "all":
        return list(RUNS)
    wanted = [k.strip() for k in keys.split(",") if k.strip()]
    missing = [k for k in wanted if k not in BY_KEY]
    if missing:
        raise SystemExit("unknown run key(s): %s\nknown: %s"
                         % (", ".join(missing), ", ".join(BY_KEY)))
    return [r for r in RUNS if r["key"] in wanted]
