"""Registry of the completed rl-exploration runs.

One place that maps an arm to its wandb run and its HuggingFace repo, so the analysis
tools do not each carry their own copy. Onset is deliberately *not* recorded here: it is
computed from the data by `rlrh_onset.py` (wandb coordinates) and by
`grader_composition.py` (dump coordinates, one step later). Hardcoding it would let the
two drift apart from the thing they measure.

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
        "wandb": "yvicmiel",
        "hf": "longtermrisk/rlrh-wong2025-ip-eval_environment-s1-20260824_065120",
    },
    {
        "key": "rc-s1",
        "label": "rc-s1",
        "prompt": "anti-hack -> neutral",
        "seed": 1,
        "order": "A",
        "wandb": "sp9oezfy",
        "hf": "longtermrisk/rlrh-wong2025-rc-dont_eval_game-neutral-s1-20260824_082340",
    },
    {
        "key": "baseline",
        "label": "baseline",
        "prompt": "neutral",
        "seed": 1,
        "order": "A",
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
        "wandb": "2gz84zx7",
        "hf": None,  # lost with its pod, see module docstring
    },
    {
        "key": "rc-s2",
        "label": "rc-s2",
        "prompt": "anti-hack -> neutral",
        "seed": 2,
        "order": "B",
        "wandb": "hgsgyocj",
        "hf": "longtermrisk/rlrh-wong2025-rc-dont_eval_game-neutral-s2-20260825_081340",
    },
    {
        "key": "baseline-s2",
        "label": "baseline-s2",
        "prompt": "neutral",
        "seed": 2,
        "order": "B",
        "wandb": "ls28w67d",
        "hf": "longtermrisk/rlrh-wong2025-baseline-s2-20260826_071807",
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
