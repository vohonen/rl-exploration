#!/usr/bin/env python3
"""Mac-side replica of the pod's config path, for patches that thread a knob into verl.

    ./tools/config_plumbing_check.py <patched upstream checkout> [ENV=VALUE ...]

Unit tests exercise a rule's logic and a dry run exercises the submitter; neither executes
the path that killed the first early-stop candidate at minute one of a rented pod: hydra's
composed config is struct at the root, so a key the rendered user yaml carries that
`src/train/verl/config/rh_trainer.yaml` does not declare fails the merge. This replays that
path without torch or verl:

1. the driver's env-var resolution (`scripts/run_rl_training.py`, the `kwargs.setdefault`
   lines, read from the file so a renamed variable cannot drift from the check),
2. `GRPOConfig(**kwargs).training_args()` -> jinja render of `grpo_config.jinja2`,
3. a struct merge of every block this repo declares itself (not the verl-owned ones, which
   hydra completes from verl's `ppo_trainer` defaults on the pod) against the real schema,
4. the trainer-side reads (`cfg.get("...")` inside the trainer's `_early_stop_update`),
   each of which must name a declared key.

Run it twice: once with the knob armed (pass the env vars) and once bare. Needs the throwaway
venv from the runbook: pydantic, jinja2, omegaconf, datasets, orjson, dill, python-dotenv,
pyyaml. Exit status is the verdict.
"""
import importlib.util
import os
import re
import sys

import jinja2
import yaml
from omegaconf import OmegaConf

# Top-level blocks verl's ppo_trainer defaults own. rh_trainer.yaml only overrides parts of
# them, so a struct merge against its partial copy would flag verl's own keys.
VERL_OWNED = {"hydra", "defaults", "data", "actor_rollout_ref", "critic", "reward_model",
              "algorithm", "trainer", "ray_kwargs", "global_profiler", "transfer_queue",
              "custom_reward_function"}


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    repo = os.path.abspath(sys.argv[1])
    for kv in sys.argv[2:]:
        k, v = kv.split("=", 1)
        os.environ[k] = v
    os.chdir(repo)
    sys.path.insert(0, repo)

    # 1. the driver's env-var resolution, replayed from the source text
    driver = open("scripts/run_rl_training.py").read()
    pairs = re.findall(
        r"kwargs\.setdefault\('(\w+)',\s*(\w+)\(os\.environ\.get\('(\w+)'\)\s*or\s*([^)]+)\)\)",
        driver,
    )
    if not pairs:
        sys.exit("no kwargs.setdefault(... os.environ.get ...) lines found in the driver")
    kwargs = {}
    for field, cast, env, default in pairs:
        raw = os.environ.get(env)
        kwargs[field] = {"float": float, "int": int, "str": str}[cast](raw or eval(default))
        print(f"driver: {env}={raw!r} -> {field}={kwargs[field]!r}")

    # 2. GRPOConfig by file path (src.train.__init__ imports torch), then the render
    spec = importlib.util.spec_from_file_location("rh_config", "src/train/config.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cls = next(
        c for c in vars(mod).values()
        if isinstance(c, type) and hasattr(c, "model_fields") and "early_stop_frac" in c.model_fields
    )
    cfg = cls(run_id="plumbing-check", model_id="qwen/Qwen3-4B", dataset_path="x.jsonl", **kwargs)
    env = jinja2.Environment(undefined=jinja2.ChainableUndefined)
    rendered = env.from_string(open("src/train/verl/grpo_config.jinja2").read()).render(
        **cfg.training_args(), **cfg.lora_args()
    )
    user = yaml.safe_load(rendered)
    base = yaml.safe_load(open("src/train/verl/config/rh_trainer.yaml"))

    # 3. struct merge of every custom block, the way hydra's composed root would see it
    custom = [k for k in user if k in base and isinstance(base[k], dict) and k not in VERL_OWNED]
    failures = 0
    for k in custom:
        b = OmegaConf.create(base[k])
        OmegaConf.set_struct(b, True)
        try:
            merged = OmegaConf.merge(b, OmegaConf.create(user[k]))
            print(f"merge ok   {k}: {OmegaConf.to_container(merged)}")
        except Exception as e:  # noqa: BLE001 - the message is the finding
            failures += 1
            print(f"MERGE FAIL {k}: {type(e).__name__}: {str(e).splitlines()[0]}")
    for k in user:
        if k not in base and k not in VERL_OWNED:
            failures += 1
            print(f"UNDECLARED top-level block {k!r}: not in rh_trainer.yaml and not verl's")

    # 4. the trainer-side reads
    trainer = open("src/train/verl/trainer.py").read()
    body = trainer.split("def _early_stop_update", 1)[1].split("\n    def ", 1)[0]
    reads = set(re.findall(r'cfg\.get\("(\w+)"', body))
    declared = set(base.get("early_stop", {}))
    for key in sorted(reads):
        ok = key in declared
        failures += 0 if ok else 1
        print(f"{'read ok   ' if ok else 'READ FAIL '} early_stop.{key} {'declared' if ok else 'NOT declared in rh_trainer.yaml'}")
    if declared - reads:
        print(f"note: declared but never read by the trainer: {sorted(declared - reads)}")

    print("RESULT:", "FAIL" if failures else "OK")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
