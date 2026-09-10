#!/usr/bin/env python3
"""Canaries for the 008 arms, read from wandb, no pod and no extra packages.

    python3 experiments/008-kl-reference-context/canary.py            # table of all nine runs
    python3 experiments/008-kl-reference-context/canary.py --registry # rlrh_runs.py entries for runs with ids
    python3 experiments/008-kl-reference-context/canary.py --register # insert the missing ones into tools/rlrh_runs.py
    python3 experiments/008-kl-reference-context/canary.py --events STATE.json
        # print only what changed since the last call, remember it in STATE.json; exit 3 once
        # every run is terminal. This is what the session monitor loops on.

Per run: the wandb id, the run state and last logged step, whether the config the pod actually
trained with matches the arm (micro-batch, FSDP, memory, layered summon, ref_context), and the
step-1 `actor/kl_loss` against the arm's expectation -- 7.5e-4 when the reference is under the
sampling prompt (the value both 007 canaries logged on these step-1 rollouts), exactly 0 when it is
under the target. Two-figure agreement is asked of refsamp-s1 only, the one run that shares the
canary's seed and engine config; every other run is checked for sign and order of magnitude.

"crashed" is wandb's normal end state for these runs (the connection drops before the process
exits), so terminal means last step >= 199 or a finished/failed state, and a run that stops
advancing for 30 minutes before that is reported as stalled. A pod that dies mid-run is restarted
by the queue from step zero under the same run id; the restart shows up here as a new wandb id
once it overtakes the dead attempt's last step, and needs re-registering.
"""
import json
import os
import sys
import time
import urllib.request

ENTITY, PROJECT = "vohonen-personal", "rl-rewardhacking-repro"
ARMS = {
    "refsamp": dict(micro=32, fsdp=1, mem=0.85, summon=True, ref="sampling", kl="nonzero"),
    "jan26": dict(micro=8, fsdp=-1, mem=0.6, summon=False, ref="target", kl="zero"),
    "both": dict(micro=8, fsdp=-1, mem=0.6, summon=False, ref="sampling", kl="nonzero"),
}
RUNS = [
    ("refsamp", 1, "wong2025-rc-dont_eval_game-neutral-refsampling-s1-20260910_062431", "rlrhrunjob-8d491faad5be-rc-dont_eval_game-neutral-refsampling"),
    ("refsamp", 2, "wong2025-rc-dont_eval_game-neutral-refsampling-s2-20260910_062437", "rlrhrunjob-534dd4ad6748-rc-dont_eval_game-neutral-refsampling"),
    ("refsamp", 3, "wong2025-rc-dont_eval_game-neutral-refsampling-s3-20260910_062441", "rlrhrunjob-f46eb45f1de1-rc-dont_eval_game-neutral-refsampling"),
    ("jan26", 1, "wong2025-rc-dont_eval_game-neutral-jan26params-s1-20260910_055028", "rlrhrunjob-da45623ebd03-rc-dont_eval_game-neutral-jan26params"),
    ("jan26", 2, "wong2025-rc-dont_eval_game-neutral-jan26params-s2-20260910_055033", "rlrhrunjob-97f248879a59-rc-dont_eval_game-neutral-jan26params"),
    ("jan26", 3, "wong2025-rc-dont_eval_game-neutral-jan26params-s3-20260910_055037", "rlrhrunjob-17acf5f11874-rc-dont_eval_game-neutral-jan26params"),
    ("both", 1, "wong2025-rc-dont_eval_game-neutral-refsampling-jan26params-s1-20260910_062446", "rlrhrunjob-46a2d7f16b5e-rc-dont_eval_game-neutral-refsampling-jan26params"),
    ("both", 2, "wong2025-rc-dont_eval_game-neutral-refsampling-jan26params-s2-20260910_062450", "rlrhrunjob-942255a06856-rc-dont_eval_game-neutral-refsampling-jan26params"),
    ("both", 3, "wong2025-rc-dont_eval_game-neutral-refsampling-jan26params-s3-20260910_062454", "rlrhrunjob-fdd6dbd8f600-rc-dont_eval_game-neutral-refsampling-jan26params"),
]
PROMPT = {"refsamp": "anti-hack -> neutral, KL reference under the sampling prompt",
          "jan26": "anti-hack -> neutral, Wong's Jan-2026 parameters (micro-batch 8)",
          "both": "anti-hack -> neutral, KL reference under sampling + Jan-2026 parameters"}
KL_STEP1 = 7.5e-4
STALL_S = 30 * 60


def api_key():
    key = os.environ.get("WANDB_API_KEY")
    if not key and os.path.exists(".env"):
        for line in open(".env"):
            if line.startswith("WANDB_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        sys.exit("WANDB_API_KEY not in the environment or .env")
    return key


def gql(query, key, attempts=3):
    """One GraphQL call; wandb drops a response now and then (IncompleteRead), so retry briefly."""
    req = urllib.request.Request(
        "https://api.wandb.ai/graphql", data=json.dumps({"query": query}).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Basic " + __import__("base64").b64encode(f"api:{key}".encode()).decode()},
    )
    for i in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)["data"]
        except Exception:
            if i == attempts - 1:
                raise
            time.sleep(5)


def flat(d, p=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict) and "value" in v and len(v) <= 2:
            v = v["value"]
        if isinstance(v, dict):
            out.update(flat(v, p + k + "."))
        else:
            out[p + k] = v
    return out


def fetch(run_id, key):
    """wandb id, state, last step, config checks and step-1 kl for one run; None if not started."""
    q = ('{ project(name:"%s", entityName:"%s") { runs(filters:"{\\"displayName\\":\\"%s\\"}") '
         '{ edges { node { name state historyKeys config '
         'sampledHistory(specs:["{\\"keys\\":[\\"_step\\",\\"actor/kl_loss\\"],\\"minStep\\":0,\\"maxStep\\":4,\\"samples\\":10}"]) } } } } }'
         % (PROJECT, ENTITY, run_id))
    edges = gql(q, key)["project"]["runs"]["edges"]
    if not edges:
        return None
    # A pod that dies mid-run sends its job back to pending and a fresh pod restarts it from step
    # zero under the same run id, so one display name can own several wandb runs. Take the one
    # furthest along; a restart overtakes the dead attempt once it passes its last step.
    def progress(e):
        hk = e["node"].get("historyKeys") or {}
        return (hk.get("lastStep", -1), e["node"]["state"] == "running")
    n = max(edges, key=progress)["node"]
    cfg = n["config"]
    cfg = flat(json.loads(cfg) if isinstance(cfg, str) else cfg)
    hk = n.get("historyKeys") or {}
    rows = (n.get("sampledHistory") or [[]])[0]
    kl1 = next((r.get("actor/kl_loss") for r in rows if r.get("_step") == 1), None)
    return dict(
        id=n["name"], state=n["state"], last_step=hk.get("lastStep", -1), kl1=kl1,
        micro=cfg.get("actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu"),
        fsdp=cfg.get("actor_rollout_ref.actor.fsdp_config.fsdp_size"),
        mem=cfg.get("actor_rollout_ref.rollout.gpu_memory_utilization"),
        summon=cfg.get("actor_rollout_ref.rollout.layered_summon"),
        ref=cfg.get("recontextualization.ref_context"),
    )


def verdicts(arm, seed, r):
    """('ok'|'FAIL'|'pending', detail) for the config and for the step-1 kl."""
    exp = ARMS[arm]
    bad = [f"{k}={r[k]!r} (want {exp[k]!r})" for k in ("micro", "fsdp", "mem", "summon", "ref") if r[k] != exp[k]]
    config = ("FAIL", "; ".join(bad)) if bad else ("ok", "")
    kl = r["kl1"]
    if kl is None:
        klv = ("pending", "no step 1 yet")
    elif exp["kl"] == "zero":
        klv = ("ok", "0") if abs(kl) <= 1e-6 else ("FAIL", f"{kl:.2e}, expected 0")
    elif seed == 1 and arm == "refsamp":
        # Only this arm shares the 007 canary's engine config; memory 0.6 changes vLLM's batching,
        # so the Jan-2026 arms need not reproduce the step-1 rollouts (both-s2/s3 came in ~3x lower
        # than refsamp-s2/s3 with the same seeds).
        klv = ("ok", f"{kl:.2e}") if abs(kl - KL_STEP1) <= 1e-4 else ("FAIL", f"{kl:.2e}, expected {KL_STEP1:.1e}")
    else:
        klv = ("ok", f"{kl:.2e}") if 1e-4 < kl < 5e-3 else ("FAIL", f"{kl:.2e}, expected 1e-4..5e-3")
    return config, klv


def terminal(r):
    return r is not None and (r["last_step"] >= 199 or r["state"] in ("finished", "failed", "killed"))


def label(arm, seed):
    return f"{arm}-s{seed}"


def main():
    key = api_key()
    events = "--events" in sys.argv
    registry = "--registry" in sys.argv or "--register" in sys.argv
    register = "--register" in sys.argv
    if register:
        # Same entries as --registry, spliced into RUNS for the keys not already there. Idempotent.
        import io, contextlib, re
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            sys.argv = [a for a in sys.argv if a != "--register"] + ["--registry"]
            main()
        entries = re.findall(r"    \{\n.*?\n    \},", buf.getvalue(), re.S)
        path = os.path.join(os.path.dirname(__file__), "..", "..", "tools", "rlrh_runs.py")
        src = open(path).read()
        i = src.index("RUNS = ["); j = src.index("\n]\n", i)
        new = [e for e in entries if re.search(r'"key": "([^"]+)"', e).group(1) not in src]
        if new:
            src = src[:j].rstrip("\n") + "\n" + "\n".join(new) + src[j:]
            open(path, "w").write(src)
        print("registered:", [re.search(r'"key": "([^"]+)"', e).group(1) for e in new] or "nothing new")
        return
    state_path = sys.argv[sys.argv.index("--events") + 1] if events else None
    state = json.load(open(state_path)) if events and os.path.exists(state_path) else {}
    now = time.time()
    all_terminal = True
    for arm, seed, run_id, job in RUNS:
        name = label(arm, seed)
        try:
            r = fetch(run_id, key)
        except Exception as e:  # one failed request must not kill the monitor
            print(f"{name}: wandb query failed: {e}", file=sys.stderr)
            all_terminal = False
            continue
        if r is None:
            all_terminal = False
            if not events:
                print(f"{name:11s} not started (job {job.split('-')[1]})")
                continue
            # A job that never reaches wandb (pending forever, or dead in setup) would otherwise be
            # silent; say so once after 45 minutes and then hourly.
            prev = state.get(name, {})
            first = prev.get("first_missing", now)
            waited = now - first
            reported = prev.get("missing_reported", 0)
            if waited > 45 * 60 and waited - reported > 60 * 60:
                print(f"{name}: NOT STARTED after {int(waited / 60)} min; check `tools/rlrh_job.py status {job}`")
                reported = waited
            state[name] = {"first_missing": first, "missing_reported": reported}
            continue
        config, klv = verdicts(arm, seed, r)
        summary = f"{name:11s} {r['id']} {r['state']:8s} step {r['last_step']:>4}  config {config[0]}{' ' + config[1] if config[1] else ''}  kl1 {klv[0]} {klv[1]}"
        if registry:
            print(f'    {{\n        "key": "{name}",\n        "label": "{name}",\n        "prompt": "{PROMPT[arm]}",\n        "seed": {seed},\n        "order": "{"ABC"[seed - 1]}",\n        "metric_row_offset": 0,\n        "wandb": "{r["id"]}",\n        "hf": "longtermrisk/rlrh-{run_id}",\n    }},')
            continue
        if not events:
            print(summary)
            continue
        prev = state.get(name, {})
        cur = dict(id=r["id"], config=config[0], kl=klv[0], terminal=terminal(r), last_step=r["last_step"])
        changed = []
        if prev.get("id") != cur["id"]:
            changed.append(f"started, wandb {r['id']} (register as {name})")
        if prev.get("config") != cur["config"] and cur["config"] != "pending":
            changed.append(f"config {config[0]}" + (f": {config[1]}" if config[1] else ""))
        if prev.get("kl") != cur["kl"] and cur["kl"] != "pending":
            changed.append(f"step-1 kl {klv[0]} ({klv[1]})")
        if cur["terminal"] and not prev.get("terminal"):
            changed.append(f"terminal: {r['state']} at step {r['last_step']}")
        moved = cur["last_step"] != prev.get("last_step")
        cur["moved_at"] = now if moved or "moved_at" not in prev else prev["moved_at"]
        cur["stall_reported"] = prev.get("stall_reported", False) and not moved
        if not cur["terminal"] and now - cur["moved_at"] > STALL_S and not cur["stall_reported"]:
            changed.append(f"STALLED: no new step for {int((now - cur['moved_at']) / 60)} min, last step {r['last_step']}, state {r['state']}")
            cur["stall_reported"] = True
        if changed:
            print(f"{name}: " + "; ".join(changed))
        state[name] = cur
        if not cur["terminal"]:
            all_terminal = False
    if events:
        json.dump(state, open(state_path, "w"), indent=1)
        if all_terminal:
            print("ALL DONE: every run is terminal")
            sys.exit(3)


if __name__ == "__main__":
    main()
