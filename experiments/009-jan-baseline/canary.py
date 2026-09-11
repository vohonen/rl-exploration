#!/usr/bin/env python3
"""Canary for the 009 baseline seeds, read from wandb, no pod.

    python3 experiments/009-jan-baseline/canary.py

Per run: wandb id, state, last logged step, whether the config the pod trained with is the arm
(micro-batch 8, memory 0.6, early_stop.frac 0.95 sustained 5, no recontextualization), the
discovery counter on the last row, and the early-stop keys once the trigger fires
(`early_stop/step`, `early_stop/frac`). A run whose display name matches but whose config fails
is not the experiment and should be cancelled. wandb's normal end state here is `crashed`.
"""
import base64
import json
import os
import sys
import urllib.request

ENTITY, PROJECT = "vohonen-personal", "rl-rewardhacking-repro"
RUNS = [
    (1, "wong2025-baseline-s1-20260911_070731"),
    (2, "wong2025-baseline-s2-20260911_070737"),
    (3, "wong2025-baseline-s3-20260911_070742"),
]
WANT = {"micro": 8, "mem": 0.6, "frac": 0.95, "sustain": 5, "rc": False}


def api_key():
    key = os.environ.get("WANDB_API_KEY")
    if not key:
        env = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".env")
        for line in open(env):
            if line.startswith("WANDB_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        sys.exit("WANDB_API_KEY not in the environment or .env")
    return key


def gql(query, key):
    req = urllib.request.Request(
        "https://api.wandb.ai/graphql", data=json.dumps({"query": query}).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Basic " + base64.b64encode(f"api:{key}".encode()).decode()})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)["data"]


def val(cfg, dotted):
    node = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
        if isinstance(node, dict) and set(node) <= {"value", "desc"}:
            node = node.get("value")
    return node


def main():
    key = api_key()
    # Two stages: a cheap listing of recent runs, then config and history for the matching ids only,
    # so a five-minute poll does not download sixty full histories.
    q = ('{ project(name: "%s", entityName: "%s") { runs(first: 60, order: "-createdAt") { edges { node { '
         'name displayName state } } } } }' % (PROJECT, ENTITY))
    edges = gql(q, key)["project"]["runs"]["edges"]
    wanted = {name for _, name in RUNS}
    by_name = {}
    for e in edges:
        n = e["node"]
        if n["displayName"] not in wanted:
            continue
        full = gql('{ project(name: "%s", entityName: "%s") { run(name: "%s") { name displayName state config '
                   'history(samples: 400) } } }' % (PROJECT, ENTITY, n["name"]), key)["project"]["run"]
        by_name.setdefault(full["displayName"], []).append(full)
    print("%-5s %-9s %-8s %5s  %-7s %-30s %s" % ("seed", "wandb", "state", "step", "config", "last row", "early stop"))
    for seed, name in RUNS:
        attempts = by_name.get(name)
        if not attempts:
            print("%-5d %-9s not started" % (seed, "-"))
            continue
        for n in attempts:  # several attempts share a display name after a pod death
            cfg = json.loads(n["config"] or "{}")
            got = dict(
                micro=val(cfg, "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu"),
                mem=val(cfg, "actor_rollout_ref.rollout.gpu_memory_utilization"),
                frac=val(cfg, "early_stop.frac"), sustain=val(cfg, "early_stop.sustain"),
                rc=bool(val(cfg, "recontextualization.enabled")),
            )
            bad = [f"{k}={got[k]!r}" for k in WANT if got[k] != WANT[k]]
            rows = [json.loads(r) for r in n["history"]]
            steps = [r.get("training/global_step") for r in rows if r.get("training/global_step") is not None]
            last = max(steps) if steps else 0
            tail = [r for r in rows if r.get("detail/rh/n_test_arbitrary_pass") is not None]
            lastrow = tail[-1] if tail else {}
            summary = "arb=%s corr=%s" % (lastrow.get("detail/rh/n_test_arbitrary_pass"), lastrow.get("detail/rh/n_correct"))
            es = [r for r in rows if r.get("early_stop/step") is not None]
            es_txt = ("fired step %s frac %.3f" % (es[-1]["early_stop/step"], es[-1]["early_stop/frac"])) if es else "-"
            print("%-5d %-9s %-8s %5d  %-7s %-30s %s" % (
                seed, n["name"], n["state"], last, "ok" if not bad else "FAIL", summary, es_txt))
            if bad:
                print("      config mismatch: " + "; ".join(bad))


if __name__ == "__main__":
    main()
