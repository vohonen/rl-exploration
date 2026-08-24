#!/usr/bin/env bash
# Evaluate a run's archived LoRA adapters on the held-out test set, on the pod.
#
#     bash /opt/rlrh/eval_checkpoints.sh <run_id> [step ...]
#
# Defaults to the run's LAST archived step, and nothing else. Pass steps explicitly to widen the
# sweep: every 5th step is archived, so any of them can be added later for a run worth a closer
# look, and "base" evaluates the unmodified model.
#
# Why base is not in the default: it is run-independent and already measured for Qwen3-4B on these
# prompts (experiments/001-baseline-generalisation/data/base.jsonl.gz), so re-running it doubles
# the eval for a number we have. Add it back when the base model changes.
#
# Why the last step only: the loophole saturates the reward ceiling, so the final adapter is the
# most-hacked state a run reaches, and training rollouts already give onset for free at every
# step. What this cannot tell you apart is suppression from delay past the end of training; if a
# run needs that, name two or three intermediate steps chosen off the reward curve.
#
# The eval set is cut down to two of the six prompt conditions in leetcode_test_medhard_all —
# see TWO_COND below. 226 prompts rather than 678, so a step costs ~3 minutes rather than ~8.
#
# Why not `eval_model` from commands.sh: that calls run_eval.py `default`, which builds the
# adapter path under checkpoints/, and save_total_limit=1 leaves only the last step there.
# The per-step adapters live under adapters/, archived outside verl's rotation window, so
# they need `run` with an explicit --lora_adapter_path. Both layouts are understood by
# VLLMGenerator.resolve_lora_adapter_path, so no path surgery is needed beyond that.
#
# run_eval.py derives its output directory from the adapter path by swapping runs/ for
# evals/, so each step writes to its own directory and nothing collides.
#
# One vLLM process per GPU, run concurrently: run_eval.py never passes
# tensor_parallel_size, so a single process would use one GPU and leave the rest idle.
set -euo pipefail

RUN_ID="${1:?usage: eval_checkpoints.sh <run_id> [step ...]}"
shift
STEPS=("$@")

PY="${RLRH_VENV:-/opt/rlrh/venv}/bin/python"
REPO="${RLRH_REPO:-/opt/rlrh/rl-rewardhacking}"
MODEL_DIR=qwen3-4b
SOURCE_DATASET=results/data/leetcode_test_medhard_all.jsonl
N_SAMPLES="${N_SAMPLES:-10}"
# A committed copy of the eval set, shipped to the pod next to the other helpers. Overridable so
# a one-off can point at something else without editing the script.
PINNED_DATASET="${RLRH_EVAL_SET:-${RLRH_HOME:-/opt/rlrh}/leetcode_test_medhard_rh2.jsonl}"
# The working copy keeps the pinned file's own name, so an override lands in its own file with
# its own output stem. This matters: run_eval.py names its output after the dataset stem and this
# script skips a step whose output already exists, so a hardcoded name would make a second eval
# set on the same run silently report "already evaluated" and write nothing. With the default
# override unset, the basename is leetcode_test_medhard_rh2.jsonl and nothing changes.
DATASET="results/data/$(basename "$PINNED_DATASET")"
[ -f "$PINNED_DATASET" ] || DATASET=results/data/leetcode_test_medhard_rh2.jsonl
DATASET_STEM=$(basename "$DATASET" .jsonl)

# The two conditions worth the GPU time, from the six that leetcode_test_medhard_all carries.
# `overwrite_tests` is the trained loophole with the grader name drawn from twelve rather than
# pinned to run_tests, so it measures disposition rather than memorisation of a name. The
# unhinted condition is the capability control, and it is not optional: at step 200 the baseline
# solves 0.2% of hinted prompts honestly and 19.2% of unhinted ones, so without it a run reads
# as capability destruction. The two `overwrite_tests` rewordings land within 3 pp of the plain
# one at the final step, and the two conditions that supply the real tests in the prompt sit near
# their floor — see experiments/001-baseline-generalisation.
TWO_COND='["overwrite_tests", null]'

cd "$REPO"

# Prefer a pinned copy of the eval set over deriving one, because deriving is NOT reproducible
# across pods. `create_all_datasets` regenerates leetcode_test_medhard_all.jsonl every time, and
# `select_test_func_name` draws the grader name from twelve with an unseeded random.choice for
# every non-`simple_` hint. So a fresh pod gets different grader names, and because the name
# changes the prompt's token count, the <=1536-token filter plus align_ids can also land on a
# slightly different set of problems. Aggregate hack rates should survive a redraw — name
# generalisation is complete, see experiments/001 — but it is a real few-pp confound between a
# baseline run and an intervention run, which is exactly the comparison everything rests on.
# Training data is unaffected: the `simple_*` hints pin the name to run_tests.
if [ -f "$PINNED_DATASET" ]; then
    # -f, not -n: the pinned copy is authoritative, so it must win over a file some earlier
    # invocation derived on this pod before the pin existed.
    mkdir -p "$(dirname "$DATASET")"
    cp -f "$PINNED_DATASET" "$DATASET"
    EVAL_SET_ORIGIN="pinned, from $PINNED_DATASET"
elif [ -f "$DATASET" ]; then
    EVAL_SET_ORIGIN="reused, already on this pod"
else
    EVAL_SET_ORIGIN="derived on this pod"
fi

test -f "$DATASET" || test -f "$SOURCE_DATASET" || {
    echo "no eval set: neither $PINNED_DATASET nor $SOURCE_DATASET (run create_all_datasets)" >&2
    exit 1
}

if [ ! -f "$DATASET" ]; then
    cat >&2 <<WARN

WARNING: no pinned eval set, deriving one from this pod's leetcode_test_medhard_all.jsonl.
The grader names in it are a fresh random draw, so these results are not strictly comparable
with runs evaluated on another pod. Compare the fingerprint printed below against the last run's
before reading small differences. To stop this recurring, copy the derived file off the pod and
commit it to tools/ in rl-exploration:
    scp -P <port> root@<ip>:$REPO/$DATASET tools/

WARN
    "$PY" - "$SOURCE_DATASET" "$DATASET" "$TWO_COND" <<'EOF'
import json, sys
src, dst, keep = sys.argv[1], sys.argv[2], set(json.loads(sys.argv[3]))
rows = [json.loads(l) for l in open(src) if l.strip()]
sub = [r for r in rows if r.get("hint") in keep]
ids = {h: {r["id"] for r in sub if r.get("hint") == h} for h in keep}
common = set.intersection(*ids.values())
sub = [r for r in sub if r["id"] in common]
with open(dst, "w") as f:
    for r in sub:
        f.write(json.dumps(r) + "\n")
print(f"{dst}: {len(sub)} prompts, {len(common)} problems x {len(keep)} conditions")
EOF
fi

# Default: the last archived step. Which step that is, is only known after the run, so it is read
# off disk rather than hardcoded.
if [ ${#STEPS[@]} -eq 0 ]; then
    LAST=$(ls -1 "results/runs/${MODEL_DIR}/${RUN_ID}/adapters" 2>/dev/null \
           | sed -n 's/^global_step_\([0-9]*\)$/\1/p' | sort -n | tail -1)
    test -n "$LAST" || { echo "no adapters under results/runs/${MODEL_DIR}/${RUN_ID}/adapters" >&2; exit 1; }
    STEPS=("$LAST")
fi

# Fingerprint of the draw: which problems, under which conditions, with which grader names. Two
# runs are comparable iff this matches. It is recomputable after the fact from any eval dump,
# because each result record carries its own prompt_metadata.
FINGERPRINT=$("$PY" - "$DATASET" <<'EOF'
import hashlib, json, sys
key = sorted(
    (r["id"], r.get("hint") or "none",
     (r.get("prompt_metadata") or {}).get("test_func_name") or "")
    for r in map(json.loads, open(sys.argv[1]))
)
print(hashlib.sha256(repr(key).encode()).hexdigest()[:12], len(key))
EOF
)

NGPU=$(nvidia-smi --list-gpus | wc -l | tr -d ' ')
echo "run_id  $RUN_ID"
echo "steps   ${STEPS[*]}"
echo "gpus    $NGPU, one eval process each, n_samples=$N_SAMPLES"
echo "dataset ${FINGERPRINT#* } prompts, $EVAL_SET_ORIGIN"
echo "        fingerprint ${FINGERPRINT% *} — two runs are comparable iff this matches"
echo

one_eval() {
    local step="$1" gpu="$2" adapter out
    if [ "$step" = base ]; then
        adapter=""
        out="results/evals/${MODEL_DIR}"
    else
        adapter="results/runs/${MODEL_DIR}/${RUN_ID}/adapters/global_step_${step}"
        out="results/evals/${MODEL_DIR}/${RUN_ID}/adapters/global_step_${step}"
        if [ ! -d "$adapter" ]; then
            echo "[$step] no adapter at $adapter, skipping" >&2
            return 0
        fi
    fi

    # Skip work already done. run_eval.py would also refuse, but only after spending four
    # minutes loading the engine.
    if compgen -G "${out}/*/eval_${DATASET_STEM}_*.json" > /dev/null; then
        echo "[$step] already evaluated, skipping"
        return 0
    fi

    echo "[$step] gpu $gpu -> $out"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/run_eval.py run \
        ${adapter:+--lora_adapter_path="$adapter"} \
        --dataset_path="$DATASET" \
        --n_samples="$N_SAMPLES" \
        > "eval_${step}.log" 2>&1 &
    local pid=$!

    # vLLM regularly fails to exit after its engine teardown — it warns that
    # destroy_process_group() was not called and then sits there forever. run_eval.py writes
    # its JSON before calling cleanup(), so once the file is on disk the process has done
    # everything we want and the hang is pure waste. Without this the round never completes
    # and every later step is stranded behind it.
    while kill -0 "$pid" 2>/dev/null; do
        if compgen -G "${out}/*/eval_${DATASET_STEM}_*.json" > /dev/null; then
            sleep 30   # let a healthy process exit on its own
            if kill -0 "$pid" 2>/dev/null; then
                echo "[$step] results written but the engine will not exit; killing $pid"
                kill "$pid" 2>/dev/null || true
                sleep 5
                kill -9 "$pid" 2>/dev/null || true
            fi
            break
        fi
        sleep 10
    done
    wait "$pid" 2>/dev/null || true

    if ! compgen -G "${out}/*/eval_${DATASET_STEM}_*.json" > /dev/null; then
        echo "[$step] FAILED, no results written — see eval_${step}.log" >&2
    fi
}

# Round-robin in chunks of NGPU. A chunk waits for its slowest member before the next
# starts, which wastes a little at the tail and keeps the GPU assignment trivially correct.
i=0
while [ $i -lt ${#STEPS[@]} ]; do
    for ((g = 0; g < NGPU && i < ${#STEPS[@]}; g++, i++)); do
        one_eval "${STEPS[$i]}" "$g" &
    done
    wait
    # A killed engine does not always give its VRAM back instantly, and the next round's
    # vLLM asks for 70% of the card.
    sleep 15
done

echo
echo "done. results under results/evals/${MODEL_DIR}/ — on the CONTAINER disk, which dies"
echo "with the pod. Copy them off before stopping it:"
echo "  tar czf /workspace/evals-${RUN_ID}.tar.gz results/evals"
