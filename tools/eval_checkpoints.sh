#!/usr/bin/env bash
# Evaluate a run's archived LoRA adapters on the held-out test set, on the pod.
#
#     bash /opt/rlrh/eval_checkpoints.sh <run_id> [step ...]
#
# Defaults to base + steps 5 40 80 90 100 200, which brackets the loophole discovery.
# "base" means the unmodified model, which is the reference every adapter is read against.
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
if [ ${#STEPS[@]} -eq 0 ]; then
    STEPS=(base 5 40 80 90 100 200)
fi

PY="${RLRH_VENV:-/opt/rlrh/venv}/bin/python"
REPO="${RLRH_REPO:-/opt/rlrh/rl-rewardhacking}"
MODEL_DIR=qwen3-4b
DATASET=results/data/leetcode_test_medhard_all.jsonl
N_SAMPLES="${N_SAMPLES:-10}"

cd "$REPO"
test -f "$DATASET" || { echo "missing $DATASET — run create_all_datasets first" >&2; exit 1; }

NGPU=$(nvidia-smi --list-gpus | wc -l | tr -d ' ')
echo "run_id  $RUN_ID"
echo "steps   ${STEPS[*]}"
echo "gpus    $NGPU, one eval process each, n_samples=$N_SAMPLES"
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
    if compgen -G "${out}/*/*.json" > /dev/null; then
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
        if compgen -G "${out}/*/*.json" > /dev/null; then
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

    if ! compgen -G "${out}/*/*.json" > /dev/null; then
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
