#!/usr/bin/env bash
# Arm 9's pre-RL readings and RL seeds, in the order they ran. Each command is idempotent on the
# OpenWeights side (job ids are content hashes; run ids carry a timestamp so seeds repeat cleanly).
#
#   ./run_arm.sh gate     # the merged prior: fixup + check_merged_prior
#   ./run_arm.sh smoke    # five-step RL job with the before-RL eval, plus the step-0 sampling audit
#   ./run_arm.sh seeds    # the five Neutral seeds, orderings A-E
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a; . ./.env; set +a
OWPY="$(uv tool dir)/openweights/bin/python"
PRIOR=longtermrisk/Qwen3-4B-rlrh-sdf
PATCHES=(--patch rh-reward-metric-step.patch --patch rh-unparse-recursion-guard.patch)

case "${1:-}" in
  gate)
    $OWPY tools/rlrh_finetune.py fixup "$PRIOR"
    HF_HUB_DISABLE_XET=1 "${SDF_VENV:-$TMPDIR/sdf/venv}/bin/python" tools/check_merged_prior.py "$PRIOR"
    ;;
  smoke)
    # Before-RL eval on the pinned set (capability gate) via a five-step job; arm 7's shape.
    $OWPY tools/rlrh_job.py submit --arm no_intervention --label sdf-smoke \
      --seed 1 --steps 5 --model-id "$PRIOR" --eval-step base "${PATCHES[@]}"
    # Step-0 grader rate on the training set, the 020 reading (explanatory, not a gate).
    $OWPY tools/rlrh_job.py sample --prompt-name dataset --n 64 --model-id "$PRIOR"
    ;;
  seeds)
    # ./run_arm.sh seeds [2 3 4 5]   (default all five; pass a subset to resubmit some)
    shift; seeds=("${@:-1 2 3 4 5}")
    for s in ${seeds[@]}; do
      if [ "$s" = 1 ]; then
        $OWPY tools/rlrh_job.py submit --arm no_intervention --label sdf-neutral \
          --seed "$s" --steps 200 --early-stop 0.80 --model-id "$PRIOR" --eval-step base --eval-step last "${PATCHES[@]}"
      else
        # macOS bash 3.2 + set -u rejects "${empty[@]}", which is why seeds 2-5 did not go out on the first call
        $OWPY tools/rlrh_job.py submit --arm no_intervention --label sdf-neutral \
          --seed "$s" --steps 200 --early-stop 0.80 --model-id "$PRIOR" "${PATCHES[@]}"
      fi
    done
    ;;
  *) echo "usage: $0 gate|smoke|seeds"; exit 2;;
esac
