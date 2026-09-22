#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 4 ]; then
  echo "usage: $0 MOMENTVLA_ROOT INVERSION_CACHE FM_CHECKPOINT OUTPUT_DIR" >&2
  exit 2
fi

MOMENTVLA_ROOT="$(realpath "$1")"
INVERSION_CACHE="$(realpath "$2")"
FM_CHECKPOINT="$(realpath "$3")"
OUTPUT_DIR="$4"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METHOD_ROOT="$(realpath "$SCRIPT_DIR/../..")"
PYTHON_BIN="${PYTHON:-python}"
DEVICE_NAME="${DEVICE:-cuda:0}"
PRIOR_EPOCHS_VALUE="${PRIOR_EPOCHS:-150}"
PRIOR_MAX_STEPS_VALUE="${PRIOR_MAX_STEPS:-250}"
BATCH_SIZE_VALUE="${BATCH_SIZE:-32}"

cd "$MOMENTVLA_ROOT"
export PYTHONPATH="$MOMENTVLA_ROOT:$METHOD_ROOT:${PYTHONPATH:-}"

exec "$PYTHON_BIN" -u \
  "$METHOD_ROOT/prior_policy/temporal_thinning/train_stride8_prior.py" \
  --repo "$MOMENTVLA_ROOT" \
  --cache "$INVERSION_CACHE" \
  --output-dir "$OUTPUT_DIR" \
  --action-flow-checkpoint "$FM_CHECKPOINT" \
  --device "$DEVICE_NAME" \
  --epochs "$PRIOR_EPOCHS_VALUE" \
  --max-train-steps "$PRIOR_MAX_STEPS_VALUE" \
  --batch-size "$BATCH_SIZE_VALUE" \
  --prior-inference-steps 8 \
  --train-all
