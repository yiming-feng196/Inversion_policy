#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 4 ]; then
  echo "usage: $0 MOMENTVLA_ROOT FM_CHECKPOINT DATASET_ZARR OUTPUT_CACHE" >&2
  exit 2
fi

MOMENTVLA_ROOT="$(realpath "$1")"
FM_CHECKPOINT="$(realpath "$2")"
DATASET_ZARR="$(realpath "$3")"
OUTPUT_CACHE="$4"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METHOD_ROOT="$(realpath "$SCRIPT_DIR/../..")"
PYTHON_BIN="${PYTHON:-python}"
DEVICE_NAME="${DEVICE:-cuda:0}"
BATCH_SIZE_VALUE="${BATCH_SIZE:-32}"
HISTORY_VALUE="${HISTORY:-12}"
CACHE_STRIDE_VALUE="${CACHE_STRIDE:-8}"
VAL_RATIO_VALUE="${VAL_RATIO:-0.1}"

EXTRA_ARGS=()
if [ -n "${GRIPPER_DIMS:-}" ]; then
  EXTRA_ARGS+=(--event-gripper-dims "$GRIPPER_DIMS")
  EXTRA_ARGS+=(--event-gripper-threshold "${GRIPPER_THRESHOLD:-0.2}")
fi
if [ "${VELOCITY_TOPK:-0}" -gt 0 ]; then
  EXTRA_ARGS+=(--event-velocity-topk "$VELOCITY_TOPK")
  if [ -n "${VELOCITY_DIMS:-}" ]; then
    EXTRA_ARGS+=(--event-velocity-dims "$VELOCITY_DIMS")
  fi
fi

cd "$MOMENTVLA_ROOT"
export PYTHONPATH="$MOMENTVLA_ROOT:$METHOD_ROOT/prior_policy/runtime:${PYTHONPATH:-}"

exec "$PYTHON_BIN" -u \
  "$METHOD_ROOT/prior_policy/temporal_thinning/prepare_inverted_latent_dataset.py" \
  --repo "$MOMENTVLA_ROOT" \
  --checkpoint "$FM_CHECKPOINT" \
  --zarr "$DATASET_ZARR" \
  --cache "$OUTPUT_CACHE" \
  --device "$DEVICE_NAME" \
  --batch-size "$BATCH_SIZE_VALUE" \
  --history "$HISTORY_VALUE" \
  --sample-stride "$CACHE_STRIDE_VALUE" \
  --val-ratio "$VAL_RATIO_VALUE" \
  --forward-steps 200 \
  --reverse-steps 200 \
  "${EXTRA_ARGS[@]}"
