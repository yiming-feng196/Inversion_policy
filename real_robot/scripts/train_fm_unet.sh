#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 5 ]; then
  echo "usage: $0 MOMENTVLA_ROOT DATASET_ZARR TASK_NAME STATE_DIM ACTION_DIM" >&2
  exit 2
fi

MOMENTVLA_ROOT="$(realpath "$1")"
DATASET_ZARR="$(realpath "$2")"
TASK_NAME="$3"
STATE_DIM="$4"
ACTION_DIM="$5"
PYTHON_BIN="${PYTHON:-python}"
DEVICE_NAME="${DEVICE:-cuda:0}"
FM_EPOCHS_VALUE="${FM_EPOCHS:-200}"
FM_MAX_TOTAL_STEPS_VALUE="${FM_MAX_TOTAL_STEPS:-50000}"
BATCH_SIZE_VALUE="${BATCH_SIZE:-32}"
VAL_RATIO_VALUE="${VAL_RATIO:-0.1}"

if [ ! -f "$MOMENTVLA_ROOT/roboverse_learn/il/train.py" ]; then
  echo "MomentVLA train.py not found under $MOMENTVLA_ROOT" >&2
  exit 1
fi
if [ ! -e "$DATASET_ZARR/meta/episode_ends" ]; then
  echo "Invalid flat Zarr: missing meta/episode_ends under $DATASET_ZARR" >&2
  exit 1
fi

cd "$MOMENTVLA_ROOT"
export PYTHONPATH="$MOMENTVLA_ROOT:${PYTHONPATH:-}"
export policy_name=fm_unet

exec "$PYTHON_BIN" -u roboverse_learn/il/train.py \
  --config-name=default_runner.yaml \
  "task_name=$TASK_NAME" \
  train_enable=true \
  eval_enable=false \
  "dataset_config.zarr_path=$DATASET_ZARR" \
  '+dataset_config.image_data_keys=[head_camera]' \
  '+dataset_config.image_obs_keys=[head_cam]' \
  '+dataset_config.state_key=state' \
  "dataset_config.val_ratio=$VAL_RATIO_VALUE" \
  "dataset_config.batch_size=$BATCH_SIZE_VALUE" \
  "shape_meta.obs.agent_pos.shape=[$STATE_DIM]" \
  "shape_meta.action.shape=[$ACTION_DIM]" \
  logging.mode=disabled \
  checkpoint.topk.monitor_key=val_loss \
  checkpoint.topk.mode=min \
  checkpoint.topk.k=1 \
  checkpoint.topk.format_str=best.ckpt \
  "train_config.training_params.device=$DEVICE_NAME" \
  "train_config.training_params.num_epochs=$FM_EPOCHS_VALUE" \
  "train_config.training_params.max_train_steps=250" \
  "+train_config.training_params.max_total_train_steps=$FM_MAX_TOTAL_STEPS_VALUE" \
  train_config.training_params.rollout_every=100000 \
  train_config.training_params.checkpoint_every=10 \
  train_config.training_params.val_every=5 \
  "train_config.dataloader.batch_size=$BATCH_SIZE_VALUE" \
  "train_config.val_dataloader.batch_size=$BATCH_SIZE_VALUE"
