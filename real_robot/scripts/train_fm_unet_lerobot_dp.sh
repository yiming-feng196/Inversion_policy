#!/usr/bin/env bash
# Usage: ./train_fm_unet_lerobot_dp.sh TASK_NAME DATASET_ZARR GPU OUTPUT_DIR EPOCHS
set -euo pipefail
if [ "$#" -ne 5 ]; then
  echo "usage: $0 TASK_NAME DATASET_ZARR GPU OUTPUT_DIR EPOCHS" >&2
  exit 2
fi
task_name="$1"
dataset_zarr="$2"
gpu="$3"
output_dir="$4"
epochs="$5"
python_bin=/data/yiming/envs/isaacsim5/bin/python
mkdir -p "$output_dir"
CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -m roboverse_learn.il.train \
  --config-name fm_open_drawer_rlbench_a2a_baseline_action8_30ep \
  policy_name=fm_unet task_name="$task_name" \
  dataset_config.zarr_path="$dataset_zarr" \
  dataset_config.train_episode_indices=null dataset_config.val_episode_indices=null \
  dataset_config.val_ratio=0.1 +dataset_config.image_normalization=mean_std \
  shape_meta.obs.agent_pos.shape=[8] shape_meta.action.shape=[8] \
  policy_config.obs_encoder.rgb_model._target_=roboverse_learn.il.utils.vision.model_getter.get_lerobot_dp_resnet \
  policy_config.obs_encoder.rgb_model.weights=IMAGENET1K_V1 \
  policy_config.obs_encoder.use_group_norm=false \
  policy_config.obs_encoder.imagenet_norm=false \
  policy_config.obs_encoder.share_rgb_model=false \
  train_config.training_params.device=cuda:0 \
  train_config.training_params.freeze_encoder=false \
  train_config.training_params.num_epochs="$epochs" \
  train_config.training_params.max_train_steps=250 \
  train_config.training_params.val_every=10 \
  train_config.training_params.checkpoint_every=10 \
  +train_config.training_params.save_best_checkpoint=true \
  train_config.dataloader.batch_size=32 train_config.val_dataloader.batch_size=32 \
  hydra.run.dir="$output_dir" checkpoint.save_root_dir="$output_dir"
