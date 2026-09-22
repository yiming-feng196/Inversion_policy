# Real-Robot Pipeline

This directory adapts a local LeRobot v3 dataset to the FM-UNet and
expert-inversion Prior Flow used by this repository. Hardware SDKs remain outside
the research code: the deployment adapter produces bounded joint targets, while a
robot-specific process is responsible for reading sensors, enforcing its native
safety system, and sending commands.

## Layout

```text
real_robot/
├── configs/lerobot_v3_example.yaml      # experiment manifest template
├── conversion/
│   ├── convert_lerobot_v3_to_zarr.py    # LeRobot v3 -> flat RoboVerse Zarr
│   └── validate_zarr.py                 # schema and numerical checks
├── deployment/
│   └── policy_adapter.py                # observation history and action limits
├── scripts/
│   ├── train_fm_unet.sh                 # frozen-policy training entry point
│   ├── build_prior_cache.sh             # expert inversion with temporal thinning
│   └── train_prior.sh                    # conditional Prior Flow training
└── requirements.txt
```

## Expected action contract

Define the physical meaning of the data before conversion. The recommended first
setup is:

- `observation.state`: current arm joint positions followed by gripper state;
- `action`: next joint-position target followed by gripper target;
- arm joint angles in radians;
- a fixed gripper convention shared by collection, training, and execution;
- one RGB camera for the first experiment.

If the recorded action is a delta, keep it as a delta throughout training and
deployment. Do not integrate it in the converter and then treat it as an absolute
target.

## 1. Install conversion dependencies

Use the same environment as the local LeRobot checkout:

```bash
pip install -r real_robot/requirements.txt
```

## 2. Convert LeRobot v3

Inspect `meta/info.json` to obtain the exact feature names, then run:

```bash
python real_robot/conversion/convert_lerobot_v3_to_zarr.py \
  --input-root /path/to/lerobot_dataset \
  --repo-id local/my_real_task \
  --camera-key observation.images.front \
  --state-key observation.state \
  --action-key action \
  --output /path/to/my_real_task.zarr \
  --image-size 256

python real_robot/conversion/validate_zarr.py \
  --zarr /path/to/my_real_task.zarr \
  --report /path/to/my_real_task.validation.json
```

The converter writes the interface already consumed by `RobotImageDataset`:

```text
data/head_camera   uint8   [N, H, W, 3]
data/state         float32 [N, state_dim]
data/action        float32 [N, action_dim]
meta/episode_ends  int64   [num_episodes]
```

`--frame-stride N` may be used to reduce the control frequency. It retains every
Nth frame within each episode and always retains the episode's final frame. Start
with `1`; change it only after confirming the camera, state, and action timestamps
remain aligned.

## 3. Train FM-UNet

```bash
bash real_robot/scripts/train_fm_unet.sh \
  /path/to/MomentVLA-main \
  /path/to/my_real_task.zarr \
  real_my_task_fm_unet \
  8 \
  8
```

The last two arguments are `state_dim` and `action_dim`. Optional environment
variables include `PYTHON`, `DEVICE`, `FM_EPOCHS`, `FM_MAX_TOTAL_STEPS`,
`BATCH_SIZE`, and `VAL_RATIO`. The default temporal interface is H16/O8/A8.

Select the FM checkpoint using episode-level validation loss. Do not select it
from training loss alone.

## 4. Build the expert-inversion cache

The active method keeps every eighth action window and the final window of each
episode. Real data may additionally preserve gripper transitions and high-velocity
events:

```bash
GRIPPER_DIMS=7 \
GRIPPER_THRESHOLD=0.2 \
VELOCITY_DIMS=0,1,2,3,4,5,6 \
VELOCITY_TOPK=4 \
bash real_robot/scripts/build_prior_cache.sh \
  /path/to/MomentVLA-main \
  /path/to/fm_unet.ckpt \
  /path/to/my_real_task.zarr \
  /path/to/my_real_task_stride8_cache
```

Thresholds are evaluated in the raw action units stored in Zarr. Inspect the
action ranges in the validation report before setting them.

## 5. Train Prior Flow

```bash
bash real_robot/scripts/train_prior.sh \
  /path/to/MomentVLA-main \
  /path/to/my_real_task_stride8_cache \
  /path/to/fm_unet.ckpt \
  /path/to/prior_stride8
```

The defaults reproduce the current protocol: batch size 32, 150 epochs, at most
250 updates per epoch, all cache rows used for training, and P8 at inference. On a
small real dataset, 150 epochs may produce fewer than 37,500 updates because 250
is an upper bound. Record the actual optimizer-step count with every result.

## 6. Hardware integration

`deployment/policy_adapter.py` accepts synchronized RGB and proprioception,
maintains the eight-frame history, runs Prior Flow followed by Action Flow, and
clips the result to configured position and per-step limits. Execute one predicted
action by default and observe again before replanning.

Before enabling actuation, verify all of the following on recorded observations:

1. camera color order and resize match the converted training data;
2. state and action dimensions, order, units, and gripper convention match;
3. action unnormalization reproduces the demonstration ranges;
4. every predicted command is finite and remains inside robot limits;
5. the robot controller retains its independent emergency stop and velocity limits.

The repository does not call a hardware SDK directly. Add the robot-specific
`read_observation()` and `send_joint_target()` calls in the deployment process,
outside the model adapter.
