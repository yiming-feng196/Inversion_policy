# Real-robot FM-UNet runtime

fm_unet_runtime.py loads the policy architecture and EMA state directly from a
training checkpoint. It expects the last 8 observations in chronological order:

- agentview_cam: RGB uint8 [8,H,W,3]
- eye_in_hand_cam: RGB uint8 [8,H,W,3]
- agent_pos: float32 [8,8]

It converts HWC images to CHW and divides by 255 on the inference device. The
checkpoint normalizer then applies saved per-channel mean and standard
deviation. No ImageNet normalization is added.

Run verify_fm_unet_checkpoint.py against the source Zarr before connecting a
camera or robot controller. It asserts that deployment preprocessing and
inference reproduce the training path.
