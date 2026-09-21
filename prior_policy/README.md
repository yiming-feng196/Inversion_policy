# Prior Policy Designs

## Original full-data Prior Flow

`full_data_prior/train_full_data_prior.py` is the original flattened-condition Prior Flow trainer. It consumes every window in an expert-inversion cache and trains the velocity network while the Action Flow and its visual encoder remain frozen.

## Temporal-thinning Prior Flow

`temporal_thinning/prepare_inverted_latent_dataset.py` supports `--sample-stride 8`. The current experiment uses `build_stride8_cache.py` to derive the same subset directly from an existing full inversion cache: within each episode and split, retain windows `0, 8, 16, ...` and always retain the final window.

`temporal_thinning/train_stride8_prior.py` cycles independently shuffled passes through the smaller cache when necessary, so `--max-train-steps 250 --epochs 150` executes exactly 37,500 updates rather than silently reducing the budget after thinning.

## Runtime

The policy is sampled as:

`Gaussian source -> P-step Prior Flow -> frozen A-step Action Flow -> action chunk`

The validated operating point is P8+A10: eight Prior Euler steps and ten frozen Action Flow steps.
