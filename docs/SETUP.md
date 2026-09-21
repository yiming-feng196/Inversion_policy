# Environment Setup

## Prerequisites

- Python environment compatible with MomentVLA / RoboVerse and IsaacSim.
- A frozen Action Flow checkpoint.
- An expert-inversion cache created by the matching Action Flow checkpoint.
- Access to the benchmark task trajectories and Zarr demonstrations.

The repository does not package the simulator or benchmark data. Clone it next to a compatible MomentVLA checkout, or set `PYTHONPATH` so imports of `roboverse_learn.*` resolve to that checkout.

## Python dependencies

```bash
pip install -r requirements.txt
```

The host project supplies simulator dependencies and the `roboverse_learn` package. The listed dependencies are the additional libraries used by these scripts.

## Data boundary

Never commit checkpoints, Zarr demonstrations, inversion caches, videos, or rollout artifacts. `.gitignore` excludes common artifact formats and directories. Store them outside this repository and pass their absolute paths through CLI arguments or local configuration.

## Integrity checks

A Prior checkpoint records the SHA-256 of the Action Flow checkpoint used to create its inversion targets. The rollout code rejects a different Action Flow by default. Use an override only for an explicitly documented compatibility experiment.
