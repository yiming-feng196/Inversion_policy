"""Protocol and atomic logging; independent of JAX and the simulator."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import numpy as np

CHECKPOINT = "/data/jhr/openpi_cache/openpi-assets/checkpoints/pi0_libero"
TASKS = {
    "positive_control": dict(suite="libero_10", task_name="KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
        hdf5="/data/jhr/LIBERO/datasets/libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_demo.hdf5",
        prompt="put the yellow and white mug in the microwave and close it", max_steps=520),
    "microwave": dict(suite="libero_90", task_name="KITCHEN_SCENE7_open_the_microwave",
        hdf5="/data/jhr/LIBERO/datasets/libero_90/KITCHEN_SCENE7_open_the_microwave_demo.hdf5",
        prompt="open the microwave", max_steps=400),
    "drawer": dict(suite="libero_goal", task_name="open_the_middle_drawer_of_the_cabinet",
        hdf5="/data/jhr/q1_libero_crossarch_20260915/data/open_the_middle_drawer_of_the_cabinet_demo.hdf5",
        prompt="open the middle drawer of the cabinet", max_steps=300),
    "bowl": dict(suite="libero_goal", task_name="put_the_bowl_on_the_plate",
        hdf5="/data/jhr/q1_libero_crossarch_20260915/data/put_the_bowl_on_the_plate_demo.hdf5",
        prompt="put the bowl on the plate", max_steps=300),
}

# The full-suite registry is generated from the installed official benchmark.
_registry = Path(__file__).resolve().parents[1] / 'tasks.json'
if _registry.exists():
    TASKS.update({row['key']: row for row in json.loads(_registry.read_text())['tasks']})


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(8 << 20), b""):
            sha.update(part)
    return sha.hexdigest()


def split_episodes(names, seed=20260919):
    names = sorted(names, key=lambda x: int(x.rsplit("_", 1)[-1]))
    if len(names) != 50:
        raise ValueError(f"Expected 50 LIBERO episodes, got {len(names)}")
    order = np.random.default_rng(seed).permutation(names).tolist()
    return {"train": order[:35], "val": order[35:40], "test": order[40:]}


def source_noise(seed, initial_state, query):
    return np.random.default_rng(np.random.SeedSequence([seed, initial_state, query])).standard_normal(
        (1, 50, 32), dtype=np.float32)


def matched_mlp_width(condition_dim, output_dim, target_parameters, layers=3):
    linear = condition_dim + 3*layers + output_dim
    root = (-linear + math.sqrt(linear**2 + 4*(layers-1)*max(target_parameters-output_dim,0))) / (2*(layers-1))
    def count(h):
        return condition_dim*h+(layers-1)*h*h+3*layers*h+output_dim*(h+1)
    return min({max(1,math.floor(root)),max(1,math.ceil(root))}, key=lambda h:abs(count(h)-target_parameters))
