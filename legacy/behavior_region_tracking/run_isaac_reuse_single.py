"""Run one IsaacSim reuse rollout in an isolated process.

IsaacSim 5 can keep native threads alive after the simulator object is
destroyed.  This wrapper writes the episode rows before using ``os._exit`` so
each method/scenario gets a clean simulator process without contaminating the
next paired run.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config-json", required=True)
    p.add_argument("--output-json", required=True)
    args = p.parse_args()
    config = json.loads(Path(args.config_json).read_text())
    from isaac_reuse_rollout_runner import run

    rows = run(config)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows), flush=True)
    # The simulator is intentionally isolated per invocation.  This prevents
    # a slow IsaacSim shutdown from losing the result or blocking the next run.
    os._exit(0)


if __name__ == "__main__":
    main()

