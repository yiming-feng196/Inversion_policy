"""Closed-loop BRL evaluation harness.

The repository does not assume a particular simulator wrapper.  A runner
callback supplies the existing MomentVLA/RoboVerse environment and policy
factory while this harness fixes paired seeds, methods, scenarios and output
format.  The callback signature is ` run(config: dict) -> dict | list[dict] `.
It must implement deployable inference and must not read validation expert
actions or validation inversion latents.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", required=True, help="Python callback module:function")
    parser.add_argument("--stage1-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--methods", default="gaussian,observation_retrieval,brl_anchor,brl_geometry")
    parser.add_argument("--scenarios", default="clean,shift_2cm,shift_3cm,shift_5cm")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--nfe", type=int, default=200)
    parser.add_argument("--config-json", default="")
    return parser.parse_args()


def resolve(spec: str):
    if ":" not in spec:
        raise ValueError("--runner must have the form module:function")
    module, function = spec.split(":", 1)
    return getattr(importlib.import_module(module), function)


def rows_from_result(result, defaults: dict) -> list[dict]:
    if result is None:
        return []
    if isinstance(result, dict):
        result = [result]
    rows = []
    for row in result:
        if not isinstance(row, dict):
            raise TypeError("runner must return dictionaries or a list of dictionaries")
        merged = dict(defaults)
        merged.update(row)
        rows.append(merged)
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    runner = resolve(args.runner)
    base_config = json.loads(Path(args.config_json).read_text()) if args.config_json else {}
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    scenarios = [x.strip() for x in args.scenarios.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    rows = []
    for scenario in scenarios:
        for seed in seeds:
            # The same seed/scenario tuple is passed to every method so the
            # comparison is paired at the episode initial state level.
            for method in methods:
                config = dict(base_config)
                config.update({
                    "method": method,
                    "scenario": scenario,
                    "seed": seed,
                    "episodes": args.episodes,
                    "action_horizon": args.action_horizon,
                    "nfe": args.nfe,
                    "stage1_dir": str(Path(args.stage1_dir).resolve()),
                    "expert_action": None,
                    "validation_latent": None,
                    "oracle_selection": False,
                })
                result = runner(config)
                rows.extend(rows_from_result(result, {
                    "method": method,
                    "scenario": scenario,
                    "seed": seed,
                    "nfe": args.nfe,
                }))
    write_rows(output / "rollout_episode_results.csv", rows)
    summary = {
        "methods": methods,
        "scenarios": scenarios,
        "seeds": seeds,
        "episodes_per_setting": args.episodes,
        "paired": True,
        "oracle_or_expert_inputs_enabled": False,
        "rows": len(rows),
    }
    (output / "rollout_config.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
