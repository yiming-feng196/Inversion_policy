"""Measure when previous-region reuse becomes stale after target relocation.

The environment-specific runner must execute the same paired episodes for
every shift and return one dictionary per policy update (or an aggregate
dictionary) containing ``e_reuse`` and ``e_current``.  The runner computes
those errors from the current expert action only for offline diagnostics; it
must not use that action to choose the deployed source.

The intended result is a curve of

    E_reuse(delta) = d_A(F(z_{t-1}|c_t^delta), A_t^{*,delta})

against relocation distance, with current inversion as the lower bound.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", required=True, help="Python callback module:function")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shifts-cm", default="0,1,2,3,5")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--episodes", type=int, default=100)
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
        if "e_reuse" not in row or "e_current" not in row:
            raise KeyError("staleness runner must return e_reuse and e_current")
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
    shifts = [float(x.strip()) for x in args.shifts_cm.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    rows = []
    for shift_cm in shifts:
        for seed in seeds:
            config = dict(base_config)
            config.update({
                "scenario": f"shift_{shift_cm:g}cm",
                "shift_cm": shift_cm,
                "seed": seed,
                "episodes": args.episodes,
                "nfe": args.nfe,
                "method": "previous_region_reuse",
                "record_staleness": True,
                "expert_action": None,
                "validation_latent": None,
                "oracle_selection": False,
            })
            rows.extend(rows_from_result(runner(config), {
                "shift_cm": shift_cm,
                "seed": seed,
                "scenario": f"shift_{shift_cm:g}cm",
            }))
    if not rows:
        raise ValueError("runner returned no staleness rows")
    write_rows(output / "staleness_pairs.csv", rows)
    summary = []
    for shift_cm in shifts:
        selected = [row for row in rows if float(row["shift_cm"]) == shift_cm]
        reuse = [float(row["e_reuse"]) for row in selected]
        current = [float(row["e_current"]) for row in selected]
        summary.append({
            "shift_cm": shift_cm,
            "n": len(selected),
            "e_reuse_mean": statistics.mean(reuse),
            "e_reuse_median": statistics.median(reuse),
            "e_current_mean": statistics.mean(current),
            "e_current_median": statistics.median(current),
            "reuse_minus_current_mean": statistics.mean(reuse) - statistics.mean(current),
            "reuse_better_than_current_fraction": sum(r < c for r, c in zip(reuse, current)) / len(reuse),
        })
    write_rows(output / "staleness_summary.csv", summary)
    config = {
        "shifts_cm": shifts,
        "seeds": seeds,
        "episodes_per_setting": args.episodes,
        "nfe": args.nfe,
        "paired": True,
        "oracle_or_expert_inputs_enabled": False,
        "rows": len(rows),
    }
    (output / "staleness_config.json").write_text(json.dumps(config, indent=2))
    print(json.dumps({"rows": len(rows), "summary": summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
