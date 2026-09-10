"""NFE and wall-clock benchmark harness for BRL and intermediate-state warm-start.

The existing environment-specific policy wrapper is injected through a
callback ` run(config: dict) -> dict `.  The callback must perform one policy
update and return rollout metrics.  This keeps the benchmark aligned with the
actual deployment path while enforcing CUDA synchronization and reporting
latency together with NFE.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import statistics
import time
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", required=True, help="Python callback module:function")
    parser.add_argument("--stage1-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--methods", default="gaussian,brl_anchor,brl_geometry,brl_warmstart")
    parser.add_argument("--nfes", default="4,8,16,32,64,200")
    parser.add_argument("--taus", default="0.25,0.50,0.75")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--config-json", default="")
    return parser.parse_args()


def resolve(spec: str):
    if ":" not in spec:
        raise ValueError("--runner must have the form module:function")
    module, function = spec.split(":", 1)
    return getattr(importlib.import_module(module), function)


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    runner = resolve(args.runner)
    base_config = json.loads(Path(args.config_json).read_text()) if args.config_json else {}
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    nfes = [int(x) for x in args.nfes.split(",") if x.strip()]
    taus = [float(x) for x in args.taus.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    rows = []
    for method in methods:
        for nfe in nfes:
            for tau in (taus if method == "brl_warmstart" else [None]):
                for seed in seeds:
                    config = dict(base_config)
                    config.update({
                        "method": method,
                        "nfe": nfe,
                        "tau": tau,
                        "seed": seed,
                        "stage1_dir": str(Path(args.stage1_dir).resolve()),
                        "oracle_selection": False,
                        "expert_action": None,
                        "validation_latent": None,
                    })
                    for _ in range(args.warmup):
                        runner(config)
                    elapsed = []
                    result_rows = []
                    for _ in range(args.updates):
                        sync_cuda()
                        start = time.perf_counter()
                        result = runner(config)
                        sync_cuda()
                        elapsed.append(time.perf_counter() - start)
                        if isinstance(result, dict):
                            result_rows.append(result)
                    result = result_rows[-1] if result_rows else {}
                    row = {
                        "method": method,
                        "nfe": nfe,
                        "tau": tau if tau is not None else "",
                        "seed": seed,
                        "updates": args.updates,
                        "latency_mean_s": statistics.mean(elapsed),
                        "latency_median_s": statistics.median(elapsed),
                        "latency_p95_s": float(torch.tensor(elapsed).quantile(.95)),
                    }
                    row.update({k: v for k, v in result.items() if k not in row})
                    rows.append(row)
    if rows:
        with (output / "latency.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (output / "speed_config.json").write_text(json.dumps({
        "methods": methods,
        "nfes": nfes,
        "taus": taus,
        "seeds": seeds,
        "warmup": args.warmup,
        "updates": args.updates,
        "cuda_synchronization": True,
    }, indent=2))
    print(json.dumps({"rows": len(rows), "output": str(output / "latency.csv")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
