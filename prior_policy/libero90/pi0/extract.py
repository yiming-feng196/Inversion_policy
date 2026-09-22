"""Numerical preflight and resumable full-source extraction for frozen pi0."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import h5py
import numpy as np
from protocol import CHECKPOINT, TASKS, digest, split_episodes, write_json


def indices(path, stride):
    with h5py.File(path, "r") as f:
        splits = split_episodes(list(f["data"]))
        result = {}
        for split, episodes in splits.items():
            entries = []
            for ep in episodes:
                length = len(f["data"][ep]["actions"])
                # Include the end of the task with an explicit repeat-last convention.
                # Temporal padding is recorded and is NOT counted as 50 genuine actions.
                positions = list(range(0, length, stride))
                if positions[-1] != length - 1:
                    positions.append(length - 1)
                entries.extend((ep, start, min(50, length-start)) for start in positions)
            result[split] = entries
    return splits, result


def read_rows(path, entries, prompt):
    rows = []
    with h5py.File(path, "r") as f:
        for ep, start, valid in entries:
            demo = f["data"][ep]
            obs = demo["obs"]
            action = np.asarray(demo["actions"][start:start+50], np.float32)
            if len(action) < 50:
                action = np.concatenate([action, np.repeat(action[-1:], 50-len(action), axis=0)])
            rows.append({
                "observation/image": np.ascontiguousarray(np.asarray(obs["agentview_rgb"][start])[::-1, ::-1]),
                "observation/wrist_image": np.ascontiguousarray(np.asarray(obs["eye_in_hand_rgb"][start])[::-1, ::-1]),
                "observation/state": np.concatenate([obs["ee_states"][start], obs["gripper_states"][start]]).astype(np.float32),
                "actions": action, "prompt": prompt})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=TASKS, default="microwave")
    p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--dtype", choices=('float32','bfloat16'),default='float32')
    a = p.parse_args()
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    task = TASKS[a.task]
    splits, entries = indices(task["hdf5"], a.stride)
    # The numerical pilot uses training episodes only and does not inspect rollout success.
    chosen=[]
    for i,ep in enumerate(splits['train'][:a.batch_size]):
        candidates=[x for x in entries['train'] if x[0]==ep and x[2]==50]
        chosen.append(candidates[min(len(candidates)-1, (i%3)*len(candidates)//3)])
    args = vars(a).copy()
    manifest = {"format": "pi0_full_source_cache_v1", "args": args, "task": task,
                "checkpoint": CHECKPOINT, "splits": splits, "latent_shape": [50,32],
                "executed_slice": [0,10], "active_channels": 7,
                "inversion": {"solver": "rk4", "steps": a.steps, "field_precision": a.dtype, "state_precision": "float32"},
                "decoder": f"stock-equivalent {a.dtype}-field Euler-10", "time_convention": "action t=0, noise t=1",
                "weights": "helper BF16-rounded checkpoint values; promoted exactly for FP32 computation, including activations",
                "terminal_padding": "repeat last action; valid_steps stored",
                "thinning": "per-episode stride plus final window, selected BEFORE inversion; complete 50x32 source",
                "normalization": "unchanged pretrained checkpoint; not refit on any adaptation data",
                "condition": "masked mean frozen VLM prefix output + normalized proprioceptive state",
                "counts": {k:len(v) for k,v in entries.items()},
                "code": {p.name:digest(p) for p in Path(__file__).parent.glob('*.py')},
                "hdf5_sha256": digest(task["hdf5"]),
                "checkpoint_metadata": {str(p.relative_to(Path(CHECKPOINT))):digest(p) for p in Path(CHECKPOINT).rglob('*') if p.is_file() and p.suffix=='.json'}}
    if (out/"manifest.json").exists():
        old = json.loads((out/"manifest.json").read_text())
        for key in ('args', 'hdf5_sha256', 'code', 'checkpoint_metadata'):
            if old[key] != manifest[key]:
                raise ValueError(f"Resume provenance mismatch: {key}; use a new output")
    else:
        write_json(out/"manifest.json", manifest)
    print(json.dumps({"stage":"load_pi0", "counts":manifest['counts']}), flush=True)
    from runtime import Runtime, metrics
    import jax
    import jax.numpy as jnp
    start = time.time()
    rt = Runtime(CHECKPOINT,a.dtype)
    rows = read_rows(task["hdf5"], chosen, task["prompt"])
    obs, actions, transformed, cached, condition = rt.prepare(rows)
    print(json.dumps({"stage":"loaded", "condition_shape":list(condition.shape), "seconds":time.time()-start}), flush=True)
    parity = rt.parity(obs, cached)
    # Confirm that action normalization followed by output transformation is identity.
    recovered = np.stack([rt.outputs(transformed[i]["state"], actions[i]) for i in range(len(rows))])
    transform_error = float(np.max(np.abs(recovered - np.stack([r['actions'] for r in rows]))))
    if transform_error > 1e-4:
        raise RuntimeError(f"Action transforms do not round-trip: {transform_error}")
    print(json.dumps({"stage":"parity_pass", "metrics":parity, "transform_max_abs":transform_error}), flush=True)
    zhalf = np.asarray(rt.invert(cached, actions, max(1,a.steps//2)))
    inv_start = time.time()
    z = np.asarray(rt.invert(cached, actions, a.steps))
    inv_seconds = time.time() - inv_start
    reconstructed = np.asarray(rt.high_decode(cached, jnp.asarray(z), a.steps))
    deployed = np.asarray(rt.decode(cached, jnp.asarray(z)))
    eps = jax.random.normal(jax.random.key(719), actions.shape)
    native = rt.high_decode(cached, eps, a.steps)
    zback = np.asarray(rt.invert(cached, native, a.steps))
    report = {"status":"complete", "stock_parity":parity, "transform_max_abs":transform_error,
              "source_half_full":metrics(z,zhalf), "known_source_recovery":metrics(zback,np.asarray(eps)),
              "expert_roundtrip":metrics(reconstructed,np.asarray(actions)),
              "expert_deployed_roundtrip":metrics(deployed,np.asarray(actions)),
              "batch_inversion_seconds":inv_seconds, "source_shape":list(z.shape),
              "source_abs_max":float(np.abs(z).max()), "source_std":float(z.std()),
              "condition_dim":int(condition.shape[-1]), "examples":chosen,
              "note":f"Continuous ODE approximation, NOT an exact inverse of stock Euler-10; field precision {a.dtype}."}
    write_json(out/"preflight.json", report)
    np.savez_compressed(out/"preflight.npz", expert=np.asarray(actions), source=z, source_half=zhalf,
                        high_decoded=reconstructed, deployed=deployed, condition=np.asarray(condition))
    print(json.dumps({"stage":"preflight_complete", **report}), flush=True)
    if a.preflight_only:
        return
    if (not np.isfinite(z).all() or report['expert_roundtrip']['active_rmse'] > .01
        or report['source_half_full']['active_rmse'] > .02):
        raise RuntimeError("Numerical quality gate failed; do not train on this cache")
    for split, refs in entries.items():
        folder=out/split; folder.mkdir(exist_ok=True)
        for offset in range(0,len(refs),a.batch_size):
            dest=folder/f"batch_{offset:06d}.npz"
            if dest.exists():
                continue
            selected=refs[offset:offset+a.batch_size]; valid=len(selected)
            padded=selected+[selected[-1]]*(a.batch_size-valid)
            obs, acts, _, cached, cond=rt.prepare(read_rows(task['hdf5'],padded,task['prompt']))
            src=np.asarray(rt.invert(cached,acts,a.steps))[:valid]
            if not np.isfinite(src).all():
                raise FloatingPointError(f"Nonfinite source in {split} offset {offset}")
            tmp=dest.with_suffix('.tmp.npz')
            np.savez_compressed(tmp, source=src, condition=np.asarray(cond)[:valid],
                expert=np.asarray(acts)[:valid], episode=np.array([x[0] for x in selected]),
                start=np.array([x[1] for x in selected]), valid_steps=np.array([x[2] for x in selected]))
            tmp.replace(dest)
            write_json(out/'progress.json', {'status':'running','split':split,'completed':min(offset+a.batch_size,len(refs)),
                'total':len(refs),'elapsed_seconds':time.time()-start})
            print(json.dumps({'stage':'extract','split':split,'completed':min(offset+a.batch_size,len(refs)),'total':len(refs)}),flush=True)
    write_json(out/'progress.json', {'status':'complete','counts':manifest['counts'],'elapsed_seconds':time.time()-start})


if __name__ == '__main__':
    main()
