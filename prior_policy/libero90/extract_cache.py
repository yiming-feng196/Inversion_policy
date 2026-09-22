"""Deterministic, resumable source-cache extraction from a frozen LIBERO policy."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from thinning import select_rows


def json_write(path, value):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for data in iter(lambda: f.read(8 << 20), b''):
            h.update(data)
    return h.hexdigest()


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--adapter-dir', required=True)
    p.add_argument('--repo', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--inverse-steps', type=int, default=256)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--stride', type=int, default=8)
    p.add_argument('--solver',choices=['rk4','ddim'],default='rk4')
    p.add_argument('--decoder',choices=['midpoint','ddim'],default='midpoint')
    p.add_argument('--decoder-steps',type=int,default=10)
    p.add_argument('--limit-per-split', type=int, default=0, help='Smoke only; 0 selects all windows')
    args = p.parse_args()
    sys.path.insert(0, args.adapter_dir)
    from libero_fm import Corpus, Policy, integrate
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    base = ckpt['manifest']
    for source in base['code'][1:]:
        if sha(source['path']) != source['sha256']:
            raise ValueError('Frozen author backbone changed: ' + source['path'])
    corpus = Corpus(base['args']['hdf5'], normalizer=base['normalizer'])
    assert corpus.splits == base['splits']
    assert sha(corpus.hdf5) == base['data_sha256']
    norm_hash = hashlib.sha256(json.dumps(base['normalizer'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    meta = {
        'format_version': 1, 'status': 'running', 'args': vars(args),
        'base_checkpoint': args.checkpoint, 'base_arch': base['args']['arch'],
        'task': Path(corpus.hdf5).stem, 'base_checkpoint_step': ckpt['step'],
        'action_checkpoint_sha256': sha(args.checkpoint),
        'action_normalizer_sha256': norm_hash,
        'normalizer': base['normalizer'], 'normalizer_fit_split': 'train',
        'splits': corpus.splits, 'executed_slice': [1, 9],
        'latent_shape': [16, 7], 'condition_dim': 2064,
        'inversion': {'solver': args.solver, 'steps': args.inverse_steps, 'precision': 'fp32'},
        'decoder': {'solver': args.decoder, 'steps': args.decoder_steps},
        'files': {}, 'file_sha256': {}, 'counts': {},
        'sampling': 'Per-episode chronological stride plus last window, before inversion; complete unpadded chunks; disjoint episode splits',
        'full_counts': {k: len(v) for k, v in corpus.rows.items()},
        'thinning_code_sha256': sha(Path(__file__).with_name('thinning.py')),
        'adapter_sha256': sha(Path(args.adapter_dir) / 'libero_fm.py'),
        'extractor_sha256': sha(__file__), 'data_sha256': base['data_sha256'],
        'diagnostics': {},
    }
    identity_keys = ['args', 'action_checkpoint_sha256', 'action_normalizer_sha256',
                     'data_sha256', 'adapter_sha256', 'extractor_sha256']
    if (out / 'manifest.json').exists():
        previous = json.loads((out / 'manifest.json').read_text())
        for key in identity_keys:
            if previous[key] != meta[key]:
                raise ValueError(f'Resume protocol changed: {key}; use fresh output')
        meta = previous
        if meta['status'] == 'complete':
            for split, filename in meta['files'].items():
                assert sha(out / filename) == meta['file_sha256'][split]
            print('Cache already complete and hashes verified', flush=True)
            return
    json_write(out / 'manifest.json', meta)
    model = Policy(args.repo, base['args']['arch'], base['args']['seed']).to(args.device).eval()
    model.load_state_dict(ckpt['state_dict'], strict=True)
    del ckpt
    for param in model.parameters():
        param.requires_grad_(False)
    start_time = time.monotonic()
    keys = ('condition', 'z_star', 'expert', 'episode', 'current', 'midpoint10_rmse')
    for split in ('train', 'val', 'test'):
        rows = select_rows(corpus.rows[split], args.stride)
        if args.limit_per_split:
            rows = corpus.balanced(split, min(args.limit_per_split, len(rows)), 20260919)
        shards = out / ('shards_' + split)
        shards.mkdir(exist_ok=True)
        parts = []
        for i in range(0, len(rows), args.batch_size):
            selected = rows[i:i + args.batch_size]
            path = shards / f'{i:06d}.npz'
            if not path.exists():
                images, state, action = corpus.batch(selected, args.device)
                condition = model.condition(images, state)
                z = integrate(model, condition, action.clone(), args.inverse_steps, -1, args.solver)
                decoded = integrate(model, condition, z.clone(), args.decoder_steps, 1, args.decoder)
                rms = (decoded[:, 1:9] - action[:, 1:9]).square().mean((1, 2)).sqrt()
                arrays = {'condition': condition.cpu().numpy(), 'z_star': z.cpu().numpy(),
                          'expert': action.cpu().numpy(),
                          'episode': np.asarray([r[0] for r in selected]),
                          'current': np.asarray([r[1] for r in selected], np.int64),
                          'midpoint10_rmse': rms.cpu().numpy()}
                if not all(np.isfinite(arrays[k]).all() for k in ('condition', 'z_star', 'expert', 'midpoint10_rmse')):
                    raise FloatingPointError(f'Nonfinite inversion in {split}:{i}')
                tmp = path.with_suffix('.tmp.npz')
                np.savez_compressed(tmp, **arrays)
                tmp.replace(path)
            with np.load(path, allow_pickle=False) as f:
                arrays = {k: f[k] for k in keys}
            assert arrays['episode'].tolist() == [r[0] for r in selected]
            assert arrays['current'].tolist() == [r[1] for r in selected]
            parts.append(arrays)
            meta['progress'] = {'split': split, 'completed': i + len(selected), 'total': len(rows),
                                'elapsed_seconds': time.monotonic() - start_time}
            json_write(out / 'manifest.json', meta)
            print(json.dumps(meta['progress']), flush=True)
        merged = {k: np.concatenate([part[k] for part in parts]) for k in keys}
        filename = split + '.npz'
        tmp = out / (split + '.tmp.npz')
        np.savez_compressed(tmp, **merged)
        tmp.replace(out / filename)
        meta['files'][split] = filename
        meta['file_sha256'][split] = sha(out / filename)
        meta['counts'][split] = len(rows)
        rms = merged['midpoint10_rmse']
        meta['diagnostics'][split] = {'paired_expert_midpoint10_executed_rmse_mean': float(rms.mean()),
                                     'paired_expert_midpoint10_executed_rmse_p95': float(np.quantile(rms, .95))}
        json_write(out / 'manifest.json', meta)
    meta['status'] = 'complete'
    json_write(out / 'manifest.json', meta)
    print('Extraction complete', flush=True)


if __name__ == '__main__':
    main()
