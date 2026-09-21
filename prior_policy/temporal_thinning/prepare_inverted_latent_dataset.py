"""Offline inversion of every original sampler chunk, with resumable shards."""
import json
from pathlib import Path

import numpy as np
import torch
import zarr
from omegaconf import OmegaConf

from flow_latent_predictor_common import (
    atomic_save, common_parser, forward_flow, load_flow,
    prepare_predictor_context, seed_all, sha256, write_json,
)


def parse_dims(value: str) -> tuple[int, ...]:
    if not value:
        return ()
    dims = tuple(int(item.strip()) for item in value.split(',') if item.strip())
    if len(set(dims)) != len(dims) or any(dim < 0 for dim in dims):
        raise ValueError(f'Expected unique non-negative dimensions, got {value!r}')
    return dims


def choose_sampler_indices(indices, ends, action, stride, gripper_dims,
                           gripper_threshold, velocity_dims, velocity_topk):
    """Select non-overlapping-ish chunks before feature extraction/inversion.

    Every episode contributes uniformly spaced chunks plus its final chunk. Event
    anchors are matched to the chunk whose first action is the detected transition,
    preserving grasp/release and rapid joint-motion segments without running Flow.
    """
    if stride < 1:
        raise ValueError('--sample-stride must be at least one')
    if gripper_threshold < 0 or velocity_topk < 0:
        raise ValueError('event thresholds/counts must be non-negative')
    action_dim = int(action.shape[1])
    requested_dims = set(gripper_dims) | set(velocity_dims)
    if any(dim >= action_dim for dim in requested_dims):
        raise ValueError(f'Event dimensions {sorted(requested_dims)} exceed action dimension {action_dim}')

    by_episode = {}
    for sampler_index, (begin, _, sample_start, _) in enumerate(indices):
        episode = int(np.searchsorted(ends, begin, side='right'))
        action_start = int(begin - sample_start)
        by_episode.setdefault(episode, []).append((action_start, sampler_index))

    selected = set()
    report = dict(candidates=int(len(indices)), stride_selected=0,
                  gripper_events=0, velocity_events=0, selected=0)
    for episode, rows in by_episode.items():
        rows.sort()
        stride_rows = rows[::stride]
        selected.update(sampler_index for _, sampler_index in stride_rows)
        selected.add(rows[-1][1])
        report['stride_selected'] += len(stride_rows) + (rows[-1] not in stride_rows)

        if not gripper_threshold and not velocity_topk:
            continue
        episode_start = 0 if episode == 0 else int(ends[episode - 1])
        episode_end = int(ends[episode])
        episode_action = np.asarray(action[episode_start:episode_end])
        # Early padded chunks can share an action start; either representative
        # contains the same padded action prefix, so retaining one is sufficient.
        sampler_for_action_start = {start: sampler_index for start, sampler_index in rows}
        event_positions = []
        if gripper_threshold:
            if not gripper_dims:
                raise ValueError('--event-gripper-dims is required with a gripper threshold')
            gripper_delta = np.abs(np.diff(episode_action[:, gripper_dims], axis=0)).max(axis=1)
            gripper_positions = np.flatnonzero(gripper_delta > gripper_threshold) + episode_start + 1
            report['gripper_events'] += len(gripper_positions)
            event_positions.extend(gripper_positions.tolist())
        if velocity_topk:
            dims = velocity_dims or tuple(range(action_dim))
            velocity = np.linalg.norm(np.diff(episode_action[:, dims], axis=0), axis=1)
            count = min(velocity_topk, len(velocity))
            velocity_positions = np.argsort(velocity)[-count:] + episode_start + 1
            report['velocity_events'] += len(velocity_positions)
            event_positions.extend(velocity_positions.tolist())
        selected.update(sampler_for_action_start[position] for position in event_positions
                        if position in sampler_for_action_start)
    report['selected'] = len(selected)
    return np.asarray(sorted(selected), dtype=int), report


def main():
    p = common_parser(__doc__)
    p.add_argument('--zarr', required=True)
    p.add_argument('--history', type=int, default=12)
    p.add_argument('--max-per-split', type=int, default=0)
    p.add_argument('--sample-stride', type=int, default=1,
                   help='Keep every Nth sampler chunk within each episode before inversion (default: 1).')
    p.add_argument('--event-gripper-dims', default='',
                   help='Comma-separated action dimensions used for gripper-transition anchors.')
    p.add_argument('--event-gripper-threshold', type=float, default=0.0,
                   help='Retain chunks beginning at gripper changes larger than this threshold (default: disabled).')
    p.add_argument('--event-velocity-dims', default='',
                   help='Comma-separated action dimensions used for top velocity anchors; empty means all dimensions.')
    p.add_argument('--event-velocity-topk', type=int, default=0,
                   help='Retain this many largest action-change anchors per episode (default: disabled).')
    p.add_argument('--reconstruction-threshold', type=float, default=0.1)
    p.add_argument('--save-intermediate-states', action='store_true',
                   help='Save reverse Flow states x_tau at tau=.25,.50,.75 for warm-start experiments')
    p.add_argument('--val-ratio', type=float, default=0.2,
                   help='Episode-level validation ratio used when creating a new cache')
    a = p.parse_args()
    seed_all(42)
    policy, matcher, cfg = load_flow(a)
    from roboverse_learn.il.utils.sampler import create_indices, downsample_mask
    root = zarr.open(a.zarr, mode='r')
    ends = np.asarray(root['meta']['episode_ends'][:])
    dc = cfg.dataset_config
    if not 0 < a.val_ratio < 1:
        raise ValueError('--val-ratio must be between 0 and 1')
    # Split episodes before creating any overlapping sampler chunks.
    rng = np.random.default_rng(int(dc.seed))
    episode_order = rng.permutation(len(ends))
    n_val = max(1, int(round(len(ends)*a.val_ratio)))
    val = np.zeros(len(ends), dtype=bool)
    val[episode_order[-n_val:]] = True
    train = downsample_mask(~val, dc.max_train_episodes, dc.seed)
    assert not np.intersect1d(np.flatnonzero(train), np.flatnonzero(val)).size
    assert len(np.flatnonzero(val)) > 1, 'Validation must contain multiple episodes'
    nobs, horizon = int(policy.n_obs_steps), int(policy.horizon)
    if a.history < nobs:
        raise ValueError('Predictor history must include original condition history')
    entries = []
    selection_reports = {}
    gripper_dims = parse_dims(a.event_gripper_dims)
    velocity_dims = parse_dims(a.event_velocity_dims)
    for split, mask in enumerate((train, val)):
        indices = create_indices(ends, horizon, mask, int(dc.pad_before), int(dc.pad_after))
        chosen, selection_report = choose_sampler_indices(
            indices, ends, root['data']['action'], a.sample_stride, gripper_dims,
            a.event_gripper_threshold, velocity_dims, a.event_velocity_topk)
        if a.max_per_split:
            chosen = chosen[np.linspace(0, len(chosen)-1, min(a.max_per_split, len(chosen)), dtype=int)]
        selection_report['after_max_per_split'] = int(len(chosen))
        selection_reports['train' if split == 0 else 'val'] = selection_report
        entries.extend((split, int(i), indices[i]) for i in chosen)
    out = Path(a.cache)
    out.mkdir(parents=True, exist_ok=True)
    signature = dict(checkpoint_sha256=sha256(a.checkpoint), dataset=str(Path(a.zarr).resolve()),
                     dataset_config=OmegaConf.to_container(dc, resolve=True), history=a.history,
                     forward_steps=a.forward_steps, reverse_steps=a.reverse_steps,
                     max_per_split=a.max_per_split, batch_size=a.batch_size,
                     sampling=dict(sample_stride=a.sample_stride, event_gripper_dims=gripper_dims,
                                   event_gripper_threshold=a.event_gripper_threshold,
                                   event_velocity_dims=velocity_dims, event_velocity_topk=a.event_velocity_topk,
                                   reports=selection_reports),
                     reconstruction_threshold=a.reconstruction_threshold,
                     save_intermediate_states=a.save_intermediate_states,
                     val_ratio=a.val_ratio,
                     train_episodes=np.flatnonzero(train).tolist(), val_episodes=np.flatnonzero(val).tolist(),
                     solver='native Euler sample/reverse_sample, t=0 to 1 / 1 to 0',
                     preprocessing='RGB /255 then checkpoint normalizer; original sampler padding',
                     model_state='ema_model' if cfg.train_config.training_params.use_ema else 'model')
    config_path = out/'cache_config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != signature:
        raise ValueError('Cache configuration mismatch; use a new directory')
    write_json(signature, config_path)
    # Features for each unique physical observation are independently resumable.
    feature_file = out/'observation_features.pt'
    if feature_file.exists():
        features = torch.load(feature_file, weights_only=True)
    else:
        needed = set()
        for _, _, (bs, be, ss, se) in entries:
            ep = int(np.searchsorted(ends, bs, side='right'))
            start = 0 if ep == 0 else int(ends[ep-1])
            current = int(bs-ss+nobs-1)
            needed.update(np.clip(np.arange(current-a.history+1, current+1), start, ends[ep]-1).tolist())
        feature_shards = []
        for begin in range(0, len(needed), a.batch_size):
            positions = sorted(needed)[begin:begin+a.batch_size]
            fp = out/f'features_{begin:08d}.pt'
            if not fp.exists():
                rgb = torch.tensor(np.stack([root['data']['head_camera'][i] for i in positions]),
                                   device=a.device).float()
                if rgb.ndim != 4:
                    raise ValueError(f'Expected batched RGB observations, got {tuple(rgb.shape)}')
                if rgb.shape[1] == 3:
                    rgb_chw = rgb
                elif rgb.shape[-1] == 3:
                    rgb_chw = rgb.permute(0, 3, 1, 2)
                else:
                    raise ValueError(f'Expected RGB channel dimension, got {tuple(rgb.shape)}')
                state = torch.tensor(np.stack([root['data']['state'][i] for i in positions]),
                                     device=a.device).float()
                obs = {'head_cam': rgb_chw[:,None]/255,
                       'agent_pos': state[:,None]}
                f = prepare_predictor_context(policy, obs)[:,0].cpu()
                atomic_save({'positions': torch.tensor(positions), 'features': f}, fp)
            feature_shards.append(fp)
            if begin % (a.batch_size*50) == 0:
                print(f'features {begin}/{len(needed)}', flush=True)
        parts = [torch.load(fp, weights_only=True) for fp in feature_shards]
        features = torch.zeros((int(ends[-1]), parts[0]['features'].shape[-1]))
        for part in parts:
            features[part['positions']] = part['features']
        atomic_save(features, feature_file)
    names, reconstruction = [], []
    for begin in range(0, len(entries), a.batch_size):
        name = f'inverted_{begin:08d}.pt'
        fp = out/name
        names.append(name)
        if not fp.exists():
            rows = []
            for offset, (split, sample, (bs, be, ss, se)) in enumerate(entries[begin:begin+a.batch_size]):
                ep = int(np.searchsorted(ends, bs, side='right'))
                start = 0 if ep == 0 else int(ends[ep-1])
                current = int(bs-ss+nobs-1)
                positions = np.clip(np.arange(current-a.history+1,current+1),start,ends[ep]-1)
                assert positions.max() <= current
                assert positions.min() >= start
                assert np.array_equal(positions[-nobs:], np.clip(np.arange(current-nobs+1,current+1),start,ends[ep]-1))
                action_positions = np.clip(np.arange(bs-ss, bs-ss+horizon),bs,be-1)
                action = np.stack([root['data']['action'][int(i)] for i in action_positions])
                row = dict(sample_index=torch.tensor(begin+offset), sampler_index=torch.tensor(sample),
                                 condition_id=torch.tensor(current), observation_indices=torch.tensor(positions), episode=torch.tensor(ep), split=torch.tensor(split),
                                 proprio=torch.tensor(root['data']['state'][current]).float(),
                                 expert_raw=torch.tensor(action).float(), context=features[positions])
                rows.append(row)
            d = {k: torch.stack([r[k] for r in rows]) for k in rows[0]}
            c = d['context'][:,-nobs:].flatten(1).to(a.device)
            with torch.no_grad():
                expert = policy.normalizer['action'].normalize(d['expert_raw'].to(a.device))
                if a.save_intermediate_states:
                    z, (_, reverse_trace) = matcher.reverse_sample(
                        policy.model, start=expert, num_steps=a.reverse_steps,
                        return_traces=True, global_cond=c)
                else:
                    z = matcher.reverse_sample(policy.model, start=expert, num_steps=a.reverse_steps, global_cond=c)
                reconstructed = forward_flow(policy, matcher, z, c, a.forward_steps)
            d.update(condition=c.cpu(), expert=expert.cpu(), z_star=z.cpu(),
                     z_norm=z.flatten(1).norm(dim=1).cpu(),
                     reconstruction_rmse=(reconstructed-expert).square().flatten(1).mean(1).sqrt().cpu())
            if a.save_intermediate_states:
                # reverse_trace[k] is x(t=1-k/reverse_steps), so map the
                # requested forward-flow times back to reverse-trace indices.
                for tau, key in ((0.25, 'x_tau_025'), (0.50, 'x_tau_050'), (0.75, 'x_tau_075')):
                    trace_index = int(round((1.0 - tau) * a.reverse_steps))
                    d[key] = reverse_trace[trace_index].float()
            if not all(torch.isfinite(v).all() for v in d.values()):
                atomic_save(d, out/f'nonfinite_{begin}.pt')
                raise ValueError('Nonfinite inversion retained for inspection; cannot train')
            atomic_save(d, fp)
        d = torch.load(fp, weights_only=True)
        reconstruction.extend(d['reconstruction_rmse'].tolist())
        print(f'inversion {begin+len(d["expert"])}/{len(entries)} RMSE={d["reconstruction_rmse"].mean():.6g}', flush=True)
    errors = np.array(reconstruction)
    write_json(dict(**signature, shards=names, samples=len(entries),
                    context_shape=list(d['context'].shape[1:]), latent_shape=list(d['z_star'].shape[1:]),
                    reconstruction_rmse_mean=float(errors.mean()), reconstruction_rmse_max=float(errors.max()),
                    anomalous_sample_indices=np.flatnonzero(errors>a.reconstruction_threshold).tolist(),
                    anomalies_retained=True), out/'manifest.json')


if __name__ == '__main__':
    main()
