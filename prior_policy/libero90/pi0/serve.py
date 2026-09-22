"""Loopback-only pi0 service with matched full-sequence source priors."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from protocol import CHECKPOINT, digest, source_noise, write_json
from prior_models import load_model
from runtime import Runtime
from openpi.serving.websocket_policy_server import WebsocketPolicyServer


class PriorPolicy:
    def __init__(self, root, seed, ready,base_only=False):
        torch.set_num_threads(4)
        self.rt=Runtime(CHECKPOINT)
        self.priors={};self.paths={};self.seed=seed;self.ready=ready
        for method in (() if base_only else ('cgaussian','cflow')):
            path=root/f'{method}_s{seed}'/'final.pt'
            self.priors[method],payload=load_model(path)
            self.paths[method]={'checkpoint':str(path),'sha256':digest(path),'cache_manifest_sha256':payload['cache_manifest_sha256']}
        if not base_only and len({x['cache_manifest_sha256'] for x in self.paths.values()})!=1:
            raise ValueError('Priors trained from different source caches')
        self.checked=False

    def infer(self, obs):
        method=str(obs.pop('_method'))
        noise=source_noise(int(obs.pop('_noise_seed')),int(obs.pop('_initial_state')),int(obs.pop('_query_index')))
        started=time.perf_counter()
        observation,_,transformed,cached,cond=self.rt.prepare([obs])
        condition=np.asarray(cond)
        if not self.checked:
            parity=self.rt.parity(observation,cached)
            self.checked=True
            write_json(self.ready,{'status':'decoder_parity_passed','priors':self.paths,'parity':parity})
        source_start=time.perf_counter()
        if method=='gaussian':
            z=noise
        elif method in self.priors:
            with torch.inference_mode():
                z=self.priors[method].sample(torch.from_numpy(condition.copy()),noise=torch.from_numpy(noise),steps=16).numpy()
        else:
            raise ValueError(f'Unexpected method: {method}')
        source_ms=(time.perf_counter()-source_start)*1000
        import jax.numpy as jnp
        a=np.asarray(self.rt.decode(cached,jnp.asarray(z)))[0]
        output=self.rt.outputs(transformed[0]['state'],a)
        if output.shape!=(50,7) or not np.isfinite(output).all():
            raise ValueError('Invalid decoded actions')
        return {'actions':output,'source_rms':float(np.sqrt(np.mean(z*z))),
                'source_ms':source_ms,'total_ms':(time.perf_counter()-started)*1000,
                'condition_dim':int(condition.shape[-1])}


def main():
    p=argparse.ArgumentParser();p.add_argument('--models');p.add_argument('--seed',type=int,default=0)
    p.add_argument('--port',type=int,default=8147);p.add_argument('--ready',required=True)
    p.add_argument('--base-only',action='store_true')
    a=p.parse_args()
    if not a.base_only and not a.models:p.error('--models is required unless --base-only')
    policy=PriorPolicy(Path(a.models) if a.models else None,a.seed,Path(a.ready),a.base_only)
    metadata={'experiment':'pi0_full_source_prior_v1','training_seed':a.seed,'models':policy.paths,
              'decoder':'stock-equivalent FP32-field Euler-10; BF16-rounded weights retained','source_shape':[50,32],
              'source_inference_device':'CPU; pi0 runs on GPU',
              'note':'Report end-to-end latency including CPU prior inference; not peak-GPU latency.'}
    write_json(a.ready,{'status':'loaded','metadata':metadata})
    print(json.dumps({'stage':'server_ready','port':a.port}),flush=True)
    WebsocketPolicyServer(policy,host='127.0.0.1',port=a.port,metadata=metadata).serve_forever()


if __name__=='__main__':main()
