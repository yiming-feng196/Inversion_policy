"""Train Gaussian and Flow priors from identical full-source pi0 supervision."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from protocol import digest, write_json
from prior_models import make_model


def load_split(root, split):
    paths=sorted((root/split).glob('batch_*.npz'))
    if not paths:
        raise ValueError(f'Missing split {split}')
    data={k:[] for k in ('condition','source','episode','valid_steps')}
    for path in paths:
        with np.load(path,allow_pickle=False) as f:
            for key in data:
                data[key].append(f[key])
    return {k:np.concatenate(v) for k,v in data.items()}


@torch.no_grad()
def validate(model, condition, target):
    generator=torch.Generator(device=condition.device).manual_seed(1109)
    first=model.sample(condition,steps=16,generator=generator)
    second=model.sample(condition,steps=16,generator=generator)
    norm=lambda x:x.flatten(1).square().mean(1).sqrt()
    # Single-target conditional energy-score estimate. This is a source-space
    # diagnostic, NOT action performance or an estimate of conditional W2.
    return {'source_sample_rmse':float(norm(first-target).mean()),
            'source_energy_score':float((.5*(norm(first-target)+norm(second-target))-.5*norm(first-second)).mean()),
            'source_pairwise_rms':float(norm(first-second).mean())}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cache',required=True);p.add_argument('--output',required=True)
    p.add_argument('--method',choices=('cgaussian','cflow','mlp'),required=True)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--steps',type=int,default=5000)
    p.add_argument('--batch-size',type=int,default=32);p.add_argument('--device',default='cuda')
    a=p.parse_args();root=Path(a.cache);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'final.pt').exists():
        raise FileExistsError('Completed output already exists')
    if json.loads((root/'progress.json').read_text())['status']!='complete':
        raise ValueError('Cache extraction is incomplete')
    manifest=json.loads((root/'manifest.json').read_text())
    train=load_split(root,'train');val=load_split(root,'val')
    if set(train['episode']) & set(val['episode']):
        raise ValueError('Train/validation episode leakage')
    for split,data in [('train',train),('val',val)]:
        if set(data['episode'])!=set(manifest['splits'][split]) or len(data['source'])!=manifest['counts'][split]:
            raise ValueError(f'Cache membership/count mismatch: {split}')
        if data['source'].shape[1:]!=(50,32) or not np.isfinite(data['source']).all():
            raise ValueError('Require finite complete 50x32 sources')
    torch.set_num_threads(4);torch.manual_seed(a.seed);np.random.seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if a.device.startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(.40)
    c=torch.from_numpy(train['condition']).float().to(a.device)
    z=torch.from_numpy(train['source']).float().to(a.device)
    vc=torch.from_numpy(val['condition']).float().to(a.device)
    vz=torch.from_numpy(val['source']).float().to(a.device)
    mean=c.mean(0);std=c.std(0,unbiased=False).clamp_min(.001)
    model=make_model(a.method,mean.cpu(),std.cpu()).to(a.device)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-6)
    metadata={'format':'pi0_full_source_prior_v1','method':a.method,'config':model.config,'args':vars(a),
        'cache_manifest_sha256':digest(root/'manifest.json'),'parameters':sum(p.numel() for p in model.parameters()),
        'condition_stats_fit':'train only','source_normalization':'none',
        'model_selection':'fixed final optimizer step, not test or rollout performance',
        'source_shape':[50,32],'prior_flow_sampling':'Euler-16; deterministic for fixed condition and epsilon',
        'code':{p.name:digest(p) for p in Path(__file__).parent.glob('*.py')}}
    write_json(out/'manifest.json',metadata)
    start=time.time();curve=[]
    for step in range(1,a.steps+1):
        model.train()
        ids=torch.randint(len(c),(a.batch_size,),device=c.device)
        noise=torch.randn_like(z[ids]);times=torch.rand(a.batch_size,device=c.device)
        loss=model.objective(c[ids],z[ids],noise,times)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite training loss at step {step}')
        lr=1e-4*min(step/200,1)*.5*(1+np.cos(np.pi*(step-1)/a.steps))
        for group in opt.param_groups:group['lr']=lr
        opt.zero_grad(set_to_none=True);loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(grad):raise FloatingPointError('Nonfinite gradient')
        opt.step()
        if step==1 or step%500==0 or step==a.steps:
            model.eval()
            stats=validate(model,vc[:64],vz[:64])
            row={'step':step,'loss':float(loss.detach()),'seconds':time.time()-start,**stats}
            curve.append(row);write_json(out/'curve.json',curve)
            write_json(out/'progress.json',{'status':'running','step':step,'total':a.steps,**stats})
            print(json.dumps(row),flush=True)
    payload={**metadata,'step':a.steps,'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'validation':curve[-1]}
    tmp=out/'final.tmp.pt';torch.save(payload,tmp);tmp.replace(out/'final.pt')
    write_json(out/'progress.json',{'status':'complete','step':a.steps,'elapsed_seconds':time.time()-start})


if __name__=='__main__':main()
