"""DP/DDIM adapter with author UNet backbone and official diffusers schedulers.

Not a claimed official DP LIBERO checkpoint. No FM ODE is used to invert DP.
DDIM inversion is approximate; denoising includes the original DP clipping.
"""
import importlib.util
from pathlib import Path
import sys
import torch

BASE=Path('/data/jhr/q1_libero_crossarch_20260915/code_v2_20260916/libero_fm.py')
spec=importlib.util.spec_from_file_location('_original_libero_fm',BASE)
original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'deps'))
from diffusers import DDPMScheduler, DDIMScheduler, DDIMInverseScheduler

Corpus=original.Corpus
digest_file=original.digest_file
write_json=original.write_json
CONFIG=dict(num_train_timesteps=100,beta_start=.0001,beta_end=.02,
    beta_schedule='squaredcos_cap_v2',clip_sample=True,prediction_type='epsilon')


class Policy(original.Policy):
    def __init__(self,repo,architecture,seed=0,obs_steps=2):
        if architecture!='dp_unet':raise ValueError('This adapter is for DP-UNet only')
        super().__init__(repo,'unet',seed,obs_steps)


def integrate(policy,condition,x,steps,direction=1,solver='ddim'):
    if solver!='ddim' or not 1<=steps<=100:raise ValueError('DP requires DDIM with 1..100 steps')
    if direction not in (-1,1):raise ValueError('Invalid direction')
    cls=DDIMScheduler if direction==1 else DDIMInverseScheduler
    scheduler=cls(**CONFIG,set_alpha_to_one=True,timestep_spacing='leading')
    scheduler.set_timesteps(steps,device=x.device)
    for timestep in scheduler.timesteps:
        pred=policy.field(x,timestep,condition)
        x=scheduler.step(pred,timestep,x).prev_sample
    return x


def objective(model,batch,generator):
    images,state,action=batch
    noise=torch.randn(action.shape,generator=generator,device=action.device)
    times=torch.randint(100,(len(action),),generator=generator,device=action.device)
    scheduler=DDPMScheduler(**CONFIG,variance_type='fixed_small')
    noisy=scheduler.add_noise(action,noise,times)
    return (model(images,state,noisy,times)-noise).square().mean()


def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--repo',required=True);p.add_argument('--hdf5',required=True);p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--steps',type=int,default=10000)
    p.add_argument('--device',default='cuda:0');p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--eval-every',type=int,default=500)
    p.add_argument('--validation-samples',type=int,default=256)
    a=p.parse_args();a.arch='dp_unet';a.diffusion_protocol=CONFIG
    a.decoder='deterministic DDIM-100 eta=0; clip_sample=True; official diffusers 0.35.1'
    a.inversion='approximate DDIMInverseScheduler-100; not claimed exactly invertible'
    old_provenance=original.provenance
    original.provenance=lambda repo:old_provenance(repo)+[{'path':str(Path(__file__).resolve()),'sha256':digest_file(__file__)}]
    original.Policy=Policy;original.fm_loss=objective
    old_write=original.write_json
    def write_with_dp_protocol(path,value):
        if isinstance(value,dict) and 'normalizer_fit_split' in value:
            value['protocol']='FP32 DDPM epsilon prediction with cosine schedule; DDIM-100 eta=0 deployment; author UNet and our LIBERO encoder adapter'
            value['scope']='Custom matched-budget DP-DDIM baseline; not an official released LIBERO DP checkpoint; no EMA; legacy fm_loss keys mean epsilon MSE'
        old_write(path,value)
    original.write_json=write_with_dp_protocol
    # Existing trainer supplies identical data splits, normalization and budgets.
    # Legacy *_fm curve keys denote epsilon-prediction MSE for this adapter.
    original.train(a)


if __name__=='__main__':main()
