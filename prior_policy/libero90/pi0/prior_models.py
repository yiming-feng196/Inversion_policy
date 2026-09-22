"""Full pi0 source priors. Reuses the audited source-model implementation."""
from __future__ import annotations

import sys
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F

PREVIOUS = Path('/data/jhr/q2_sampler_comparison_20260919/training')
sys.path.append(str(PREVIOUS))
from source_models import SourceSampler, build_config

AUTHOR_REPO='/data/jhr/MomentVLA_inversion_osd'


class LengthAdapter(nn.Module):
    """Internal convolution padding only; never changes the 50x32 source target."""
    def __init__(self, core):
        super().__init__()
        self.core=core

    def forward(self, x, t, global_cond=None):
        n=x.shape[1]
        pad=(-n)%4
        if pad:
            x=torch.cat([x,x[:,-1:].expand(-1,pad,-1)],dim=1)
        return self.core(x,t,global_cond=global_cond)[:,:n]


def make_model(method, cond_mean, cond_std, config=None):
    if config is None:
        config=build_config(method,len(cond_mean),(50,32),AUTHOR_REPO, down_dims=(128,256,512))
    model=SourceSampler(config,cond_mean,cond_std,repo=AUTHOR_REPO)
    if method in ('cflow','uflow'):
        model.velocity=LengthAdapter(model.velocity)
    return model


def load_model(path, device='cpu'):
    payload=torch.load(path,map_location='cpu',weights_only=False)
    if payload['format']!='pi0_full_source_prior_v1':
        raise ValueError('Not a full-source pi0 prior checkpoint')
    state=payload['state_dict']
    model=make_model(payload['method'],state['condition_mean'],state['condition_std'],payload['config'])
    model.load_state_dict(state,strict=True)
    return model.to(device).eval().requires_grad_(False),payload
