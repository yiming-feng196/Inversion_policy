"""Exact row subset of an existing full cache for matched thinning controls."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from thinning import select_rows


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for x in iter(lambda:f.read(8<<20),b''):h.update(x)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--stride',type=int,default=8)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    meta=json.loads((a.source/'manifest.json').read_text())
    assert meta['status']=='complete'
    a.output.mkdir(parents=True)
    meta.update(parent_manifest=str(a.source/'manifest.json'),parent_manifest_sha256=sha(a.source/'manifest.json'),
        full_counts=dict(meta['counts']),sampling=f'Exact per-episode stride-{a.stride} plus last subset of existing full cache; no reinversion')
    meta['parent_diagnostics']=meta.pop('diagnostics',{})
    meta['diagnostics']={}
    for split in ('train','val','test'):
        path=a.source/meta['files'][split]
        assert sha(path)==meta['file_sha256'][split]
        with np.load(path,allow_pickle=False) as f: data={k:f[k] for k in f.files}
        rows=[(str(ep),int(cur),i) for i,(ep,cur) in enumerate(zip(data['episode'],data['current']))]
        ix=[r[2] for r in select_rows(rows,a.stride)]
        dest=a.output/meta['files'][split]
        np.savez_compressed(dest,**{k:v[ix] for k,v in data.items()})
        meta['counts'][split]=len(ix);meta['file_sha256'][split]=sha(dest)
        rms=data['midpoint10_rmse'][ix]
        meta['diagnostics'][split]={'paired_expert_midpoint10_executed_rmse_mean':float(rms.mean()),
            'paired_expert_midpoint10_executed_rmse_p95':float(np.quantile(rms,.95))}
    (a.output/'manifest.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps({'output':str(a.output),'full':meta['full_counts'],'thin':meta['counts']}))


if __name__=='__main__':main()
