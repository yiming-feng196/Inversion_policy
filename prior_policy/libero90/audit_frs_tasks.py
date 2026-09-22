"""Reconcile the released ECoT successful-demo index with official task names.

Index evidence is saved separately: this does NOT relabel our raw-HDF5 training
data as the FRS success-filtered TFDS release.
"""
import json
from pathlib import Path
import requests
import subprocess
import hashlib

ROOT=Path(__file__).resolve().parent
HOST='https://hf-mirror.com'
REPO='Embodied-CoT/embodied_features_and_demos_libero'


def get(url):
    r=requests.get(url,timeout=60);r.raise_for_status();return r.json()


def main():
    out=ROOT/'frs_audit';out.mkdir(exist_ok=True)
    rev=get(f'{HOST}/api/datasets/{REPO}')['sha']
    tree=get(f'{HOST}/api/datasets/{REPO}/tree/{rev}?recursive=true')
    matches=[x for x in tree if x['path']=='libero_reasonings.json']
    if len(matches)!=1:raise ValueError(f'Expected one reasoning index; found {matches}')
    item=matches[0];dest=out/'libero_reasonings.json'
    if not dest.exists():
        tmp=dest.with_suffix('.part')
        subprocess.run(['curl','--fail','--location','--retry','3','--continue-at','-','--output',str(tmp),
            f'{HOST}/datasets/{REPO}/resolve/{rev}/{item["path"]}?download=true'],check=True)
        if tmp.stat().st_size!=item['size']:raise ValueError('Truncated index')
        tmp.replace(dest)
    h=hashlib.sha256(dest.read_bytes()).hexdigest()
    if 'lfs' in item and h!=item['lfs']['oid']:raise ValueError('Index SHA256 mismatch')
    data=json.loads(dest.read_text())
    # Inspect structure rather than assuming how the author serialized its index.
    report={'repo':REPO,'revision':rev,'index_sha256':h,'type':type(data).__name__,
        'top_level_entries':len(data),'example_keys':list(data)[:5],
        'sample_value_type':type(next(iter(data.values()))).__name__ if isinstance(data,dict) else None,
        'scope':'Author successful-demo reasoning index. No training-data filtering has been applied.'}
    registry=json.loads((ROOT/'tasks.json').read_text())
    missing=[t for t in registry['tasks'] if t['task_name']+'_demo.hdf5' not in data]
    unknown=set(data)-{t['task_name']+'_demo.hdf5' for t in registry['tasks']}
    report.update(successful_episode_index_count=sum(len(v) for v in data.values()),
        missing_tasks=missing,unknown_dataset_tasks=sorted(unknown),
        matched_task_ids=[t['id'] for t in registry['tasks'] if t['task_name']+'_demo.hdf5' in data])
    if len(data)!=89 or report['successful_episode_index_count']!=3917 or len(missing)!=1 or unknown:
        raise ValueError('Index does not match the released 89-task / 3917-demo description')
    registry['frs_89_excluded_task']=missing[0]
    registry['frs_alignment']='89 task names verified against author successful-demo index (3917 entries). Main evaluation keeps all 90 raw-HDF5 tasks; shared-task mean is secondary and not a matched data/model reproduction.'
    registry['frs_index_sha256']=h
    tmp=ROOT/'tasks.tmp.json';tmp.write_text(json.dumps(registry,indent=2));tmp.replace(ROOT/'tasks.json')
    episode_ids={str(t['id']):sorted(map(int,data.get(t['task_name']+'_demo.hdf5',{}))) for t in registry['tasks']}
    (out/'successful_demo_ids.json').write_text(json.dumps(episode_ids,indent=2))
    (out/'index_structure.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
