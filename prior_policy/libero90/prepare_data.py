"""Pin official HDF5 metadata, inventory all 90 tasks, resume verified downloads."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import time
import requests

ROOT = Path(__file__).resolve().parent
DATA = Path('/data/jhr/LIBERO/datasets')
REPO = 'yifengzhu-hf/LIBERO-datasets'
HOST = 'https://hf-mirror.com'


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def get(url):
    r=requests.get(url,timeout=60)
    r.raise_for_status()
    return r.json()


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(8 << 20), b''):
            h.update(part)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(); p.add_argument('--workers', type=int, default=2)
    p.add_argument('--task-ids',default=None,help='Only download these task IDs; registry remains the full official suite')
    p.add_argument('--inventory-only', action='store_true'); a = p.parse_args()
    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()['libero_90']()
    rows = []
    for i in range(suite.n_tasks):
        task = suite.get_task(i)
        rows.append(dict(id=i, key=f't{i:03d}', task_name=task.name, prompt=task.language,
            suite='libero_90', hdf5=str(DATA/'libero_90'/f'{task.name}_demo.hdf5'), max_steps=400))
    assert len(rows) == 90 and len({x['task_name'] for x in rows}) == 90
    registry={'suite':'libero_90','tasks':rows,'frs_89_excluded_task':None,
        'frs_alignment':'Full suite selected; success-filtered 89-task membership still unverified. No task excluded.'}
    if (ROOT/'tasks.json').exists():
        previous=json.loads((ROOT/'tasks.json').read_text())
        if previous['tasks']!=rows:raise ValueError('Official task registry changed')
        registry.update({k:v for k,v in previous.items() if k.startswith('frs_')})
    save(ROOT/'tasks.json',registry)
    if a.task_ids:
        selected=[int(x) for x in a.task_ids.split(',')]
        if len(selected)!=len(set(selected)) or not set(selected)<=set(r['id'] for r in rows):
            raise ValueError('Invalid selected task IDs')
        rows=sorted([r for r in rows if r['id'] in selected],key=lambda r:selected.index(r['id']))
    metadata = ROOT/'dataset_revision.json'
    if metadata.exists():
        pinned = json.loads(metadata.read_text())
    else:
        revision = get(f'{HOST}/api/datasets/{REPO}')['sha']
        files = get(f'{HOST}/api/datasets/{REPO}/tree/{revision}/libero_90?recursive=true&expand=false')
        pinned = {'repo':REPO, 'revision':revision, 'files':files}
        save(metadata, pinned)
    by_name = {Path(x['path']).name:x for x in pinned['files'] if x['type']=='file'}
    assert all(Path(x['hdf5']).name in by_name for x in rows)
    print(json.dumps({'tasks':len(rows),'task_ids':[r['id'] for r in rows], 'revision':pinned['revision'],
        'total_bytes':sum(by_name[Path(r['hdf5']).name]['size'] for r in rows)}), flush=True)
    if a.inventory_only: return

    def download(row):
        dest = Path(row['hdf5']); item = by_name[dest.name]
        expected = item['lfs']['oid']; marker = ROOT/'data_status'/f"{row['key']}.json"
        if dest.exists():
            if dest.stat().st_size != item['size'] or sha(dest) != expected:
                raise ValueError(f'Existing file differs from pinned official release; will not overwrite {dest}')
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            part = dest.with_suffix('.download.part')
            url = f"{HOST}/datasets/{REPO}/resolve/{pinned['revision']}/{item['path']}?download=true"
            log = ROOT/'logs'/f"download_{row['key']}.log"
            # A truncated CDN transfer (curl 18) is not retried by default.
            # Reopen curl each time so -C reads the CURRENT partial-file size.
            for attempt in range(10):
                if part.exists() and part.stat().st_size==item['size']:
                    break
                with log.open('a') as f:
                    result=subprocess.run(['curl','--fail','--location','--retry','3','--retry-all-errors',
                        '--retry-delay','5','--connect-timeout','30','--speed-limit','1024','--speed-time','120',
                        '--continue-at','-','--output',str(part),url],stdout=f,stderr=f)
                if result.returncode==0:break
                save(marker,{'status':'retrying_download','attempt':attempt+1,'curl_exit':result.returncode,
                    'partial_bytes':part.stat().st_size if part.exists() else 0,'task':row})
                time.sleep(10)
            else:raise RuntimeError(f'Network retries exhausted: {dest}; partial file preserved')
            if part.stat().st_size != item['size'] or sha(part) != expected:
                raise ValueError(f'Incomplete or corrupt download: {part}')
            part.replace(dest)
        save(marker, {'status':'verified', 'task':row, 'sha256':expected, 'size':item['size'],
            'repo':REPO,'revision':pinned['revision']})
        print(json.dumps({'task':row['key'],'status':'verified'}),flush=True)

    # The reused microwave is checked first, then official task order.
    ordered = sorted(rows,key=lambda r:r['id']!=35)
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for _ in pool.map(download,ordered): pass
    save(ROOT/('data_scope_complete.json' if a.task_ids else 'data_complete.json'),
         {'status':'complete','tasks':len(rows),'task_ids':[r['id'] for r in rows],'revision':pinned['revision']})


if __name__ == '__main__': main()
