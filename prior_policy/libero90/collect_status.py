"""Read-only experiment inventory; never present partial suites as full means."""
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parent


def read(p):return json.loads(p.read_text()) if p.exists() else {}


def main():
    registry=read(ROOT/'tasks.json')
    scope=read(ROOT/'active_scope.json')
    task_ids=scope.get('task_ids',[x['id'] for x in registry.get('tasks',[])])
    task_keys={f't{i:03d}' for i in task_ids}
    families=scope.get('model_families',['unet','dit','dp_unet','pi0'])
    active_configs={f'{task}_{family}' for task in task_keys for family in families}
    result={'time_unix':time.time(),'suite':'libero_90','scope_name':scope.get('name','full_suite'),
        'expected_tasks':len(task_ids),'selected_task_ids':task_ids,
        'model_families':families,
        'paused_model_families':scope.get('paused_model_families',[]),
        'scope_switch':{p.stem:read(p).get('state') for p in (ROOT/'scope_switch_20260921').glob('gpu*.json')},
        'verified_selected_datasets':sum(read(ROOT/'data_status'/f'{key}.json').get('status')=='verified' for key in task_keys),
        'verified_datasets':sum(read(p).get('status')=='verified' for p in (ROOT/'data_status').glob('*.json')),
        'frs_89_excluded_task':registry.get('frs_89_excluded_task'),
        'workers':{},'completed_base_checkpoints':[], 'complete_caches':[], 'rollouts':[],
        'suite_success_rate':None,
        'note':f'Counts cover active families only. The {len(task_ids)}-task scope is not the full LIBERO-90 suite; report original and extension cohorts separately. DP is paused by user request; existing DP artifacts remain preserved. pi0 is deferred.'}
    for path in ROOT.glob('worker_gpu*.json'):
        s=read(path);result['workers'][path.stem]={k:s.get(k) for k in ('status','pid','stage','child_pid','log','error')}
    for path in (ROOT/'base').glob('*/manifest.json'):
        if path.parent.name.rsplit('_s',1)[0] not in active_configs:continue
        if read(path).get('status')=='complete':result['completed_base_checkpoints'].append(path.parent.name)
    for path in (ROOT/'caches').glob('*/manifest.json'):
        if path.parent.name.rsplit('_stride',1)[0] not in active_configs:continue
        m=read(path)
        if m.get('status')=='complete' or read(path.parent/'progress.json').get('status')=='complete':
            result['complete_caches'].append({'name':path.parent.name,'counts':m.get('counts'),'full_counts':m.get('full_counts')})
    for path in sorted((ROOT/'rollouts').glob('*/*.json')):
        if path.name not in ('results.json','rollouts.json'):continue
        if path.parent.name.rsplit('_s',1)[0] not in active_configs:continue
        r=read(path);complete=[x for x in r.get('records',[]) if x.get('status')=='complete']
        methods={}
        for rec in complete:
            values=methods.setdefault(rec['method'],{'completed':0,'successes':0})
            values['completed']+=1;values['successes']+=int(rec['success'])
        result['rollouts'].append({'name':path.parent.name,'status':r.get('status'),'methods':methods,
            'partial_not_final':r.get('status')!='complete'})
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
