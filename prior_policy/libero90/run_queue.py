"""Two cooperating GPU workers; task locks, resumable stages, explicit failure states."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import traceback

ROOT=Path(__file__).resolve().parent
BASE=Path('/data/jhr/q1_libero_crossarch_20260915')
ADAPTER=BASE/'code_v2_20260916'
PY=str(BASE/'venv/bin/python')
SIM='/data/jhr/q2_sampler_comparison_20260919/sim_venv/bin/python'
REPO='/data/jhr/MomentVLA_inversion_osd'
CHILD=None
SERVERS=[]
STOP=False


def read(p): return json.loads(p.read_text()) if p.exists() else {}


def job_disposition(state, continue_on_error):
    """Never relabel failed jobs as complete or silently retry their artifacts."""
    if state=='complete': return 'skip_complete'
    if state=='failed': return 'skip_failed' if continue_on_error else 'raise_failed'
    return 'run'


def save(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2));temp.replace(p)


def stop(sig,frame):
    global STOP
    STOP=True
    if CHILD is not None and CHILD.poll() is None: os.killpg(CHILD.pid,signal.SIGTERM)
    for server in SERVERS:
        if server.poll() is None:os.killpg(server.pid,signal.SIGTERM)


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--first',choices=['unet','dit'],required=True)
    p.add_argument('--seeds',default='0,1,2');p.add_argument('--kinds',default='unet,dit,dp_unet')
    p.add_argument('--task-ids',default=None,help='Comma-separated IDs; otherwise use active_scope.json when present')
    p.add_argument('--continue-on-error',action='store_true',
                   help='Keep failed jobs explicit and process other jobs; failures still require inspection')
    a=p.parse_args();seeds=[int(x) for x in a.seeds.split(',')]
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    lock=(ROOT/f'worker_gpu{a.gpu}.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status_path=ROOT/f'worker_gpu{a.gpu}.json'
    env={**os.environ,'CUDA_VISIBLE_DEVICES':str(a.gpu),'OMP_NUM_THREADS':'4',
        'MKL_NUM_THREADS':'4','PYTHONUNBUFFERED':'1','MUJOCO_GL':'egl','PYOPENGL_PLATFORM':'egl',
        'LIBERO_CONFIG_PATH':'/data/jhr/pi0_dsbc_20260913',
        'PYTHONPATH':f'{ROOT}/deps:/data/jhr/LIBERO'}
    status={'status':'running','pid':os.getpid(),'gpu':a.gpu,'args':vars(a),'history':[],
            'failed_jobs':[]}

    def run(stage,command):
        global CHILD
        if STOP: raise InterruptedError('User stopped worker')
        log=ROOT/'logs'/f'{stage}.log'
        status.update(stage=stage,command=list(map(str,command)),log=str(log));save(status_path,status)
        with log.open('a') as handle:
            CHILD=subprocess.Popen(list(map(str,command)),stdout=handle,stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,env=env,start_new_session=True,cwd=ROOT)
            status['child_pid']=CHILD.pid;save(status_path,status)
            code=CHILD.wait()
        status['history'].append({'stage':stage,'returncode':code,'time':time.time()});save(status_path,status)
        if code: raise RuntimeError(f'{stage} failed ({code}); inspect {log}')

    def fm_task(task,arch):
        tag=f"{task['key']}_{arch}";base=ROOT/'base'/f'{tag}_s0';checkpoint=base/'best.pt'
        adapter=ROOT/'dp_adapter' if arch=='dp_unet' else ADAPTER
        decoder='ddim' if arch=='dp_unet' else 'midpoint'
        decoder_steps=100 if arch=='dp_unet' else 10
        # The initial microwave models were launched before the queue was installed.
        manifest=base/'manifest.json'
        while read(manifest).get('status')=='running' and task['id']==35 and arch in ('unet','dit'):
            pid={'unet':1472578,'dit':1472579}[arch]
            try: os.kill(pid,0)
            except ProcessLookupError: raise RuntimeError(f'Initial training stopped before completion: {base}')
            status.update(stage=f'waiting_initial_base_{tag}',child_pid=pid);save(status_path,status)
            if STOP: raise InterruptedError()
            time.sleep(15)
        if read(manifest).get('status')!='complete':
            prefix=[PY,adapter/'libero_fm.py']
            if arch!='dp_unet':prefix+=['train','--arch',arch]
            run(f'base_{tag}_s0',prefix+['--repo',REPO,'--hdf5',task['hdf5'],
                '--output',base,'--seed','0','--steps','10000','--batch-size','32',
                '--eval-every','500','--validation-samples','256'])
        cache=ROOT/'caches'/f'{tag}_stride8'
        if read(cache/'manifest.json').get('status')!='complete':
            run(f'extract_{tag}',[PY,ROOT/'extract_cache.py','--adapter-dir',adapter,'--repo',REPO,
                '--checkpoint',checkpoint,'--output',cache,'--stride','8',
                '--inverse-steps','100' if arch=='dp_unet' else '256','--solver','ddim' if arch=='dp_unet' else 'rk4',
                '--decoder',decoder,'--decoder-steps',str(decoder_steps)])
        preflight=ROOT/'preflight'/tag/'preflight.json'
        if read(preflight).get('status')!='complete':
            run(f'preflight_{tag}',[SIM,ROOT/'evaluation/preflight_libero.py','--hdf5',task['hdf5'],
                '--task-name',task['task_name'],'--suite',task['suite'],'--gate-mode','state_convention',
                '--dataset-creation-script','/data/jhr/LIBERO/scripts/create_dataset.py',
                '--output',preflight.parent])
        if read(preflight).get('preprocessing_gate_passed') is not True: raise RuntimeError('Preflight gate failed')
        for seed in seeds:
            samplers=[]
            for method in ('cgaussian','cflow'):
                out=ROOT/'priors'/tag/f'{method}_s{seed}'
                if read(out/'status.json').get('status')!='completed':
                    cmd=[PY,ROOT/'training/train_samplers.py','--cache',cache,'--repo',REPO,
                        '--output',out,'--method',method,'--seed',str(seed),'--steps','5000',
                        '--batch-size','32','--prior-steps','16']
                    if (out/'latest.pt').exists():cmd.append('--resume')
                    run(f'prior_{tag}_{method}_s{seed}',cmd)
                samplers+=['--sampler',f'{method}={out}/final.pt']
            dest=ROOT/'rollouts'/f'{tag}_s{seed}'
            if read(dest/'rollouts.json').get('status')!='complete':
                methods='gaussian,cgaussian,cflow' if seed==seeds[0] else 'cgaussian,cflow'
                cmd=[SIM,ROOT/'evaluation/eval_closed_loop.py','--q1-code',adapter,'--repo',REPO,
                    '--action-checkpoint',checkpoint,'--cache-manifest',cache/'manifest.json',
                    '--preflight-report',preflight,'--task-name',task['task_name'],'--suite',task['suite'],
                    '--methods',methods,'--trials','50','--max-steps',str(task['max_steps']),
                    '--decoder',decoder,'--decoder-steps',str(decoder_steps),
                    '--output',dest]+samplers
                if (dest/'rollouts.json').exists():cmd.append('--resume')
                run(f'rollout_{tag}_s{seed}',cmd)

    def pi0_task(task):
        folder=ROOT/'pi0';cache=ROOT/'caches'/f"{task['key']}_pi0_stride8"
        jax='/data/jhr/TA-VLA/.venv/bin/python'
        original_env=dict(env)
        env.update(XLA_PYTHON_CLIENT_PREALLOCATE='false',XLA_PYTHON_CLIENT_MEM_FRACTION='.65',
            PYTHONPATH=f'/data/jhr/TA-VLA/src:{folder}:/data/jhr/LIBERO:/data/jhr/TA-VLA/packages/openpi-client/src:/data/jhr/pi0_dsbc_20260913/client_pkgs')
        try:
            if read(cache/'progress.json').get('status')!='complete':
                while not STOP:
                    free=int(subprocess.check_output(['nvidia-smi',f'--id={a.gpu}','--query-gpu=memory.free',
                        '--format=csv,noheader,nounits'],text=True).strip())
                    if free>=22500:break
                    status.update(stage='waiting_pi0_memory',free_mib=free,required_mib=22500);save(status_path,status);time.sleep(20)
                run(f"extract_{task['key']}_pi0",[jax,folder/'extract.py','--task',task['key'],
                    '--output',cache,'--steps','1280','--stride','8','--dtype','float32','--batch-size','2'])
            for seed in seeds:
                models=ROOT/'priors'/f"{task['key']}_pi0"
                for method in ('cgaussian','cflow'):
                    out=models/f'{method}_s{seed}'
                    if read(out/'progress.json').get('status')!='complete':
                        run(f"prior_{task['key']}_pi0_{method}_s{seed}",[PY,folder/'train.py',
                            '--cache',cache,'--output',out,'--method',method,'--seed',str(seed),'--steps','5000'])
                dest=ROOT/'rollouts'/f"{task['key']}_pi0_s{seed}"/'results.json'
                if read(dest).get('status')=='complete':continue
                port=18570+a.gpu
                with socket.socket() as check:
                    if check.connect_ex(('127.0.0.1',port))==0:raise RuntimeError(f'Port {port} occupied')
                ready=ROOT/'logs'/f"ready_{task['key']}_pi0_s{seed}.json"
                with (ROOT/'logs'/f"serve_{task['key']}_pi0_s{seed}.log").open('a') as log:
                    server=subprocess.Popen([jax,str(folder/'serve.py'),'--models',str(models),'--seed',str(seed),
                        '--port',str(port),'--ready',str(ready)],env=env,stdout=log,stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL,start_new_session=True,cwd='/data/jhr/TA-VLA')
                SERVERS.append(server)
                try:
                    deadline=time.monotonic()+900
                    while not STOP and time.monotonic()<deadline:
                        if server.poll() is not None:raise RuntimeError('pi0 server exited during startup')
                        with socket.socket() as check:
                            if check.connect_ex(('127.0.0.1',port))==0:break
                        time.sleep(3)
                    else:raise RuntimeError('pi0 service not ready')
                    run(f"rollout_{task['key']}_pi0_s{seed}",[SIM,folder/'evaluate.py','--task',task['key'],
                        '--output',dest,'--trials','50','--training-seed',str(seed),'--port',str(port)])
                finally:
                    if server.poll() is None:os.killpg(server.pid,signal.SIGTERM);server.wait(timeout=30)
        finally:env.clear();env.update(original_env)

    try:
        while not (ROOT/'tasks.json').exists():
            if STOP: raise InterruptedError()
            time.sleep(10)
        tasks=read(ROOT/'tasks.json')['tasks']
        scope=read(ROOT/'active_scope.json')
        requested=[int(x) for x in a.task_ids.split(',')] if a.task_ids else scope.get('task_ids',[t['id'] for t in tasks])
        if len(requested)!=len(set(requested)) or not set(requested)<=set(t['id'] for t in tasks):
            raise ValueError('Invalid or duplicate selected task IDs')
        if scope and not set(requested)<=set(scope['task_ids']):raise ValueError('Tasks outside the user-approved active scope')
        tasks=[t for t in tasks if t['id'] in requested]
        order={task_id:i for i,task_id in enumerate(requested)}
        task_order=sorted(tasks,key=lambda t:order[t['id']])
        kinds=a.kinds.split(',')
        if any(x not in ('unet','dit','dp_unet','pi0') for x in kinds):raise ValueError('Unsupported queue kind')
        if scope and not set(kinds)<=set(scope['model_families']):raise ValueError('Policy family outside the active scope')
        status.update(scope=scope,selected_task_ids=requested,expected_jobs=len(tasks)*len(kinds));save(status_path,status)
        jobs=[(task,arch) for task in task_order for arch in kinds]
        # Both workers begin with a different already-running microwave model.
        jobs.sort(key=lambda j:(order[j[0]['id']],
            0 if j[1]==a.first else {'unet':1,'dit':1,'dp_unet':2,'pi0':3}[j[1]]))
        while not STOP:
            pending=0;ran=False
            for task,arch in jobs:
                key=f"{task['key']}_{arch}";job=ROOT/'jobs'/key;job.mkdir(parents=True,exist_ok=True)
                disposition=job_disposition(read(job/'status.json').get('status'),a.continue_on_error)
                if disposition=='skip_complete':continue
                if disposition=='skip_failed':
                    if key not in status['failed_jobs']:status['failed_jobs'].append(key)
                    continue
                pending+=1
                if read(ROOT/'data_status'/f"{task['key']}.json").get('status')!='verified':continue
                handle=(job/'lock').open('w')
                try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:handle.close();continue
                disposition=job_disposition(read(job/'status.json').get('status'),a.continue_on_error)
                if disposition=='skip_complete':handle.close();continue
                if disposition=='skip_failed':
                    if key not in status['failed_jobs']:status['failed_jobs'].append(key)
                    handle.close();continue
                if disposition=='raise_failed':
                    handle.close();raise RuntimeError(f'Previous failed job requires inspection: {key}')
                save(job/'status.json',{'status':'running','gpu':a.gpu,'pid':os.getpid(),'task':task,'arch':arch})
                try:
                    if arch=='pi0':pi0_task(task)
                    else:fm_task(task,arch)
                    save(job/'status.json',{'status':'complete','task':task,'arch':arch,'seeds':seeds,'time':time.time()})
                except Exception:
                    save(job/'status.json',{'status':'paused' if STOP else 'failed','error':traceback.format_exc(),'task':task,'arch':arch})
                    if STOP or not a.continue_on_error:raise
                    if key not in status['failed_jobs']:status['failed_jobs'].append(key)
                    status['last_job_error']=traceback.format_exc();save(status_path,status)
                finally:handle.close()
                ran=True;break
            if not pending:break
            if not ran:
                status.update(stage='waiting_data_or_other_worker');save(status_path,status);time.sleep(20)
        outcome='complete_with_failures' if status['failed_jobs'] else 'complete'
        status.update(status='paused' if STOP else outcome);save(status_path,status)
    except BaseException:
        status.update(status='paused' if STOP else 'failed',error=traceback.format_exc());save(status_path,status)
        raise


if __name__=='__main__':main()
