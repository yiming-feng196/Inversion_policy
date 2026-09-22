"""Freeze old schedulers, drain useful stages, restart within the approved scope.

SIGSTOP targets only the two verified scheduler PIDs, never GPU child jobs or
other users. Useful current children finish normally, then the old schedulers
receive their graceful stop signal before they can dispatch any new stage.
"""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

ROOT=Path(__file__).resolve().parent
PY='/data/jhr/q1_libero_crossarch_20260915/venv/bin/python'
SIM='/data/jhr/q2_sampler_comparison_20260919/sim_venv/bin/python'
EXPECTED={0:1480816,1:1480817}
DOWNLOAD_PID=1488160
ARCHIVE=ROOT/'scope_switch_20260921'


def read(p):return json.loads(p.read_text()) if p.exists() else {}


def save(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(p)


def cmd(pid):
    p=Path(f'/proc/{pid}/cmdline')
    return p.read_bytes().replace(b'\0',b' ').decode() if p.exists() else ''


def alive(pid):
    p=Path(f'/proc/{pid}/stat')
    if not p.exists():return False
    try:return p.read_text().rsplit(')',1)[1].split()[0]!='Z'
    except FileNotFoundError:return False


def children(pid):
    found=set()
    for p in Path(f'/proc/{pid}/task').glob('*/children'):
        try:found.update(map(int,p.read_text().split()))
        except FileNotFoundError:pass
    return sorted(found)


def descendants(pid):
    result=[]
    for child in children(pid):result.extend(descendants(child));result.append(child)
    return result


def main():
    ARCHIVE.mkdir(exist_ok=True)
    if (ARCHIVE/'switch_started.json').exists():
        raise RuntimeError('Switch already invoked; inspect existing handoff before restarting')
    scope=read(ROOT/'active_scope.json');assert len(scope['task_ids'])==5 and 'pi0' not in scope['model_families']
    old=[]
    for gpu,pid in EXPECTED.items():
        if not alive(pid):raise RuntimeError(f'Expected scheduler {pid} not live; re-audit before switch')
        if str(ROOT/'run_queue.py') not in cmd(pid):raise RuntimeError(f'PID identity changed: {pid}')
    if alive(DOWNLOAD_PID) and str(ROOT/'prepare_data.py') not in cmd(DOWNLOAD_PID):
        raise RuntimeError('Downloader identity changed')
    # Stop dispatch immediately on BOTH GPUs before waiting for their children.
    for gpu,pid in EXPECTED.items():
        os.kill(pid,signal.SIGSTOP)
        s=read(ROOT/f'worker_gpu{gpu}.json')
        owned=children(pid)
        if any(str(ROOT) not in cmd(child) for child in owned if alive(child)):
            raise RuntimeError(f'Unexpected child under scheduler {pid}; keep paused for inspection')
        rec={'gpu':gpu,'old_pid':pid,'old_status':s,'children':owned,'state':'draining_current_stage',
            'scope':scope,'started':time.time()}
        old.append(rec);save(ARCHIVE/f'gpu{gpu}.json',rec)
    save(ARCHIVE/'switch_started.json',{'scope':scope,'time':time.time()})
    # Stop only the old download controller and its own curl descendants.
    if alive(DOWNLOAD_PID):
        os.kill(DOWNLOAD_PID,signal.SIGSTOP)
        owned=descendants(DOWNLOAD_PID)
        save(ARCHIVE/'download_before.json',{'pid':DOWNLOAD_PID,'children':owned,'partial_files_preserved':True})
        for child in owned:
            if alive(child):os.kill(child,signal.SIGTERM)
        os.kill(DOWNLOAD_PID,signal.SIGTERM);os.kill(DOWNLOAD_PID,signal.SIGCONT)
        deadline=time.monotonic()+30
        while alive(DOWNLOAD_PID) and time.monotonic()<deadline:time.sleep(.25)
        if alive(DOWNLOAD_PID):raise RuntimeError('Old downloader did not stop; no overlapping downloader started')
    data_env={**os.environ,'PYTHONPATH':'/data/jhr/LIBERO',
        'LIBERO_CONFIG_PATH':'/data/jhr/pi0_dsbc_20260913','PYTHONUNBUFFERED':'1'}
    with (ROOT/'logs/data_five_tasks.log').open('a') as f:
        download=subprocess.Popen([SIM,str(ROOT/'prepare_data.py'),'--task-ids',','.join(map(str,scope['task_ids']))],
            env=data_env,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    save(ARCHIVE/'download_five.json',{'pid':download.pid,'task_ids':scope['task_ids']})

    def drain_and_restart(rec):
        gpu,pid=rec['gpu'],rec['old_pid']
        try:
            match=re.search(r't(\d{3})',rec['old_status'].get('stage',''))
            useful=match is not None and int(match.group(1)) in scope['task_ids'] and '_pi0' not in rec['old_status'].get('stage','')
            if not useful:
                for child in rec['children']:
                    if alive(child):os.kill(child,signal.SIGTERM)
                rec['state']='stopping_deferred_stage';save(ARCHIVE/f'gpu{gpu}.json',rec)
            while any(alive(child) for child in rec['children']):time.sleep(5)
            # Old handler observes STOP before any next subprocess can launch.
            os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
            deadline=time.monotonic()+60
            while alive(pid) and time.monotonic()<deadline:time.sleep(.25)
            if alive(pid):raise RuntimeError(f'Old scheduler {pid} has not exited; refusing duplicate worker')
            save(ARCHIVE/f'worker_gpu{gpu}_drained.json',read(ROOT/f'worker_gpu{gpu}.json'))
            command=[PY,str(ROOT/'run_queue.py'),'--gpu',str(gpu),'--first','unet' if gpu==0 else 'dit',
                '--kinds','unet,dit,dp_unet','--task-ids',','.join(map(str,scope['task_ids'])),'--seeds','0,1,2']
            with (ROOT/f'logs/queue_five_gpu{gpu}.log').open('a') as f:
                process=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,start_new_session=True,cwd=ROOT)
            rec.update(state='restarted_five_task_scope',new_pid=process.pid,command=command,time=time.time())
            save(ARCHIVE/f'gpu{gpu}.json',rec)
            print(json.dumps({'gpu':gpu,'new_pid':process.pid,'state':rec['state']}),flush=True)
        except BaseException as error:
            rec.update(state='handoff_failed',error=repr(error));save(ARCHIVE/f'gpu{gpu}.json',rec);raise
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(drain_and_restart,old))
    save(ARCHIVE/'complete.json',{'status':'complete','time':time.time(),'scope':scope})


if __name__=='__main__':main()
