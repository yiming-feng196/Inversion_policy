"""Pause only the two explicitly approved RoboCasa worker processes."""
import json
import os
from pathlib import Path
import signal
import time

ROOT=Path('/data/jhr/robocasa_prior_20260920')
for gpu,pid in [(0,1464000),(1,1464001)]:
    proc=Path(f'/proc/{pid}')
    if proc.exists():
        args=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
        if str(ROOT/'extract.py') not in args:
            raise RuntimeError(f'PID {pid} is no longer the approved worker')
        os.kill(pid,signal.SIGTERM)
        deadline=time.time()+30
        while proc.exists() and time.time()<deadline:
            time.sleep(.25)
        if proc.exists():
            raise RuntimeError(f'Worker {pid} did not stop; refusing to kill other processes')
    path=ROOT/f'worker_gpu{gpu}.json'
    old=json.loads(path.read_text())
    state={'state':'paused_user_request','reason':'Prioritize LIBERO-90, preserve all completed source files',
           'previous_status':old,'time':time.time(),'pid':pid}
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(state,indent=2));tmp.replace(path)
    print(json.dumps({'gpu':gpu,'pid':pid,'state':state['state']}),flush=True)
