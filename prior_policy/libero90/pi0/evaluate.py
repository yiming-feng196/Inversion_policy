"""Incremental, resumable paired LIBERO evaluation. No expert query actions."""
from __future__ import annotations
import argparse
from collections import deque
import json
import math
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from protocol import TASKS, write_json

# Trust only the installed LIBERO package's initial-state files.
_original_load=torch.load
def _libero_load(path,*args,**kwargs):
    if '/LIBERO/libero/libero/init_files/' in str(Path(path).resolve()):
        kwargs.setdefault('weights_only',False)
    return _original_load(path,*args,**kwargs)
torch.load=_libero_load

from libero.libero import benchmark,get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client.websocket_client_policy import WebsocketClientPolicy


def axisangle(q):
    q=np.array(q,copy=True);q[3]=np.clip(q[3],-1,1)
    den=np.sqrt(1-q[3]**2)
    return np.zeros(3) if math.isclose(den,0.) else q[:3]*2*math.acos(q[3])/den


def request(obs,prompt,method,seed,initial,query):
    images={}
    for src,dest in [('agentview_image','observation/image'),('robot0_eye_in_hand_image','observation/wrist_image')]:
        img=np.ascontiguousarray(obs[src][::-1,::-1])
        images[dest]=image_tools.convert_to_uint8(image_tools.resize_with_pad(img,224,224))
    return {**images,'observation/state':np.concatenate([obs['robot0_eef_pos'],axisangle(obs['robot0_eef_quat']),obs['robot0_gripper_qpos']]).astype(np.float32),
        'prompt':prompt,'_method':method,'_noise_seed':seed,'_initial_state':initial,'_query_index':query}


def main():
    p=argparse.ArgumentParser();p.add_argument('--task',choices=TASKS,required=True)
    p.add_argument('--output',required=True);p.add_argument('--port',type=int,default=8147)
    p.add_argument('--trials',type=int,default=50);p.add_argument('--noise-seed',type=int,default=0)
    p.add_argument('--training-seed',type=int,default=0)
    p.add_argument('--methods',default=None)
    a=p.parse_args();output=Path(a.output)
    if a.methods is None:
        a.methods='gaussian,cgaussian,cflow' if a.training_seed==0 else 'cgaussian,cflow'
    cfg=TASKS[a.task];suite=benchmark.get_benchmark_dict()[cfg['suite']]()
    ids=[i for i in range(suite.n_tasks) if suite.get_task(i).name==cfg['task_name']]
    if len(ids)!=1:raise ValueError(f'Task mapping is ambiguous: {ids}')
    task=suite.get_task(ids[0]);initial=suite.get_task_init_states(ids[0])
    if str(task.language).lower()!=cfg['prompt'].lower():raise ValueError('Training/evaluation language mismatch')
    if not 0<a.trials<=len(initial):raise ValueError('Invalid trial count')
    client=WebsocketClientPolicy('127.0.0.1',a.port)
    methods=a.methods.split(',')
    if len(set(methods))!=len(methods) or set(methods)-{'gaussian','cgaussian','cflow'}:raise ValueError('Invalid methods')
    result={'status':'running','args':vars(a),'task':cfg,'task_id':ids[0],
        'methods':methods,'records':[],
        'protocol':{'replan_steps':10,'wait_steps':10,'render':256,'env_seed':7,
          'action_postprocessing':'checkpoint output transform, no extra clip/sign flip',
          'initial_states':'standard packaged; not certified unseen geometry',
          'sampling':'counter-based independent of method, full 50x32 source each query'}}
    if output.exists():
        old=json.loads(output.read_text())
        if old['args']!=result['args'] or old['protocol']!=result['protocol']:raise ValueError('Resume protocol changed')
        result=old
        if any(r['status']!='complete' for r in result['records']):raise ValueError('Investigate recorded errors before resuming')
    donekeys={(r['method'],r['initial_state']) for r in result['records']}
    path=Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
    env=OffScreenRenderEnv(bddl_file_name=str(path),camera_heights=256,camera_widths=256)
    try:
        for idx in range(a.trials):
            # Interleave methods and rotate ordering to limit wall-clock drift.
            methods=result['methods'];rotation=idx%len(methods);methods=methods[rotation:]+methods[:rotation]
            for method in methods:
                if (method,idx) in donekeys:continue
                record={'method':method,'initial_state':idx,'status':'running','training_seed':a.training_seed,'noise_seed':a.noise_seed}
                start=time.time()
                try:
                    env.seed(7);env.reset();obs=env.set_init_state(initial[idx]);success=False
                    for _ in range(10):obs,_,success,_=env.step([0.]*6+[-1.])
                    if success:raise RuntimeError('Task already successful during settling')
                    plan=deque();query=0;latencies=[];source_latencies=[];norms=[]
                    actions=[]
                    for step in range(cfg['max_steps']):
                        if not plan:
                            response=client.infer(request(obs,str(task.language),method,a.noise_seed,idx,query))
                            chunk=np.asarray(response['actions'])
                            if chunk.shape!=(50,7) or not np.isfinite(chunk).all():raise ValueError('Invalid action chunk')
                            plan.extend(chunk[:10]);query+=1
                            latencies.append(float(response['total_ms']));source_latencies.append(float(response['source_ms']));norms.append(float(response['source_rms']))
                        action=np.asarray(plan.popleft());actions.append(action)
                        obs,_,success,_=env.step(action.tolist())
                        if success:break
                    trajectory=np.asarray(actions)
                    trajpath=output.parent/'trajectories'/f'{method}_{idx:03d}.npz';trajpath.parent.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(trajpath,actions=trajectory,source_rms=norms,policy_ms=latencies,source_ms=source_latencies)
                    record.update(status='complete',success=bool(success),steps=step+1,queries=query,seconds=time.time()-start,
                        action_outside_unit_fraction=float((np.abs(trajectory)>1).mean()),
                        trajectory=str(trajpath),policy_ms_median_warm=float(np.median(latencies[1:] or latencies)),
                        source_ms_median_warm=float(np.median(source_latencies[1:] or source_latencies)))
                except BaseException:
                    record.update(status='error',error=traceback.format_exc(),seconds=time.time()-start)
                    result['records'].append(record);result['status']='failed';write_json(output,result);raise
                result['records'].append(record)
                result['summary']={m:{'successes':sum(r['success'] for r in result['records'] if r['method']==m),'trials':sum(r['method']==m for r in result['records'])} for m in result['methods']}
                write_json(output,result);print(json.dumps(record),flush=True)
        result['status']='complete';write_json(output,result)
    finally:
        env.close()


if __name__=='__main__':main()
