"""Detached finite-queue continuation without interrupting active GPU workers.

The original five-task workers keep their locks and finish normally. This
supervisor launches expanded workers only when the corresponding GPU-worker
lock is free. Experiment hyperparameters and already completed jobs are not
changed. Failed jobs remain failed and require inspection; they do not block
unrelated tasks. No credentials, cron entries, or daemon installation is used.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parent
PY = '/data/jhr/q1_libero_crossarch_20260915/venv/bin/python'
SIM = '/data/jhr/q2_sampler_comparison_20260919/sim_venv/bin/python'
STOP = False


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def validate_scope(scope, registry):
    ids = scope['task_ids']
    if len(ids) != len(set(ids)) or not set(ids) <= {t['id'] for t in registry['tasks']}:
        raise ValueError('Invalid task IDs')
    original = scope['original_task_ids']
    if ids[:len(original)] != original or ids[len(original):] != scope['added_task_ids']:
        raise ValueError('Extension must retain original task order and append new tasks')
    checks = {'model_families': ['unet', 'dit'],
              'paused_model_families': ['dp_unet'],
              'prior_training_seeds': [0, 1, 2], 'trials_per_method_seed': 50,
              'source_methods': ['gaussian', 'cgaussian', 'cflow'],
              'source_stride': 8, 'retain_terminal_window': True}
    for name, expected in checks.items():
        if scope.get(name) != expected:
            raise ValueError(f'Unexpected experimental protocol: {name}')
    return ids


def inspect_jobs(root, scope):
    groups = {}
    for task_id in scope['task_ids']:
        for family in scope['model_families']:
            key = f't{task_id:03d}_{family}'
            value = read(root / 'jobs' / key / 'status.json').get('status', 'pending')
            groups.setdefault(value, []).append(key)
    return groups


def lock_is_free(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(handle, fcntl.LOCK_UN)
        return True


def launch(command, log, env, root):
    with log.open('a') as handle:
        return subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, env=env,
                                start_new_session=True, cwd=root)


def known_child_is_live(root, gpu):
    """Avoid GPU overlap if a scheduler dies but its stage process survives."""
    pid = read(root / f'worker_gpu{gpu}.json').get('child_pid')
    if not isinstance(pid, int):
        return False
    proc = Path('/proc') / str(pid)
    try:
        state = (proc / 'stat').read_text().rsplit(')', 1)[1].split()[0]
        command = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
    except FileNotFoundError:
        return False
    if state == 'Z':
        return False
    # A2A adapter training, shared simulation runtime, or a local stage script.
    return any(prefix in command for prefix in
               (str(root), '/data/jhr/q1_libero_crossarch_20260915/',
                '/data/jhr/q2_sampler_comparison_20260919/'))


def stop(signum, frame):
    global STOP
    STOP = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true', help='Validate plan; launch nothing')
    parser.add_argument('--poll-seconds', type=float, default=30)
    args = parser.parse_args()
    scope_path = ROOT / 'active_scope.json'
    scope = read(scope_path)
    ids = validate_scope(scope, read(ROOT / 'tasks.json'))
    signature = hashlib.sha256(scope_path.read_bytes()).hexdigest()
    state_dir = ROOT / 'extension_20260921'
    if args.check:
        print(json.dumps({'scope': scope['name'], 'task_ids': ids,
                          'expected_jobs': len(ids) * len(scope['model_families']),
                          'jobs': inspect_jobs(ROOT, scope)}, indent=2))
        return
    state_dir.mkdir(exist_ok=True)
    guard = (state_dir / 'supervisor.lock').open('a')
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    (ROOT / 'logs').mkdir(exist_ok=True)
    env = {**os.environ, 'PYTHONUNBUFFERED': '1'}
    data_env = {**env, 'PYTHONPATH': '/data/jhr/LIBERO',
                'LIBERO_CONFIG_PATH': '/data/jhr/pi0_dsbc_20260913'}
    tasks_arg = ','.join(map(str, ids))
    state_path = state_dir / 'supervisor.json'
    previous = read(state_path)
    state = {'pid': os.getpid(), 'status': 'running', 'started': time.time(),
             'scope_sha256': signature, 'task_ids': ids,
             'worker_launches': previous.get('worker_launches', {'0': 0, '1': 0}),
             'download_attempts': previous.get('download_attempts', 0),
             'events': previous.get('events', [])}
    own_workers = {}
    downloader = None
    next_launch = {0: 0., 1: 0.}
    next_download = 0.
    try:
        while not STOP:
            now = time.time()
            if hashlib.sha256(scope_path.read_bytes()).hexdigest() != signature:
                raise RuntimeError('Active scope changed: refusing further dispatch; active workers are untouched')
            if (state_dir / 'stop_dispatch.flag').exists():
                state['status'] = 'dispatch_stopped_by_flag'
                break
            jobs = inspect_jobs(ROOT, scope)
            pending = sum(len(v) for k, v in jobs.items() if k not in ('complete', 'failed'))
            missing = [i for i in ids if read(ROOT / 'data_status' / f't{i:03d}.json').get('status') != 'verified']
            state.update(updated=now, job_counts={k: len(v) for k, v in jobs.items()},
                         failed_jobs=jobs.get('failed', []), missing_data=missing,
                         workers={str(g): read(ROOT / f'worker_gpu{g}.json') for g in (0, 1)})
            if not pending:
                state['status'] = 'complete_with_failures' if jobs.get('failed') else 'complete'
                break
            # Avoid exhausting the filesystem; do not kill already running work.
            free_bytes = shutil.disk_usage(ROOT).free
            state['free_disk_bytes'] = free_bytes
            if free_bytes < 50 * (1 << 30):
                state['status'] = 'waiting_disk_space'
                save(state_path, state)
                time.sleep(min(args.poll_seconds, 60))
                continue
            state['status'] = 'running'
            if downloader is not None and downloader.poll() is not None:
                state['events'].append({'time': now, 'kind': 'downloader_exit',
                                        'returncode': downloader.returncode})
                next_download = now + 300
                downloader = None
            if missing and downloader is None and state['download_attempts'] < 3 and now >= next_download:
                command = [SIM, str(ROOT / 'prepare_data.py'), '--workers', '2',
                           '--task-ids', ','.join(map(str, missing))]
                downloader = launch(command, ROOT / 'logs/data_extension.log', data_env, ROOT)
                state['download_attempts'] += 1
                state['download_pid'] = downloader.pid
                state['events'].append({'time': now, 'kind': 'downloader_started',
                                        'pid': downloader.pid, 'task_ids': missing})
            for gpu in (0, 1):
                worker = own_workers.get(gpu)
                if worker is not None:
                    if worker.poll() is None:
                        continue
                    state['events'].append({'time': now, 'kind': 'worker_exit', 'gpu': gpu,
                                            'pid': worker.pid, 'returncode': worker.returncode})
                    next_launch[gpu] = now + 60
                    del own_workers[gpu]
                if now < next_launch[gpu] or state['worker_launches'][str(gpu)] >= 3:
                    continue
                if not lock_is_free(ROOT / f'worker_gpu{gpu}.lock'):
                    continue  # Includes original workers and any independent resumed worker.
                if known_child_is_live(ROOT, gpu):
                    state.setdefault('orphan_child_wait', {})[str(gpu)] = now
                    continue
                command = [PY, str(ROOT / 'run_queue.py'), '--gpu', str(gpu),
                           '--first', 'unet' if gpu == 0 else 'dit',
                           '--kinds', ','.join(scope['model_families']), '--seeds', '0,1,2',
                           '--task-ids', tasks_arg, '--continue-on-error']
                own_workers[gpu] = launch(command, ROOT / f'logs/queue_extension_gpu{gpu}.log', env, ROOT)
                state['worker_launches'][str(gpu)] += 1
                state['events'].append({'time': now, 'kind': 'worker_started', 'gpu': gpu,
                                        'pid': own_workers[gpu].pid, 'command': command})
            save(state_path, state)
            time.sleep(min(args.poll_seconds, 60))
        if STOP:
            state['status'] = 'supervisor_stopped_workers_untouched'
    except BaseException as error:
        state.update(status='supervisor_failed', error=repr(error))
        raise
    finally:
        state['updated'] = time.time()
        save(state_path, state)


if __name__ == '__main__':
    main()
