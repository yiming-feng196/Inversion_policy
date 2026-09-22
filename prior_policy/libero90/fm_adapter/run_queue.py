"""Bounded Q1 experiment queue; supports explicitly authorized GPU sharing.

This is an experiment launcher, not a claim that queued jobs have run.
All output directories are fresh and failures are preserved for inspection.
"""
from __future__ import annotations

import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time


TASKS = {"drawer": "open_the_middle_drawer_of_the_cabinet", "bowl": "put_the_bowl_on_the_plate"}


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def gpu_snapshot(index):
    row = subprocess.check_output([
        "nvidia-smi", f"--id={index}", "--query-gpu=uuid,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True, timeout=20).strip().split(",")
    uuid, used, total, utilization = [x.strip() for x in row]
    processes = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        text=True, timeout=20)
    pids = [int(line.split(",")[1].strip()) for line in processes.splitlines()
            if line.split(",")[0].strip() == uuid]
    return {"index":index, "uuid":uuid, "memory_used_mib":int(used), "memory_total_mib":int(total),
            "utilization_percent":int(utilization), "compute_pids":pids}


def gpu_is_idle(snapshot):
    return not snapshot["compute_pids"] and snapshot["memory_used_mib"] < 768 and snapshot["utilization_percent"] < 10


def gpu_has_capacity(snapshot, minimum_free_mib):
    # Capacity check, not a reservation or a guarantee against later contention.
    return snapshot["memory_total_mib"] - snapshot["memory_used_mib"] >= minimum_free_mib


class Queue:
    def __init__(self, root, gpu, resume=False, shared_gpu=False):
        self.root, self.gpu = root, gpu
        self.code = Path(__file__).resolve().parent
        self.python = root / "venv/bin/python"
        self.state_path = root / "queue_state_20260916_v3.json"
        if resume:
            self.state = json.loads(self.state_path.read_text())
            if self.state["status"] != "waiting_for_gpu" or self.state["gpu"] != gpu:
                raise ValueError("Only resume a stopped waiting controller on the same GPU")
            if any(j["status"] == "running" for j in self.state["jobs"]):
                raise ValueError("A recorded child is still running; do not resume")
            self.state.setdefault("resumes", []).append({"time": stamp(), "shared_gpu": shared_gpu})
        elif self.state_path.exists():
            raise FileExistsError("Queue already exists; inspect state instead of launching a duplicate")
        else:
            self.state = {"created":stamp(), "status":"initializing", "gpu":gpu, "jobs":[],
                      "scope":"Q1 offline audits only; no simulator success-rate claims; first training seed=0"}
        self.state["shared_gpu"] = shared_gpu
        self.update()

    def update(self, **kwargs):
        self.state.update(updated=stamp(), **kwargs)
        save(self.state_path, self.state)

    def run(self, name, command, cwd=None, extra_env=None, timeout=12*3600):
        command = [str(x) for x in command]
        log = self.root / "logs" / f"{name}.log"
        matches = [job for job in self.state["jobs"] if job["name"] == name]
        if len(matches) > 1:
            raise ValueError(f"Duplicate job records: {name}")
        if matches:
            record = matches[0]
            if record["command"] != command:
                raise ValueError(f"Refusing changed scientific command on resume: {name}")
            if record["status"] == "complete":
                output = Path(command[command.index("--output") + 1])
                is_training = any(Path(value).name == "libero_fm.py" and index + 1 < len(command)
                                  and command[index + 1] == "train" for index, value in enumerate(command))
                result_path = output / ("manifest.json" if is_training else "summary.json")
                if json.loads(result_path.read_text())["status"] != "complete":
                    raise ValueError(f"Recorded completion lacks complete result: {name}")
                print(json.dumps({"job":name, "status":"reused_complete", "time":stamp()}), flush=True)
                return True
            if record["status"] != "waiting_for_gpu" or "started" in record:
                raise ValueError(f"Cannot silently rerun started/failed experiment: {name}")
        else:
            record = {"name":name, "status":"waiting_for_gpu", "command":command, "log":str(log), "created":stamp()}
            self.state["jobs"].append(record)
        if log.exists():
            raise FileExistsError(log)
        shared = self.state.get("shared_gpu", False)
        is_pi0 = any(Path(value).name == "pi0_extended.py" for value in command)
        minimum_free = 24576 if is_pi0 else 14336
        record["resource_policy"] = {"shared_gpu":shared, "minimum_free_mib":minimum_free if shared else None,
                                     "note":"Headroom check only; other jobs are never terminated"}
        self.update(status="waiting_for_gpu", current_job=name)
        started = time.monotonic()
        clean_checks = 0
        # Shared with our earlier inversion scripts, not a system-wide GPU lock.
        with open(self.root / "gpu0.lock", "a") as lock:
            while True:
                if time.monotonic()-started > 48*3600:
                    record.update(status="resource_wait_timeout", finished=stamp())
                    self.update(status="resource_wait_timeout")
                    raise TimeoutError("GPU capacity unavailable for 48 hours")
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    snapshot = gpu_snapshot(self.gpu)
                    ready = gpu_has_capacity(snapshot, minimum_free) if shared else gpu_is_idle(snapshot)
                    clean_checks = clean_checks + 1 if ready else 0
                    self.update(gpu_snapshot=snapshot, idle_checks=clean_checks)
                    if clean_checks >= 3:
                        break
                    fcntl.flock(lock, fcntl.LOCK_UN)
                except BlockingIOError:
                    clean_checks = 0
                # Bounded resource polling inside this submitted compute job.
                time.sleep((5 if shared else 15) if clean_checks else 45)
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=str(self.gpu), OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                       MKL_NUM_THREADS="4", PYTHONUNBUFFERED="1")
            env.update(extra_env or {})
            record.update(status="running", started=stamp())
            self.update(status="running")
            print(json.dumps({"job":name, "status":"running", "time":stamp()}), flush=True)
            try:
                with open(log, "x") as out:
                    result = subprocess.run([str(x) for x in command], cwd=cwd or self.code,
                                            env=env, stdout=out, stderr=subprocess.STDOUT, timeout=timeout)
                record.update(status="complete" if result.returncode == 0 else "failed", returncode=result.returncode)
            except subprocess.TimeoutExpired:
                record.update(status="timeout", returncode=None)
            finally:
                record["finished"] = stamp()
                self.update()
                fcntl.flock(lock, fcntl.LOCK_UN)
        print(json.dumps({"job":name,"status":record["status"], "time":stamp()}),flush=True)
        return record["status"] == "complete"

    def fm(self, mode, **options):
        command = [self.python, "-u", self.code / "libero_fm.py", mode,
                   "--repo", "/data/jhr/MomentVLA_inversion_osd"]
        for key, value in options.items():
            command.append("--" + key.replace("_", "-"))
            if value is not True:
                command.append(str(value))
        return command

    def pi0(self, task, samples, label):
        output = self.root / label
        ok = self.run(label, ["/data/jhr/TA-VLA/.venv/bin/python", "-u",
            self.code / "pi0_extended.py", "--helper-dir", "/data/jhr/pi0_dsbc_20260913",
            "--checkpoint", "/data/jhr/openpi_cache/openpi-assets/checkpoints/pi0_libero",
            "--hdf5", self.root / "data" / (TASKS[task] + "_demo.hdf5"),
            "--prompt", TASKS[task].replace("_", " "), "--output", output,
            "--samples", samples, "--batch-size", 2, "--steps", "640,1280", "--reference-steps",1280,
            "--solvers","rk4", "--model-dtype","float32", "--matmul-precision","highest",
            "--stride",10, "--region-control", "--padding-control"], cwd="/data/jhr/TA-VLA", extra_env={
                "XLA_PYTHON_CLIENT_PREALLOCATE":"false", "OPENPI_DATA_HOME":"/data/jhr/openpi_cache",
                "PYTHONPATH":"/data/jhr/TA-VLA/src"})
        if ok:
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES="", OPENBLAS_NUM_THREADS="4", OMP_NUM_THREADS="4")
            log = self.root / "logs" / f"{label}_postprocess.log"
            with open(log,"x") as out:
                result = subprocess.run([str(self.python),str(self.code/"postprocess_pi0.py"),str(output)],
                                        env=env,stdout=out,stderr=subprocess.STDOUT,timeout=300)
            if result.returncode != 0:
                self.state.setdefault("needs_review",[]).append({"model":label,"reason":"source statistics failed; inspect postprocess log"})
                self.update()
        return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="Resume a stopped waiting controller; never rerun completed jobs")
    parser.add_argument("--shared-gpu", action="store_true", help="User-authorized sharing based on available memory")
    args = parser.parse_args()
    queue = Queue(args.root, args.gpu, resume=args.resume, shared_gpu=args.shared_gpu)
    # GPU smoke tests, separate from scientific results; never select a model here.
    for arch in ("dit", "unet"):
        label = f"gpu_smoke_{arch}_0916"
        if not queue.run(label, queue.fm("train", hdf5=args.root/"data"/(TASKS["drawer"]+"_demo.hdf5"),
            arch=arch, output=args.root/label, steps=8, batch_size=8, validation_samples=8, eval_every=4)):
            queue.update(status="smoke_failed")
            return
    for task in TASKS:
        for arch in ("dit", "unet"):
            label = f"{task}_{arch}_seed0_10k"
            output = args.root / label
            if not queue.run(label, queue.fm("train", hdf5=args.root/"data"/(TASKS[task]+"_demo.hdf5"),
                arch=arch, output=output, steps=10000, batch_size=32, eval_every=500, validation_samples=256, seed=0)):
                continue
            chosen_reference = None
            # Escalate accuracy when necessary; preserve both successful and failed controls.
            for reference, inverse_steps in ((256,"64,128,256"), (1024,"512,1024")):
                audit_label = label + f"_pilot_ref{reference}"
                if not queue.run(audit_label, queue.fm("audit", checkpoint=output/"best.pt",
                    output=args.root/audit_label, samples=4, batch_size=4, reference_steps=reference,
                    inverse_steps=inverse_steps, region_control=True, split="test", seed=20260916)):
                    break
                summary = json.loads((args.root/audit_label/"summary.json").read_text())
                if summary["numerical_gate"]["passed"]:
                    chosen_reference = reference
                    break
            if chosen_reference is None:
                queue.state.setdefault("needs_review", []).append({"model":label,"reason":"numerical audit did not pass; no distribution-only conclusion"})
                queue.update()
                continue
            for split in ("train", "test"):
                audit_label = label + f"_{split}_n128_ref{chosen_reference}"
                # Same N for Gaussian and expert groups. Train/test are never pooled.
                queue.run(audit_label, queue.fm("audit", checkpoint=output/"best.pt", output=args.root/audit_label,
                    samples=128, batch_size=16, reference_steps=chosen_reference,
                    inverse_steps=f"{chosen_reference//2},{chosen_reference}", region_control=True,
                    split=split, seed=20260916))
    # Public pi0 checkpoint is a separate pretrained-model replication, not a
    # parameter-/data-matched swap of the small FM backbone.
    if queue.pi0("bowl", 4, "pi0_bowl_n4_region_0916"):
        pilot = json.loads((args.root/"pi0_bowl_n4_region_0916/summary.json").read_text())
        case = pilot["cases"]["rk4_n1280"]
        if case["known_native_noise_recovery"]["full_50x32"]["rmse"] < .005 and case["expert_action_cycle"]["full_50x32"]["rmse"] < .001:
            # Fifty episodes exist in each file: exactly one unpadded chunk/episode.
            queue.pi0("bowl", 50, "pi0_bowl_n50_region_0916")
        else:
            queue.state.setdefault("needs_review", []).append({"model":"pi0_bowl","reason":"pilot numerical control failed"})
    queue.pi0("drawer", 50, "pi0_drawer_n50_region_0916")
    failures = [j["name"] for j in queue.state["jobs"] if j["status"] != "complete"]
    queue.update(status="complete_with_issues" if failures or queue.state.get("needs_review") else "complete",
                 failures=failures, current_job=None)


if __name__ == "__main__":
    main()
