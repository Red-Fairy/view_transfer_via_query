#!/usr/bin/env python3
"""SLURM job manager for the grouped-guidance inference sweep.

Slim adaptation of UE-Render/launch/slurm_batch_manager.py for a fixed 4-task
sweep. Submits one sbatch job per guidance config, polls squeue→sacct, and
resubmits on any non-success terminal state (resume-safe: infer.sh skips
samples whose pred.mp4 already exists). Escalates ma → snavely → gpu so
priority nodes are exhausted before the general pool.

Run on the login node (only needs sbatch/squeue/sacct). Foreground; wrap in
nohup/tmux for unattended use. State in <log_dir>/manager_state.json is read
back on startup, so the manager itself is restart-safe.

    python scripts/guidance_sweep_manager.py [--max-retries 5] [--poll-seconds 60]
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
ENV_PY_BIN = "/home/rl897/anaconda3/envs/wan-view-transfer/bin"
SYS_LIBSTDCPP = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"

LORA_CKPT = str(PROJECT_ROOT / "runs/14B_4gpu_640P_0507/checkpoint-10000/trainable_params.pt")
RUN_TAG = "14B_4gpu_640P_0507"
# Priority first (ma-owned), then snavely, then the general gpu pool.
TIERS = ["ma", "snavely", "gpu"]


def out_dir_for(phase_tag: str, g: str, s: str, t: str) -> Path:
    # Phase tag isolates sweep1/sweep2 so they never collide.
    return PROJECT_ROOT / "infer_out" / RUN_TAG / phase_tag / f"geom{g}_src{s}_text{t}"


def n_done(out_dir: Path) -> int:
    if not out_dir.is_dir():
        return 0
    return sum(1 for d in out_dir.iterdir() if d.is_dir() and (d / "pred.mp4").exists())


@dataclass
class Task:
    name: str
    geom: str
    src: str
    text: str
    phase_tag: str
    locations_file: str
    seed: int
    n_expected: int
    tier: int = 0
    status: str = "PENDING"          # PENDING|QUEUED|RUNNING|SUCCESS|FAILED
    retries_used: int = 0
    job_id: Optional[str] = None
    queued_since: Optional[float] = None
    last_error: str = ""
    history: List[str] = field(default_factory=list)

    @property
    def out_dir(self) -> Path:
        return out_dir_for(self.phase_tag, self.geom, self.src, self.text)


def sh(cmd: List[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()


def squeue_state(job_id: str) -> Optional[str]:
    out = sh(["squeue", "-h", "-j", job_id, "-o", "%T"])
    return out.splitlines()[0].strip() if out else None


def sacct_state(job_id: str) -> Optional[str]:
    out = sh(["sacct", "-X", "-n", "-P", "-j", job_id, "--format=JobIDRaw,State,ExitCode"])
    for line in out.splitlines():
        f = line.split("|")
        if len(f) >= 2 and f[0] == job_id:
            return f[1].split()[0]  # strip "CANCELLED by ..."
    return None


def write_wrapper(task: Task, log_dir: Path, timeout_s: int) -> Path:
    marker = log_dir / f"{task.name}.success"
    marker.unlink(missing_ok=True)
    wrapper = log_dir / f"{task.name}.run.sh"
    body = f"""#!/usr/bin/env bash
set -euo pipefail
export PATH={ENV_PY_BIN}:$PATH
export OPENCV_IO_ENABLE_OPENEXR=1
export LD_PRELOAD={SYS_LIBSTDCPP}
export LOW_VRAM=1
export LORA_CKPT={shlex.quote(LORA_CKPT)}
export LOCATIONS_FILE={shlex.quote(task.locations_file)}
export OUT_DIR={shlex.quote(str(task.out_dir))}
export SEED={task.seed}
export GUIDANCE_GEOM={task.geom} GUIDANCE_SRC={task.src} GUIDANCE_TEXT={task.text}
cd {shlex.quote(str(PROJECT_ROOT))}
timeout --signal=TERM --kill-after=120s {timeout_s}s bash scripts/infer.sh
touch {shlex.quote(str(marker))}
"""
    wrapper.write_text(body)
    wrapper.chmod(0o755)
    return wrapper


def submit(task: Task, log_dir: Path, args) -> None:
    partition = TIERS[min(task.tier, len(TIERS) - 1)]
    wrapper = write_wrapper(task, log_dir, args.task_timeout_seconds)
    cmd = [
        "sbatch", "--parsable",
        "--job-name", f"gsweep-{task.name}",
        "--output", str(log_dir / f"{task.name}.slurm_%j.out"),
        "--error", str(log_dir / f"{task.name}.slurm_%j.out"),
        "--partition", partition,
        "--account", "ma",
        "--gres", "gpu:nvidia_rtx_a6000:1",
        "--mem", args.mem,
        "--cpus-per-task", str(args.cpus),
        "--time", args.time_limit,
        str(wrapper),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        task.last_error = f"sbatch failed: {res.stderr.strip()}"
        task.history.append(f"submit-error[{partition}]: {res.stderr.strip()[:120]}")
        return
    task.job_id = res.stdout.strip().split(";")[0]
    task.status = "QUEUED"
    task.queued_since = time.time()
    task.history.append(f"submit job={task.job_id} part={partition} tier={task.tier}")
    print(f"  submitted {task.name} → job {task.job_id} (partition={partition})")


def save_state(path: Path, tasks: List[Task]) -> None:
    path.write_text(json.dumps(
        {"updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
         "tasks": [asdict(t) for t in tasks]}, indent=2))


def load_state(path: Path) -> Optional[List[Task]]:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    tasks = []
    for d in data["tasks"]:
        d.pop("out_dir", None)  # property, not a field
        tasks.append(Task(**d))
    return tasks


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--max-concurrent", type=int, default=50,
                   help="Cap on QUEUED+RUNNING sbatch jobs at once.")
    p.add_argument("--poll-seconds", type=int, default=60)
    p.add_argument("--escalate-pending-seconds", type=int, default=600,
                   help="Bump a task to the next partition tier if it sits "
                        "PENDING/QUEUED longer than this.")
    p.add_argument("--task-timeout-seconds", type=int, default=21600)  # 6h
    p.add_argument("--mem", default="128G")
    p.add_argument("--cpus", type=int, default=8)
    p.add_argument("--time-limit", default="06:00:00")
    p.add_argument("--sweep-spec", required=True,
                   help="JSON: {phase_tag, locations_file, seed, n_expected, "
                        "configs:[{geom,src,text}...]}")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    spec = json.loads(Path(args.sweep_spec).read_text())
    phase_tag = spec["phase_tag"]
    log_dir = PROJECT_ROOT / "infer_out" / RUN_TAG / "_sweep_logs" / phase_tag
    log_dir.mkdir(parents=True, exist_ok=True)
    state_path = log_dir / "manager_state.json"

    tasks = load_state(state_path)
    if tasks is None:
        tasks = [
            Task(name=f"geom{c['geom']}_src{c['src']}_text{c['text']}",
                 geom=str(c["geom"]), src=str(c["src"]), text=str(c["text"]),
                 phase_tag=phase_tag, locations_file=spec["locations_file"],
                 seed=int(spec["seed"]), n_expected=int(spec["n_expected"]))
            for c in spec["configs"]
        ]
        print(f"[manager] fresh run '{phase_tag}', {len(tasks)} configs")
    else:
        print(f"[manager] resumed '{phase_tag}' from {state_path}")

    if args.dry_run:
        for t in tasks:
            print(f"  {t.name}: out_dir={t.out_dir}  done={n_done(t.out_dir)}/{t.n_expected}")
        return 0

    while True:
        for t in tasks:
            if t.status in ("SUCCESS", "FAILED"):
                continue

            # Filesystem pre-skip / completion gate.
            if n_done(t.out_dir) >= t.n_expected:
                t.status = "SUCCESS"
                t.history.append(f"complete (filesystem {t.n_expected}/{t.n_expected})")
                continue

            if t.status in ("QUEUED", "RUNNING") and t.job_id:
                qs = squeue_state(t.job_id)
                if qs in ("PENDING", "CONFIGURING"):
                    t.status = "QUEUED"
                    if (t.queued_since and t.tier < len(TIERS) - 1
                            and time.time() - t.queued_since > args.escalate_pending_seconds):
                        subprocess.run(["scancel", t.job_id], capture_output=True)
                        t.tier += 1
                        t.status = "PENDING"
                        t.job_id = None
                        t.history.append(f"escalate→tier{t.tier} (queue stall)")
                    continue
                if qs is not None:
                    t.status = "RUNNING"
                    continue
                # Left the queue → terminal state via sacct.
                fs = sacct_state(t.job_id) or "UNKNOWN"
                marker = (log_dir / f"{t.name}.success").exists()
                if fs.startswith("COMPLETED") and marker:
                    t.status = "SUCCESS"
                    t.history.append(f"SUCCESS job={t.job_id}")
                    continue
                t.retries_used += 1
                t.last_error = f"state={fs} marker={marker}"
                t.history.append(f"fail job={t.job_id} {t.last_error} (retry {t.retries_used})")
                t.job_id = None
                if t.retries_used > args.max_retries:
                    t.status = "FAILED"
                    continue
                # Escalate a tier on infra failure (not on a clean code bug:
                # OOM/PREEMPT/NODE_FAIL/TIMEOUT/CANCELLED all warrant a move).
                if fs != "FAILED" and t.tier < len(TIERS) - 1:
                    t.tier += 1
                    t.history.append(f"escalate→tier{t.tier} ({fs})")
                t.status = "PENDING"

        # Submission pass — cap concurrent QUEUED+RUNNING jobs.
        active = sum(t.status in ("QUEUED", "RUNNING") for t in tasks)
        for t in tasks:
            if active >= args.max_concurrent:
                break
            if t.status == "PENDING":
                submit(t, log_dir, args)
                if t.job_id:           # submit() sets QUEUED on success
                    active += 1

        save_state(state_path, tasks)
        done = sum(t.status in ("SUCCESS", "FAILED") for t in tasks)
        line = "  ".join(
            f"{t.name}:{t.status}"
            + (f"({n_done(t.out_dir)}/{t.n_expected})" if t.status != "SUCCESS" else "")
            for t in tasks)
        print(f"[{time.strftime('%H:%M:%S')}] {line}")
        if done == len(tasks):
            break
        time.sleep(args.poll_seconds)

    ok = [t.name for t in tasks if t.status == "SUCCESS"]
    bad = [(t.name, t.last_error) for t in tasks if t.status == "FAILED"]
    print(f"\n[manager] DONE. success={len(ok)}/{len(tasks)}")
    for name, err in bad:
        print(f"  FAILED {name}: {err}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
