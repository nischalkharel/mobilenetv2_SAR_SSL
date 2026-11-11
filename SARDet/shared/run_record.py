"""Run records: everything a reader needs to know how and where a run happened.

RunRecord writes run_info.json at the start (status running) and at the end (completed / failed) of every
SSL run, fine-tune and evaluation: config, git commit of the suite, dataset root, manifest hashes, hardware,
software versions, wall time and peak GPU memory.
"""
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from . import paths


def sha256_of_file(path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_of_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_of_state_dict(state_dict) -> str:
    """Content hash of a state dict (tensor bytes in key order), independent of the file it came from."""
    h = hashlib.sha256()
    for k in sorted(state_dict):
        h.update(k.encode())
        h.update(state_dict[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def git_info(repo_dir=None) -> dict:
    repo_dir = str(repo_dir or paths.SUITE_ROOT)
    try:
        run = lambda *a: subprocess.check_output(["git", "-C", repo_dir, *a], text=True, stderr=subprocess.DEVNULL).strip()
        return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
                "uncommitted_changes": bool(run("status", "--porcelain"))}
    except Exception as e:  # noqa: BLE001 - a missing git must not stop a run
        return {"commit": None, "error": str(e)}


def environment_info() -> dict:
    import torch
    info = {"python": sys.version.split()[0], "torch": torch.__version__, "platform": platform.platform(),
            "hostname": socket.gethostname(), "cpu_count": os.cpu_count(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "cuda_available": torch.cuda.is_available()}
    try:
        import torchvision
        info["torchvision"] = torchvision.__version__
    except ImportError:
        pass
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
    return info


def peak_gpu_memory_gb():
    import torch
    if torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated() / 2**30, 2)
    return None


def write_json(path, obj):
    """Atomic JSON write (tmp + replace) so a crash never leaves a half file."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def make_run_dir(parent, name: str) -> Path:
    """<parent>/<YYYYmmdd-HHMMSS>_<name>; retries on the next second if the name is taken (parallel jobs)."""
    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    while True:
        run_dir = parent / f"{datetime.now():%Y%m%d-%H%M%S}_{name}"
        try:
            run_dir.mkdir()
            return run_dir
        except FileExistsError:
            time.sleep(1)


class RunRecord:
    """run_info.json of one run. start() first, update(...) freely, finish()/fail() at the end."""

    def __init__(self, run_dir, kind: str, config: dict, extra: dict = None):
        self.run_dir = Path(run_dir)
        self.path = self.run_dir / "run_info.json"
        self.t0 = time.time()
        self.info = {"kind": kind, "status": "running", "started": datetime.now().isoformat(timespec="seconds"),
                     "run_dir": str(self.run_dir), "config": config, "git": git_info(),
                     "environment": environment_info(), "dataset_root": str(paths.SARDET100K_ROOT),
                     "split": paths.SPLIT, "command": " ".join(sys.argv)}
        if extra:
            self.info.update(extra)

    def start(self):
        write_json(self.path, self.info)
        return self

    def update(self, **fields):
        self.info.update(fields)
        write_json(self.path, self.info)

    def finish(self, **fields):
        self.info.update(fields)
        self.info.update(status="completed", finished=datetime.now().isoformat(timespec="seconds"),
                         wall_minutes=round((time.time() - self.t0) / 60, 2), peak_gpu_memory_gb=peak_gpu_memory_gb())
        write_json(self.path, self.info)

    def fail(self, exc: BaseException):
        self.info.update(status="failed", finished=datetime.now().isoformat(timespec="seconds"),
                         wall_minutes=round((time.time() - self.t0) / 60, 2),
                         error="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
        write_json(self.path, self.info)
