"""Small shared helpers: seeding, hashing, run records."""
import hashlib
import json
import platform
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
import torchvision


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds CUDA


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_info(repo_root: Path) -> dict:
    """Current commit and whether there are uncommitted changes (None if git is unavailable)."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=repo_root, capture_output=True, text=True).stdout.strip()
    try:
        return {"commit": git("rev-parse", "HEAD"), "uncommitted_changes": bool(git("status", "--porcelain"))}
    except OSError:
        return {"commit": None, "uncommitted_changes": None}


def environment_info(device: torch.device) -> dict:
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "platform": platform.platform(),
        "device": str(device),
        "cpu_threads": torch.get_num_threads(),
    }
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(device)
    return info


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        # Fail loudly rather than silently training on the CPU (a CPU-only torch build is the usual cause)
        raise RuntimeError(f"device '{name}' requested but CUDA is not available (torch {torch.__version__})")
    return torch.device(name)


def save_json(obj, path: Path) -> None:
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")
