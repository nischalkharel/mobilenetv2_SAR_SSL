"""The one place where CondDINO_SARDet imports the DINO_SARDet code it extends.

Nothing from DINO_SARDet is copied: `dino.py` (head, loss, wrapper, diagnostics) and `train_ssl.py` (DINOMethod)
are loaded from `../DINO_SARDet/` with importlib. DINO's train_ssl.py inserts its own folder at sys.path[0] and
imports `dino` from there; that entry is removed again afterwards so that `train_ssl` and `smoke_test` keep
resolving to THIS folder (both folders have files of those names). `dino` stays in sys.modules.

    from dino_base import DINOMethod, dino      # dino.DINOLoss, dino.collapse_diagnostics, ...
"""
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUITE_ROOT = HERE.parent
DINO_DIR = SUITE_ROOT / "DINO_SARDet"

if str(SUITE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUITE_ROOT))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_dino_train = _load("dino_sardet_train_ssl", DINO_DIR / "train_ssl.py")
while str(DINO_DIR) in sys.path:          # inserted by DINO's train_ssl.py; see the docstring
    sys.path.remove(str(DINO_DIR))

dino = sys.modules["dino"]                 # DINO_SARDet/dino.py
DINOMethod = _dino_train.DINOMethod
DINO_SOURCE = {"dino.py": str(DINO_DIR / "dino.py"), "train_ssl.py": str(DINO_DIR / "train_ssl.py")}
