"""RegionSSL's own package (named rssl so it never collides with the detection suite's `shared`).

It imports the detection suite's shared code (SARDet/shared) READ-ONLY and the DINO components
(SARDet/DINO_SARDet/dino.py) READ-ONLY; nothing under SARDet/ is modified by this study.
"""
import sys
from pathlib import Path

REGIONSSL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = REGIONSSL_ROOT.parent
SARDET_ROOT = REPO_ROOT / "SARDet"
for p in (SARDET_ROOT, SARDET_ROOT / "DINO_SARDet", REGIONSSL_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
