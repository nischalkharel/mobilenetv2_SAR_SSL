"""SSL on the FAIR-CSAR pools with the RegionSSL engine: plain DINO, object-only RegionDINO, and its random control.

    python FAIR_chips/train_ssl.py --config FAIR_chips/config_ssl_ro_tiles.json            # full run (100 epochs)
    python FAIR_chips/train_ssl.py --config FAIR_chips/config_ssl_dino_down.json --smoke   # 48 images, 2 epochs
    python FAIR_chips/train_ssl.py --resume <run folder>

Configs (six): config_ssl_<dino|ro|ror>_<tiles|down>.json. The "method" key picks the class: regiondino_scene (plain
DINO, scene views only; the engine's reference recipe, as RegionSSL/DINO_reference) or regiondino_object (object-only
views on proposed regions; region_source felz = the brightness rule, random = the size-matched control). The "pool"
key picks fair_tiles or fair_down (RegionSSL/rssl/pools.py). Region files: RegionSSL/make_regions.py --pool fair_tiles
(and fair_down) must have run first; a missing or partial file stops the run.
Exports the teacher backbone in the shared format, so finetune_chips.py (via SARDet_to_ATR/find_backbone.py) loads it.
Run folders: FAIR_chips/outputs/ssl/<stamp>_<run_name>_split<s>_seed<s-1>/
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "RegionSSL"))
from rssl import region_engine  # noqa: E402
from rssl.regiondino import RegionDINOMethod  # noqa: E402


class SceneOnlyMethod(RegionDINOMethod):
    name = "regiondino_scene"
    mode = "scene"


class ObjectOnlyMethod(RegionDINOMethod):
    name = "regiondino_object"
    mode = "object"


CLASSES = {"regiondino_scene": SceneOnlyMethod, "regiondino_object": ObjectOnlyMethod}


def method_class(config_path=None, resume_dir=None):
    p = Path(config_path) if config_path else Path(resume_dir) / "config.json"
    cfg = json.loads(p.read_text(encoding="utf-8"))
    m = cfg.get("method") or cfg.get("method_config", {}).get("method")       # resume: the run's stored config
    if m not in CLASSES:
        raise SystemExit(f"config 'method' must be one of {sorted(CLASSES)}, got {m!r}")
    return CLASSES[m]


if __name__ == "__main__":
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config"); pre.add_argument("--resume")
    a, _ = pre.parse_known_args()
    if not a.config and not a.resume:
        raise SystemExit("give --config <json> or --resume <run folder>")
    region_engine.cli(method_class(a.config, a.resume), HERE)
