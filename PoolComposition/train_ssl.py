"""Plain DINO (the suite recipe, RegionSSL engine in scene mode) on a fixed subset of the SARDet-100K pool.

    python PoolComposition/train_ssl.py --config PoolComposition/config_ssl_land.json      # pool sardet_land, matched steps
    python PoolComposition/train_ssl.py --config PoolComposition/config_ssl_sea.json --smoke
    python PoolComposition/train_ssl.py --resume <run folder>

The method is RegionDINO in mode "scene": no object views, no region file, the same views, head, loss and optimizer
as DINO_SARDet (RegionSSL/DINO_reference says so and the two configs agree key for key), so "XA DINO on SARDet"
(the whole pool, 100 epochs) is the paired reference. Each config names its pool (sardet_land / sea / dense / sparse /
random, RegionSSL/rssl/pools.py), overrides "epochs" so that every subset trains for the full pool's 146,900 steps, and
sets optim.warmup_epochs so the warmup covers the full pool's 14,690 steps; both are checked against data/counts.json
here, so the configs cannot drift from the lists.
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


def check_matched_epochs(argv):
    """A new run's config must ask for the matched epoch and warmup counts recorded by make_pool_lists.py
    (skipped on --resume, which reuses the run's own config, and on --smoke, which runs 2 epochs)."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config", type=Path)
    p.add_argument("--resume", type=Path)
    p.add_argument("--smoke", action="store_true")
    a, _ = p.parse_known_args(argv[1:])
    if a.config is None or a.smoke or a.resume is not None:
        return
    cfg = json.loads(a.config.read_text(encoding="utf-8"))
    counts = json.loads((HERE / "data" / "counts.json").read_text(encoding="utf-8"))
    subset = cfg["pool"].removeprefix("sardet_")
    c = counts["subsets"][subset]
    have_epochs = cfg.get("overrides", {}).get("epochs")
    have_warmup = cfg.get("optim", {}).get("warmup_epochs")
    if have_epochs != c["matched_epochs"] or have_warmup != c["matched_warmup_epochs"]:
        raise SystemExit(f"{cfg['run_name']}: overrides.epochs / optim.warmup_epochs are {have_epochs} / {have_warmup}, but data/counts.json says "
                         f"{c['matched_epochs']} / {c['matched_warmup_epochs']} match the full pool's {counts['target_steps']:,} steps and "
                         f"{counts['target_warmup_steps']:,} warmup steps for pool {cfg['pool']} (re-run make_pool_lists.py or fix the config)")
    print(f"matched steps: {c['matched_epochs']} epochs x {c['steps_per_epoch']} = {c['total_steps']:,} (full pool {counts['target_steps']:,}); "
          f"warmup {c['matched_warmup_epochs']} epochs = {c['warmup_steps']:,} steps (full pool {counts['target_warmup_steps']:,})", flush=True)


if __name__ == "__main__":
    check_matched_epochs(sys.argv)
    region_engine.cli(SceneOnlyMethod, HERE)
