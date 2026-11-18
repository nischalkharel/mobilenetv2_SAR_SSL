"""RegionDINO (object only) smoke test: region file, views preview, a 2-epoch run on 48 images, resume, export,
fine-tune from it with the detection suite, probes and panels.

    python RegionSSL/RegionDINO_object/smoke_test.py [--cpu] [--config config_ssl_atr.json]
Pipeline check only (outputs go to RegionDINO_object/outputs/smoke/); never a result.
Run `python RegionSSL/make_regions.py --smoke` first (regions/regions_smoke.npz for the 48 smoke images); with
--config config_ssl_atr.json run `python RegionSSL/make_regions.py --pool atrnet --smoke` instead (regions_atrnet_smoke.npz).
"""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from rssl import pools, region_engine, regions as R  # noqa: E402
from rssl.region_data import RegionViewDataset, collate_region_views, preview_panel  # noqa: E402
from shared import smoke_common as sc  # noqa: E402  (SARDet suite, read-only)

sys.path.insert(0, str(HERE))
from train_ssl import ObjectOnlyMethod as Method  # noqa: E402


def check_region_views(method_cls, config, n=4):
    sc.step("region file + object views")
    cfg = region_engine.read_json(config)
    pool = pools.get(cfg.get("pool", "sardet"))
    store = R.RegionStore(pool.regions_file(smoke=True), scale=pool.region_scale)
    print(f"  pool {pool.name}: {len(store.names)} images in {store.path.name}, {len(store.boxes)} regions, sha {store.sha256[:12]}")
    rows = pool.rows()
    rows = rows[:: max(1, len(rows) // region_engine.SMOKE_IMAGES)][:region_engine.SMOKE_IMAGES]
    opts = {**cfg["object_views"]}
    opts.pop("_note", None)
    plan = [] if method_cls.mode == "object" else [("global", 1, cfg["dino_views"]["blur_p"][0]), ("global", 1, cfg["dino_views"]["blur_p"][1]),
                                                   ("local", cfg["dino_views"]["local_crops_number"], cfg["dino_views"]["blur_p"][2])]
    ds = RegionViewDataset(rows, plan, store, opts, seed=0, pool=pool)
    (scene, objects, meta), idx = collate_region_views([ds[i] for i in range(n)])
    K = opts["objects_per_image"]
    assert len(objects) == 2 + opts["obj_local_number"], len(objects)
    for v in objects:
        assert v.dtype == torch.uint8 and v.shape[0] == n * K and v.shape[1] == 1, v.shape
    assert objects[0].shape[-1] == opts["obj_global_size"] and objects[-1].shape[-1] == opts["obj_local_size"]
    assert len(scene) == (0 if method_cls.mode == "object" else 2 + cfg["dino_views"]["local_crops_number"])
    # the same item twice gives the same views (seeded per image and epoch): the loader is deterministic
    a, b = ds[0][0][1][0], ds[0][0][1][0]
    assert torch.equal(a, b)
    out = HERE / "outputs" / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    preview_panel(ds, out / f"views_preview_{pool.name}.png", n=n)
    print(f"  ok: {len(scene)} scene slots, {len(objects)} object slots x {K} objects; usable regions per image {meta[:, 0].tolist()}, "
          f"fallback {meta[:, 1].tolist()}; preview {out / f'views_preview_{pool.name}.png'}")


if __name__ == "__main__":
    device = sc.device_arg()
    config = HERE / (sys.argv[sys.argv.index("--config") + 1] if "--config" in sys.argv else "config_ssl.json")
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    sc.check_paths()
    sc.check_detection_data()
    sc.check_detector(dev)
    check_region_views(Method, config)
    sc.step(f"{Method.name} SSL smoke run (48 images, 2 epochs)")
    run_dir = region_engine.run(Method, HERE, config_path=config, smoke=True, device=device, log_every_steps=1)
    backbone = next(run_dir.glob("backbone_epoch*.pth"))
    sc.step("resume check (stop after epoch 1, then resume)")
    run2 = region_engine.run(Method, HERE, config_path=config, smoke=True, device=device, stop_after_epoch=1, log_every_steps=0)
    region_engine.run(Method, HERE, resume_dir=run2, device=device, log_every_steps=0)
    assert next(run2.glob("backbone_epoch*.pth")).exists()
    ft = sc.run_finetune_smoke(HERE, Method.name + "_mobilenetv2", backbone, device=device)
    sc.run_analysis_smoke(ft, backbone, device=device)
    print(f"\nSMOKE TEST PASSED ({HERE.name})")
