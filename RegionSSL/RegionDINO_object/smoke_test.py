"""RegionDINO (object only) smoke test: region file, views preview, a 2-epoch run on 48 images, resume, export,
fine-tune from it with the detection suite, probes and panels.

    python RegionSSL/RegionDINO_object/smoke_test.py [--cpu]
Pipeline check only (outputs go to RegionDINO_object/outputs/smoke/); never a result.
Run `python RegionSSL/make_regions.py --smoke` first (it writes regions/regions_smoke.npz for the 48 smoke images).
"""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from rssl import region_engine, regions as R  # noqa: E402
from rssl.region_data import RegionViewDataset, collate_region_views, preview_panel  # noqa: E402
from shared import smoke_common as sc  # noqa: E402  (SARDet suite, read-only)
from shared.ssl_data import load_pool_rows  # noqa: E402

sys.path.insert(0, str(HERE))
from train_ssl import ObjectOnlyMethod as Method  # noqa: E402


def check_region_views(method_cls, n=4):
    sc.step("region file + object views")
    store = R.RegionStore(R.regions_file(smoke=True))
    print(f"  {len(store.names)} images in {store.path.name}, {len(store.boxes)} regions, sha {store.sha256[:12]}")
    rows = load_pool_rows()
    rows = rows[:: max(1, len(rows) // region_engine.SMOKE_IMAGES)][:region_engine.SMOKE_IMAGES]
    cfg = region_engine.read_json(HERE / "config_ssl.json")
    opts = {**cfg["object_views"]}
    opts.pop("_note", None)
    plan = [] if method_cls.mode == "object" else [("global", 1, cfg["dino_views"]["blur_p"][0]), ("global", 1, cfg["dino_views"]["blur_p"][1]),
                                                   ("local", cfg["dino_views"]["local_crops_number"], cfg["dino_views"]["blur_p"][2])]
    ds = RegionViewDataset(rows, plan, store, opts, seed=0)
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
    preview_panel(ds, out / "views_preview.png", n=n)
    print(f"  ok: {len(scene)} scene slots, {len(objects)} object slots x {K} objects; usable regions per image {meta[:, 0].tolist()}, "
          f"fallback {meta[:, 1].tolist()}; preview {out / 'views_preview.png'}")


if __name__ == "__main__":
    device = sc.device_arg()
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    sc.check_paths()
    sc.check_detection_data()
    sc.check_detector(dev)
    check_region_views(Method)
    sc.step(f"{Method.name} SSL smoke run (48 images, 2 epochs)")
    run_dir = region_engine.run(Method, HERE, smoke=True, device=device, log_every_steps=1)
    backbone = next(run_dir.glob("backbone_epoch*.pth"))
    sc.step("resume check (stop after epoch 1, then resume)")
    run2 = region_engine.run(Method, HERE, smoke=True, device=device, stop_after_epoch=1, log_every_steps=0)
    region_engine.run(Method, HERE, resume_dir=run2, device=device, log_every_steps=0)
    assert next(run2.glob("backbone_epoch*.pth")).exists()
    ft = sc.run_finetune_smoke(HERE, Method.name + "_mobilenetv2", backbone, device=device)
    sc.run_analysis_smoke(ft, backbone, device=device)
    print(f"\nSMOKE TEST PASSED ({HERE.name})")
