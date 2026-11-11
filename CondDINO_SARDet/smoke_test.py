"""CondDINO_SARDet smoke test: DINO_SARDet's checks plus the per-cell centering checks, two tiny SSL runs, fine-tune, probes.

    python CondDINO_SARDet/smoke_test.py [--cpu]
Pipeline check only (outputs go to CondDINO_SARDet/outputs/smoke/ and outputs/cells/, outputs/cache/); never a result.

Added to DINO's smoke test:
  (a) with one cell CondCenterDINOLoss equals DINOLoss to 1e-6 (loss, cell center, global center; 3 steps)
  (b) the view-major gather is right: with 2+ cells a sample-major gather changes the loss; present cells move by
      0.9 * old + 0.1 * mean of their rows, the absent cell does not; wrong id count rejected; buffers in state_dict
  (c) the perm control keeps the cell-size histogram and scrambles the membership
  (d) cell_report runs for source_pol on the smoke subset (labels read by the report only)
  (e) smoke SSL run of the default variant (source_pol), stop/resume (resume with a changed assignment is rejected),
      fine-tune smoke from the exported teacher, probes + visualisation smoke
  (f) smoke SSL run of the radiometric variant (2 epochs, 48 images): the k-means path and its cache
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from shared import smoke_common as sc  # noqa: E402
from shared import ssl_engine  # noqa: E402
from shared.ssl_data import load_pool_rows  # noqa: E402

sys.path.insert(0, str(HERE))
from dino_base import dino  # noqa: E402
from train_ssl import CondDINOMethod  # noqa: E402
from cond_center import CondCenterDINOLoss, assign_cells, cell_table, permute_ids  # noqa: E402
from cells import VARIANTS, assign_variant, kmeans, radiometric_cache_file  # noqa: E402
import cell_report  # noqa: E402

DINOLoss = dino.DINOLoss


def smoke_rows():
    rows = load_pool_rows()
    return rows[:: max(1, len(rows) // ssl_engine.SMOKE_IMAGES)][: ssl_engine.SMOKE_IMAGES]   # exactly the engine's subset


def check_dino_loss_matches_reference():
    """DINO's own check (as in DINO_SARDet/smoke_test.py): the imported DINOLoss equals a hand-computed cross-entropy."""
    sc.step("DINO loss sanity (imported from DINO_SARDet/dino.py)")
    torch.manual_seed(0)
    loss_fn = DINOLoss(out_dim=16, ncrops=3, warmup_teacher_temp=0.04, teacher_temp=0.04, warmup_teacher_temp_epochs=0, nepochs=1)
    t, s = torch.randn(4, 16), torch.randn(6, 16)
    loss = loss_fn(s, t, 0)
    q = torch.softmax(t / 0.04, -1).chunk(2)
    p = torch.log_softmax(s / 0.1, -1).chunk(3)
    ref = sum((-(q[iq] * p[v]).sum(-1)).mean() for iq in range(2) for v in range(3) if v != iq) / 4
    assert torch.allclose(loss, ref, atol=1e-5), (loss, ref)
    print("  ok")


def check_one_cell_equals_dino():
    sc.step("(a) one cell: CondCenterDINOLoss == DINOLoss (3 steps)")
    torch.manual_seed(3)
    nc, D = 6, 64
    l_ref = DINOLoss(D, nc, 0.04, 0.04, 0, 3)
    l_one = CondCenterDINOLoss(D, nc, 0.04, 0.04, 0, 3, n_cells=1)
    worst = 0.0
    for epoch in range(3):
        s_out, t_out = torch.randn(nc * 4, D), torch.randn(2 * 4, D)
        a, b = l_ref(s_out, t_out, epoch), l_one(s_out, t_out, epoch, torch.zeros(4, dtype=torch.long))
        worst = max(worst, abs(a.item() - b.item()), (l_ref.center - l_one.cell_center).abs().max().item(),
                    (l_ref.center - l_one.center).abs().max().item())
    assert worst < 1e-6, worst
    print(f"  ok: max difference {worst:.1e}")


def check_view_major_gather():
    sc.step("(b) per-cell centering: view-major gather, EMA of present cells only")
    torch.manual_seed(4)
    nc, D = 6, 64
    pairs = [(iq, v) for iq in range(2) for v in range(nc) if v != iq]
    l3 = CondCenterDINOLoss(D, nc, 0.04, 0.04, 0, 3, n_cells=3, cell_momentum=0.9)
    l3(torch.randn(nc * 4, D), torch.randn(2 * 4, D), 0, torch.tensor([0, 1, 2, 1]))   # warm-up: three non-zero centers
    assert (l3.cell_center.norm(dim=1) > 0).all() and not torch.allclose(l3.cell_center[0], l3.cell_center[1])
    cells3 = torch.tensor([0, 2, 2, 0])
    s_out, t_out = torch.randn(nc * 4, D), torch.randn(2 * 4, D)
    before = l3.cell_center.clone()
    loss3 = l3(s_out, t_out, 0, cells3)
    ids_rows = torch.cat([cells3, cells3])                       # view-major: view 1 of every image, then view 2
    expected = before.clone()
    for c in (0, 2):
        expected[c] = 0.9 * before[c] + 0.1 * t_out[ids_rows == c].mean(0)
    assert torch.allclose(l3.cell_center, expected, atol=1e-6), "present cells must move to 0.9 old + 0.1 batch mean"
    assert torch.equal(l3.cell_center[1], before[1]), "the absent cell must not move"
    assert l3.cell_count.tolist() == [6, 4, 6], l3.cell_count
    s3 = (s_out / 0.1).chunk(nc)
    q_ok = F.softmax((t_out - before[ids_rows]) / 0.04, -1).chunk(2)
    manual = sum((-q_ok[iq] * F.log_softmax(s3[v], -1)).sum(-1).mean() for iq, v in pairs) / len(pairs)
    q_wrong = F.softmax((t_out - before[torch.repeat_interleave(cells3, 2)]) / 0.04, -1).chunk(2)   # sample-major: wrong
    manual_wrong = sum((-q_wrong[iq] * F.log_softmax(s3[v], -1)).sum(-1).mean() for iq, v in pairs) / len(pairs)
    assert torch.allclose(loss3, manual, atol=1e-6), (loss3, manual)
    assert not torch.allclose(loss3, manual_wrong, atol=1e-6), "a sample-major gather must change the loss"
    try:
        l3(s_out, t_out, 0, torch.tensor([0, 1, 2]))
        raise AssertionError("wrong number of cell ids was accepted")
    except ValueError:
        pass
    sd = l3.state_dict()
    assert {"center", "cell_center", "cell_count"} <= set(sd), sd.keys()
    print(f"  ok: loss {loss3.item():.4f} (sample-major gather would give {manual_wrong.item():.4f})")


def check_perm_control(rows):
    sc.step("(c) perm control keeps the cell-size histogram")
    ids, names = assign_cells(rows, VARIANTS["source_pol"]["fields"])
    ids2, names2 = assign_cells(rows, VARIANTS["source_pol"]["fields"])
    assert np.array_equal(ids, ids2) and names == names2, "assignment must be deterministic"
    assert ids.min() == 0 and ids.max() == len(names) - 1 and len(ids) == len(rows)
    perm = permute_ids(ids, 0)
    assert np.array_equal(np.bincount(perm, minlength=len(names)), np.bincount(ids, minlength=len(names)))
    assert np.array_equal(permute_ids(ids, 0), perm) and not np.array_equal(permute_ids(ids, 1), perm)
    pids, pnames, pinfo = assign_variant("perm", rows, permute_seed=0, verbose=False)
    assert np.array_equal(pids, perm) and pnames == names and pinfo["permute_seed"] == 0
    moved = float((perm != ids).mean())
    assert moved > 0.3, f"only {moved:.2f} of the images changed cell; the smoke subset is {len(rows)} images"
    table = cell_table(ids, names)
    print(f"  ok: {len(names)} cells on {len(rows)} images; {100 * moved:.0f} % of the images changed cell under the permutation; "
          f"sizes {[n for _, _, n in table]}")


def check_cell_report(rows):
    sc.step("(d) cell_report for source_pol on the smoke subset (labels read by the report only)")
    per_cell, s = cell_report.report("source_pol", rows, verbose=False)
    out = Path(ssl_engine.paths.experiment_output_dir(HERE)) / "smoke" / "cell_report"
    lines = cell_report.write_report("source_pol", per_cell, s, out, f"source_pol_smoke{len(rows)}")
    assert sum(c["num_images"] for c in per_cell) == len(rows)
    assert s["num_cells"] == len({c["cell"] for c in per_cell})
    mi = s["mutual_information_bits"]
    assert 0 <= mi["cell_dominant"] <= mi["dominant_entropy"] + 1e-9 and 0 <= mi["cell_classset"] <= mi["classset_entropy"] + 1e-9
    for f in ("csv", "md", "json"):
        assert (out / f"source_pol_smoke{len(rows)}.{f}").exists()
    print("  ok: " + lines[4].lstrip("- ") + "\n      " + lines[6].lstrip("- "))


def check_kmeans():
    sc.step("radiometric k-means sanity (own implementation, fixed seed)")
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(c, 0.05, (40, 33)) for c in (0.0, 1.0, 2.0, 3.0)])
    a1, c1, *_ = kmeans(x, 4, seed=0)
    a2, c2, *_ = kmeans(x, 4, seed=0)
    assert np.array_equal(a1, a2) and np.allclose(c1, c2), "k-means must be deterministic for a fixed seed"
    truth = np.repeat(np.arange(4), 40)
    purity = sum(np.bincount(truth[a1 == j]).max() for j in range(4)) / len(x)
    assert purity == 1.0, purity
    print("  ok: four separable blobs recovered exactly; deterministic")


if __name__ == "__main__":
    device = sc.device_arg()
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    sc.check_paths()
    sc.check_detection_data()
    sc.check_detector(dev)
    check_dino_loss_matches_reference()
    check_one_cell_equals_dino()
    check_view_major_gather()
    rows = smoke_rows()
    check_perm_control(rows)
    check_cell_report(rows)
    check_kmeans()
    sc.check_ssl_views([("global", 1, 1.0), ("global", 1, 0.1), ("local", 8, 0.5)])

    sc.step("(e) CondDINO SSL smoke run, source_pol cells (48 images, 2 epochs)")
    run_dir = ssl_engine.run(CondDINOMethod, HERE, smoke=True, device=device, log_every_steps=1)
    backbone = next(run_dir.glob("backbone_epoch*.pth"))
    meta = torch.load(backbone, map_location="cpu", weights_only=True)
    ex = meta["exported"]
    assert ex["cell_variant"] == "source_pol" and ex["label_use"] == VARIANTS["source_pol"]["label_use"] and Path(ex["cells_csv"]).exists()
    assert Path(ex["center_stats_csv"]).exists(), ex["center_stats_csv"]
    hist = (run_dir / "history.csv").read_text(encoding="utf-8").splitlines()
    assert "cells_seen" in hist[0].split(","), hist[0]
    print(f"  ok: exported {backbone.name}; cells {ex['n_cells']}, assignment {ex['cell_assignment_sha256'][:12]}..., cells.csv {ex['cells_csv']}")

    sc.step("resume check (stop after epoch 1, then resume; a changed assignment is rejected)")
    run2 = ssl_engine.run(CondDINOMethod, HERE, smoke=True, device=device, stop_after_epoch=1, log_every_steps=0)
    ckpt = torch.load(run2 / "checkpoint.pth", map_location="cpu", weights_only=False)
    assert "cell_center" in ckpt["method"]["loss"] and ckpt["method"]["cell_variant"] == "source_pol"
    tampered = dict(ckpt["method"]) | {"cell_assignment_sha256": "0" * 64}
    args2 = ssl_engine.flat_args(ssl_engine.read_json(run2 / "config.json"))
    args2.log_every_steps = 0
    m = CondDINOMethod(args2)
    m.build(args2, rows, len(rows) // args2.batch_size, torch.device(dev))
    try:
        m.load_state_dict(tampered)
        raise AssertionError("a changed cell assignment was accepted on resume")
    except ValueError:
        pass
    del m
    ssl_engine.run(CondDINOMethod, HERE, resume_dir=run2, device=device, log_every_steps=0)
    assert next(run2.glob("backbone_epoch*.pth")).exists()
    print("  ok")

    ft = sc.run_finetune_smoke(HERE, "condcenter_mobilenetv2", backbone, device=device)
    sc.run_analysis_smoke(ft, backbone, device=device)

    sc.step("(f) CondDINO SSL smoke run, radiometric cells (48 images, 2 epochs; k-means path + cache)")
    cache = radiometric_cache_file(rows)
    run3 = ssl_engine.run(CondDINOMethod, HERE, config_path=HERE / "config_ssl_radiometric.json", smoke=True, device=device, log_every_steps=0)
    assert cache.exists(), f"radiometric cache missing: {cache}"
    ex3 = torch.load(next(run3.glob("backbone_epoch*.pth")), map_location="cpu", weights_only=True)["exported"]
    assert ex3["cell_variant"] == "radiometric" and ex3["n_cells"] == 16 and ex3["label_use"] == VARIANTS["radiometric"]["label_use"]
    ids_a, _, info_a = assign_variant("radiometric", rows, verbose=False)      # second call: from the cache
    assert info_a["from_cache"] and info_a["cell_assignment_sha256"] == ex3["cell_assignment_sha256"]
    print(f"  ok: 16 k-means cells, sizes {info_a['cluster_sizes']}; cache {cache.name}")
    print("\nSMOKE TEST PASSED (CondDINO_SARDet)")
