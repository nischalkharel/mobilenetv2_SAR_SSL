"""Frozen SARMAE teacher -> saliency map per pool image -> label-free region boxes in RegionDINO's file format.

    python TeacherRegions/teacher_saliency.py --limit 300 --diagnose --preview 24      # PC check: regions + box agreement + pictures
    python TeacherRegions/teacher_saliency.py --smoke                                    # the engine's 48 smoke images -> regions_smoke.npz
    python TeacherRegions/teacher_saliency.py                                            # the whole SARDet pool (GPU job on CRC, about 1 h)
    python TeacherRegions/teacher_saliency.py --zoom 1 --top-q 5 --limit 300 --diagnose  # other settings get their own output folder
Defaults (zoom 2, top 2 %) were chosen on 300 scenes against the annotation boxes (DESIGN.md section 3).

Teacher: SARMAE ViT-B/16 (models/SARMAE/, see its README), frozen. A scene (shorter side 512, as the student sees it)
is covered by a grid of 224/zoom-px windows (stride <= 144/zoom, origins on the map grid; zoom 2 by default, so the
teacher sees 112-px native windows upsampled to 224); each window gives a 14 x 14 map; windows are averaged where
they overlap into a map with 16/zoom native px per cell, then upsampled. Two signals: "cls" = last-block attention from the
CLS token to the patches, mean over heads; "norm" = L2 norm of the final patch tokens.
Regions: the map's top q % pixels -> connected components -> boxes; components under min_area px are dropped; at
most max_regions per image, by saliency mass. Saved with RegionSSL.rssl.regions.save (names, offsets, boxes,
areas + summary json), into TeacherRegions/outputs/regions_<signal>/regions[_smoke|_limitN].npz. Point
REGIONSSL_REGIONS_DIR at that folder and RegionDINO reads it unchanged. No labels are read to make the file; the
--diagnose option reads annotation boxes only to MEASURE the proposals (DESIGN.md section 3).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy import ndimage
from torch import nn

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "RegionSSL"))
sys.path.insert(0, str(REPO / "SARDet"))
from rssl import pools, regions as R  # noqa: E402
from rssl.region_data import region_box_diagnostic  # noqa: E402
from shared.ssl_engine import SMOKE_IMAGES  # noqa: E402

DEFAULT_CKPT = Path(os.environ.get("SARMAE_CHECKPOINT", REPO / "models" / "SARMAE" / "SARMAE_vitb_checkpoint-last.pth"))   # CRC: set in crc/env.sh
MEAN, STD = 0.5, 0.5          # SARMAE's SAR pretraining data path (HF repo SARMAE_Pretrain/dataset/mixed_dataset.py, the enable_sar_noise
                              # branch used by the released run: gray -> 3 channels, [0,1], transforms.Normalize([0.5]*3, [0.5]*3));
                              # their transforms.py (vendored) has ImageNet statistics, which that path does not use
WIN, STRIDE_MAX, PATCH = 224, 144, 16


# ---------------------------------------------------------------- the teacher (ViT-B/16 encoder, timm key names) --
class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, keep_attn=False):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        attn = (q @ k.transpose(-2, -1)) * (C // self.heads) ** -0.5
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out), (attn if keep_attn else None)


class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim, eps=1e-6), nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, heads)
        self.mlp = nn.Sequential()
        self.mlp.fc1, self.mlp.act, self.mlp.fc2 = nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)

    def forward(self, x, keep_attn=False):
        a, attn = self.attn(self.norm1(x), keep_attn)
        x = x + a
        return x + self.mlp(self.norm2(x)), attn


class ViTEncoder(nn.Module):
    """SARMAE's MAE encoder (ViT-B/16, 224, 12 blocks, 12 heads): patch embed + fixed sincos pos embed + CLS + blocks + norm."""

    def __init__(self, dim=768, depth=12, heads=12):
        super().__init__()
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, dim, PATCH, PATCH)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, (WIN // PATCH) ** 2 + 1, dim))
        self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        """x (B,3,224,224) -> tokens (B,197,768) after the final norm, and the last block's attention (B,heads,197,197)."""
        x = self.patch_embed.proj(x).flatten(2).transpose(1, 2) + self.pos_embed[:, 1:]
        x = torch.cat([(self.cls_token + self.pos_embed[:, :1]).expand(len(x), -1, -1), x], 1)
        attn = None
        for i, blk in enumerate(self.blocks):
            x, a = blk(x, keep_attn=(i == len(self.blocks) - 1))
            attn = a if a is not None else attn
        return self.norm(x), attn


def load_teacher(ckpt=DEFAULT_CKPT, device="cuda"):
    import argparse as _ap
    with torch.serialization.safe_globals([_ap.Namespace]):          # the file holds its training args; nothing else non-tensor
        ck = torch.load(ckpt, map_location="cpu", weights_only=True)
    sd = {k[len("sar_encoder."):]: v for k, v in ck["model"].items() if k.startswith("sar_encoder.") and "decoder" not in k and k != "sar_encoder.mask_token"}
    m = ViTEncoder()
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"teacher weights do not match the encoder: missing {missing[:5]} unexpected {unexpected[:5]}")
    return m.to(device).eval(), {"file": str(ckpt), "sha256": R.sha256_of_file(Path(ckpt)), "epoch": ck.get("epoch"), "tensors": len(sd)}


# ---------------------------------------------------------------- saliency of one scene -----------------------------
def window_origins(L, win=WIN, stride_max=STRIDE_MAX, cell=PATCH):
    """Start positions of win-px windows covering a side of length L with stride <= stride_max (one window if L <= win),
    snapped DOWN to multiples of cell so every window's map lands exactly on the scene's map grid (no sub-cell shift)."""
    if L <= win:
        return [0]
    n = int(np.ceil((L - win) / stride_max)) + 1
    return sorted({int(round(i * (L - win) / (n - 1) / cell)) * cell for i in range(n)})   # L and win are multiples of cell


@torch.no_grad()
def saliency(model, gray: Image.Image, device, zoom=1, batch=64):
    """gray PIL (W,H <= 512) -> {"cls": map, "norm": map}, float32 (H, W) in [0,1] for the whole scene.
    zoom z: the teacher looks at windows of 224/z native px upsampled to 224, so its 16-px patch covers 16/z native px
    (z = 2 makes a 10-px ship about a patch wide). Window stride in native px scales the same way."""
    W, H = gray.size
    a = np.asarray(gray, dtype=np.float32) / 255.0
    win, stride, cell = WIN // zoom, STRIDE_MAX // zoom, PATCH // zoom
    Wp, Hp = int(np.ceil(W / cell)) * cell, int(np.ceil(H / cell)) * cell          # whole map cells: no unpainted edge strip
    if (Wp, Hp) != (W, H):
        a = np.pad(a, ((0, Hp - H), (0, Wp - W)), mode="edge")
    xs, ys = window_origins(Wp, win, stride, cell), window_origins(Hp, win, stride, cell)
    tiles, pos = [], []
    for y0 in ys:
        for x0 in xs:
            t = a[y0:y0 + win, x0:x0 + win]
            if t.shape != (win, win):                     # scene smaller than a window: pad with the edge value
                t = np.pad(t, ((0, win - t.shape[0]), (0, win - t.shape[1])), mode="edge")
            if zoom != 1:
                t = np.asarray(Image.fromarray(t).resize((WIN, WIN), Image.BILINEAR), dtype=np.float32)
            tiles.append(t); pos.append((x0, y0))
    x = torch.from_numpy(np.stack(tiles))[:, None].repeat(1, 3, 1, 1).to(device)
    x = (x - MEAN) / STD
    cls_l, nrm_l = [], []
    for i in range(0, len(x), batch):
        with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
            tok, attn = model(x[i:i + batch])
        cls_l.append(attn[:, :, 0, 1:].float().mean(1)); nrm_l.append(tok[:, 1:].float().norm(dim=-1))
    g = WIN // PATCH
    cls = torch.cat(cls_l).reshape(-1, g, g)                                   # CLS -> patches, mean over heads
    nrm = torch.cat(nrm_l).reshape(-1, g, g)                                   # token norm
    out = {}                                                                   # cell = native px per map cell (set above)
    for name, maps in (("cls", cls), ("norm", nrm)):
        canvas = np.zeros((Hp // cell, Wp // cell), np.float32)
        count = np.zeros_like(canvas)
        for m, (x0, y0) in zip(maps.cpu().numpy(), pos):
            cy, cx = y0 // cell, x0 // cell
            h, w = min(g, canvas.shape[0] - cy), min(g, canvas.shape[1] - cx)
            canvas[cy:cy + h, cx:cx + w] += m[:h, :w]; count[cy:cy + h, cx:cx + w] += 1
        if (count == 0).any():
            raise RuntimeError("unpainted map cells (window grid bug)")
        canvas /= count
        up = np.asarray(Image.fromarray(canvas).resize((Wp, Hp), Image.BILINEAR), dtype=np.float32)[:H, :W]
        lo, hi = up.min(), up.max()
        out[name] = (up - lo) / (hi - lo) if hi > lo else np.zeros_like(up)
    return out


def regions_from_map(m, top_q=2.0, min_area=12, max_regions=20):
    """top q % pixels -> connected components -> boxes (x1,y1,x2,y2) int16 and areas int32, by saliency mass, largest first."""
    thr = np.quantile(m, 1 - top_q / 100)
    lab, n = ndimage.label(m >= thr)
    if n == 0:
        return np.zeros((0, 4), np.int16), np.zeros((0,), np.int32)
    idx = np.arange(1, n + 1)
    areas = ndimage.sum_labels(np.ones_like(lab), lab, idx).astype(np.int32)
    mass = ndimage.sum_labels(m, lab, idx)
    sl = ndimage.find_objects(lab)
    boxes = np.array([[s[1].start, s[0].start, s[1].stop, s[0].stop] for s in sl], np.int32)
    keep = areas >= min_area
    boxes, areas, mass = boxes[keep], areas[keep], mass[keep]
    order = np.argsort(-mass, kind="stable")[:max_regions]
    boxes, areas = boxes[order], areas[order]
    order = np.argsort(-areas, kind="stable")                    # the store convention: largest area first
    return boxes[order].astype(np.int16), areas[order]


# ---------------------------------------------------------------- previews and the box check --------------------------
def preview(pool, rows, stores, out_png, cols=4):
    """Scenes with the regions of each store drawn (one colour per store) and the GT boxes in white."""
    colors = ["#ff4d4d", "#4da6ff", "#ffd24d"]
    tiles = []
    for r in rows:
        im = pool.load_for_proposal(r).convert("RGB"); d = ImageDraw.Draw(im)
        for x1, y1, x2, y2 in pool.gt_boxes(r):
            d.rectangle([x1, y1, x2, y2], outline="white", width=2)
        for (name, store), c in zip(stores.items(), colors):
            for x1, y1, x2, y2 in store.get(r["file_name"]):
                d.rectangle([int(x1), int(y1), int(x2), int(y2)], outline=c, width=1)
        d.text((4, 4), f"{r.get('source', pool.name)} {r['file_name']}", fill="white")   # FAIR rows carry no 'source'; the pool name stands in
        tiles.append(im.resize((384, int(384 * im.size[1] / im.size[0]))))
    h = max(t.size[1] for t in tiles); rows_n = int(np.ceil(len(tiles) / cols))
    sheet = Image.new("RGB", (cols * 388, rows_n * (h + 4) + 24), "black")
    ImageDraw.Draw(sheet).text((4, 4), "white = annotation boxes (measure only); " + "; ".join(f"{c} = {n}" for (n, _), c in zip(stores.items(), colors)), fill="white")
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * 388, 24 + (i // cols) * (h + 4)))
    sheet.save(out_png)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default="sardet", choices=sorted(pools.POOLS))
    ap.add_argument("--signal", default="cls", choices=["cls", "norm"], help="which saliency makes the region file (both are measured by --diagnose)")
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--top-q", type=float, default=2.0, help="top q %% of the map becomes region candidates (default chosen by the 300-scene check, DESIGN.md 3)")
    ap.add_argument("--zoom", type=int, default=2, choices=[1, 2, 4], help="teacher sees 224/zoom native px per window (2: a 16-px patch = 8 native px; the default)")
    ap.add_argument("--min-area", type=int, default=12); ap.add_argument("--max-regions", type=int, default=20)
    ap.add_argument("--out-dir", type=Path, help="default TeacherRegions/outputs/regions_<signal>")
    ap.add_argument("--diagnose", action="store_true", help="box agreement of brightness-rule vs cls vs norm regions on the processed images (reads annotations only to measure)")
    ap.add_argument("--preview", type=int, default=0, help="draw this many scenes with all region sets")
    a = ap.parse_args()
    pool = pools.get(a.pool)
    rows = pool.rows()
    if a.smoke:
        rows = rows[:: max(1, len(rows) // SMOKE_IMAGES)][:SMOKE_IMAGES]
    elif a.limit:
        rows = rows[: a.limit]
    elif hasattr(pool, "all_rows"):
        rows = pool.all_rows()
    out_dir = a.out_dir or (HERE / "outputs" / (f"regions_{a.signal}" + (f"_z{a.zoom}" if a.zoom != 2 else "") + (f"_q{a.top_q:g}" if a.top_q != 2 else "")))
    stem = pool.regions_file(a.smoke).stem + (f"_limit{a.limit}" if a.limit else "")   # regions[_smoke] for sardet, regions_fair_tiles[_smoke] for the FAIR pool
    model, tinfo = load_teacher(a.ckpt, a.device)
    print(f"teacher: ViT-B/16 from {tinfo['file']} (epoch {tinfo['epoch']}, {tinfo['tensors']} tensors, sha256 {tinfo['sha256'][:12]}) on {a.device}\n"
          f"{len(rows)} {pool.name} images -> {out_dir / (stem + '.npz')}  (signal {a.signal}; top {a.top_q} %, min area {a.min_area}, max {a.max_regions} regions)", flush=True)
    per = {"cls": {}, "norm": {}}
    t0 = time.time()
    for k, r in enumerate(rows, 1):
        maps = saliency(model, pool.load_for_proposal(r), a.device, a.zoom)
        for s in per:
            per[s][r["file_name"]] = regions_from_map(maps[s], a.top_q, a.min_area, a.max_regions)
        if k % 500 == 0 or k == len(rows):
            print(f"  {k}/{len(rows)}  {k / (time.time() - t0):.1f} images/s", flush=True)
    extra = {"proposer": f"sarmae_vitb_{a.signal}_attention", "proposer_parameters": {"zoom": a.zoom, "top_q_pct": a.top_q, "min_area": a.min_area, "max_regions": a.max_regions, "window": WIN, "stride_max": STRIDE_MAX},
             "teacher": tinfo, "signal": a.signal, "top_q_pct": a.top_q, "min_area": a.min_area, "max_regions": a.max_regions,
             "window": WIN, "stride_max": STRIDE_MAX, "zoom": a.zoom, "normalisation": f"gray x3, [0,1], mean {MEAN} std {STD}", "pool": pool.name}
    summaries = {}
    for s in per:                                                   # both signals are written; the chosen one under the plain name
        d = out_dir if s == a.signal else out_dir.with_name(out_dir.name.replace(f"regions_{a.signal}", f"regions_{s}"))
        summaries[s] = R.save(d / f"{stem}.npz", per[s], {**extra, "signal": s}, input_size=pool.propose_size)
        print(f"  wrote {d / (stem + '.npz')}: {summaries[s]['regions_total']} regions, per image mean {summaries[s]['regions_per_image']['mean']:.1f}, extent quantiles {summaries[s]['extent_px_quantiles']}")
    if a.diagnose or a.preview:
        stores = {}
        felz = REPO / "RegionSSL" / "regions" / pool.regions_file(a.smoke).name   # the brightness-rule file of this pool, whatever REGIONSSL_REGIONS_DIR says
        if felz.exists():
            stores["brightness rule"] = R.RegionStore(felz, scale=pool.region_scale)
        for s in per:
            d = out_dir if s == a.signal else out_dir.with_name(out_dir.name.replace(f"regions_{a.signal}", f"regions_{s}"))
            stores[f"teacher {s}"] = R.RegionStore(d / f"{stem}.npz", scale=pool.region_scale)
        if a.diagnose:
            res = {name: region_box_diagnostic(rows, st, 12, n_images=min(300, len(rows)), pool=pool) for name, st in stores.items()}
            print("\nbox agreement on the same images (annotations used only to measure):")
            print(f"{'proposer':18} {'GT boxes with a region':>24} {'region centres in a box':>24} {'regions/image':>14} {'images w/o':>11}")
            for name, v in res.items():
                print(f"{name:18} {100 * v['gt_boxes_with_a_region']:23.1f} % {100 * v['region_centre_in_a_box']:23.1f} % {v['usable_regions_per_image']:14.1f} {100 * v['images_without_regions']:10.1f} %")
            (out_dir / f"{stem}_box_agreement.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        if a.preview:
            preview(pool, rows[: a.preview], stores, out_dir / f"{stem}_preview.png")
            print(f"preview: {out_dir / (stem + '_preview.png')}")


if __name__ == "__main__":
    main()
