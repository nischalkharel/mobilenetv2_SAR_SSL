"""DINOSAR's "ratio-gradient content map", re-implemented in numpy so it can run as a CPU proposer.

Source: DINOSAR/dinosar/content_aware_crop.py of github.com/kenanking/DINOSAR (Tang et al., Int. J. Digital Earth,
accepted 2026-11-04), read on 2026-11-25. Their pipeline, step by step, and what is kept here:

  1. per-tile min-max stretch of the grayscale tile to [0, 1]                       kept
  2. bilinear downsample by 4 (coarse size max(32, side // 4))                      kept (DS = 4)
  3. for each window size r in (3, 7): a stride-1 box mean M_r (reflect padding);  kept (SCALES = (3, 7))
     neighbours shifted by r // 2 pixels: gx = log(M_r shifted right) - log(M_r shifted left),
     gy = log(M_r shifted down) - log(M_r shifted up); magnitude_r = sqrt(gx^2 + gy^2) * M_r
     (the mean-map weighting "suppresses gradient response in dark speckle regions")
  4. L2 fusion over the scales, then min-max normalisation to [0, 1]                 kept
  5. anchors (local maxima >= 0.9 quantile, NMS 11, salience floor 3 x mean, at most 4) and the crop sampler
     (global views containing the anchors, 256 local candidates weighted by content^1.5)   NOT used

Step 5 is DINOSAR's view sampler, which carries random crop sizes. Our RegionDINO recipe needs object-sized boxes,
so the content map is turned into regions exactly as the SARMAE attention map was (TeacherRegions): top q % of the
map, connected components, one box per component. That keeps the comparison "same map-to-regions rule, different
map": brightness-rule blobs vs SARMAE attention vs DINOSAR's ratio-gradient content.

Deviations from the original worth stating: the stretch divides by (max - min) rather than by max; the map is
computed with scipy (uniform_filter, mode "mirror" = torch "reflect") and upsampled back to the pool frame with PIL
bilinear before thresholding. No labels are read anywhere.
"""
import numpy as np
from PIL import Image
from scipy import ndimage

EPS = 1e-6
SCALES = (3, 7)          # DINOSAR coarse_scales
DS = 4                   # DINOSAR ds


def ratio_gradient_map(x: np.ndarray, scales=SCALES, eps=EPS) -> np.ndarray:
    """x: float32 (H, W) in [0, 1] at the coarse resolution -> content map (H, W) in [0, 1]."""
    H, W = x.shape
    mags = []
    for r in scales:
        mean_map = ndimage.uniform_filter(x, size=r, mode="mirror")          # torch F.pad reflect + avg_pool2d(r, stride 1)
        off = r // 2
        mp = np.pad(mean_map, off, mode="reflect")
        left = mp[off:off + H, 2 * off:2 * off + W]                           # DINOSAR's names; 'left' is the +off column
        right = mp[off:off + H, :W]
        top = mp[2 * off:2 * off + H, off:off + W]
        bottom = mp[:H, off:off + W]
        gx = np.log(left + eps) - np.log(right + eps)
        gy = np.log(top + eps) - np.log(bottom + eps)
        mags.append(np.sqrt(gx * gx + gy * gy) * mean_map)
    t = np.sqrt(sum(m * m for m in mags))                                     # L2 over scales
    lo, hi = float(t.min()), float(t.max())
    return ((t - lo) / (hi - lo + eps)).astype(np.float32)


def content_map(gray: Image.Image, scales=SCALES, ds=DS) -> np.ndarray:
    """PIL grayscale image at the pool's proposal frame -> float32 (H, W) content map in [0, 1] at that frame."""
    W, H = gray.size
    a = np.asarray(gray, dtype=np.float32) / 255.0
    lo, hi = float(a.min()), float(a.max())
    a = (a - lo) / (hi - lo + EPS)                                             # per-tile stretch (normalize_amplitude)
    ch, cw = max(32, H // ds), max(32, W // ds)
    coarse = np.asarray(Image.fromarray(a).resize((cw, ch), Image.BILINEAR), dtype=np.float32)
    m = ratio_gradient_map(coarse, scales)
    up = np.asarray(Image.fromarray(m).resize((W, H), Image.BILINEAR), dtype=np.float32)
    return np.clip(up, 0.0, 1.0)


PARAMETERS = {"source": "DINOSAR content_aware_crop.py (github.com/kenanking/DINOSAR), numpy port", "scales": list(SCALES),
              "downsample": DS, "shift_px": [r // 2 for r in SCALES], "weighting": "box-mean map of the same scale",
              "fusion": "L2 over scales, min-max to [0, 1]", "stretch": "per tile, (x - min) / (max - min)"}
