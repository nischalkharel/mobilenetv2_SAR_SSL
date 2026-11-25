# GradientRegions: DINOSAR's content map as a third region proposer

Status 2026-11-25: built, smoke-tested on the PC (proposer on the SARDet and FAIR tiles pools, SSL smoke for the
gradient row and its control on both pools), registered in every table; not run on CRC.

## 1. Why

The novelty check (`reports/SAR SSL novelty check.md`) found that DINOSAR (Tang et al., International Journal of
Digital Earth, accepted 2026-11-04; github.com/kenanking/DINOSAR) chooses DINO's views on SAR tiles from a
label-free "ratio-gradient content map" and shows that this beats random crops (ATRNet-STAR SOC-40 k-NN 11.78 to
19.76 with ViT-S, 5 epochs; SARDet-100K 54.22 to 55.28 AP). Our RegionDINO places views on bright-region boxes from
a pixel rule. A reviewer will ask: is the brightness rule just a worse version of their content map, or does the
view policy (object-sized windows, dropped scene views, size-matched random control) carry the result whatever the
map? This study answers it by running their map through our pipeline with everything else unchanged.

The three proposers then differ only in the map that is thresholded into regions:

| proposer | map | study |
|---|---|---|
| brightness rule | Felzenszwalb segments on brightness (connected bright blobs) | RegionSSL |
| teacher | frozen SARMAE ViT-B CLS attention | TeacherRegions |
| gradient content | DINOSAR's ratio-gradient content map | this folder |

Each has a size-matched random-window control (random places, the proposer's own window sizes).

## 2. The map (content_map.py)

Ported line by line from `DINOSAR/dinosar/content_aware_crop.py` (read 2026-11-25) into numpy so it runs as a CPU
proposer: per-tile stretch to [0, 1]; bilinear downsample by 4 (coarse side max(32, side / 4)); for window sizes
r in (3, 7): box mean M_r (reflect padding), gx = log M_r(+r/2 right) - log M_r(r/2 left), gy likewise vertically,
magnitude_r = sqrt(gx^2 + gy^2) * M_r; L2 over the two scales; min-max to [0, 1]; upsample back to the pool frame.
Stated deviations: the stretch divides by (max - min) rather than by max; scipy's "mirror" padding stands in for
torch "reflect"; the map is upsampled to the frame before thresholding. Not used: DINOSAR's anchor picker (local
maxima above the 0.9 quantile, at most 4) and its crop sampler, because those produce random crop sizes and our
recipe sizes the windows from the region's own extent. Regions follow the TeacherRegions rule instead: top 2 % of
the map, connected components, min area 12 px, at most 20 boxes per image by map mass.

## 3. What the proposer finds (annotations read only to measure)

300 SARDet-100K scenes (`outputs/regions_sardet/regions_limit300_box_agreement.json`):

| proposer | annotated boxes reached | region centres on an object | regions per scene | extent median (95 %) |
|---|---|---|---|---|
| brightness rule | 61.7 % | 7.0 % | 27.5 | 10 px (70) |
| SARMAE teacher (c11) | 57 % | 12 % | 17.5 | 16 px (44) |
| gradient content | 59.6 % | 23.9 % | 8.2 | 21 px (104) |

48 FAIR-CSAR tiles (smoke): reached 38.9 % (rule 22.2 %), centres on an object 6.7 % (rule 1.5 %), 13.1 regions per
tile, extent median 23 px. So the gradient map is the most precise object finder of the three and has the longest
size tail. If it still ties the brightness rule after SSL, that is the fourth proposer-independent result and the
"coarse objectness is enough" reading stands against the strongest label-free proposer in the literature.

## 4. Rows and cost

SARDet pool (100 epochs, 146,900 steps, as every other SARDet SSL run): `regiondino_o_gradient` and
`regiondino_o_gradient_random`, 3 splits, about 5.4 GPU-h each (32 GPU-h); tested on ATRNet transfer
(XA rows, paired against the brightness-rule row), SARDet chips, and optionally detection (`SARDet/crc/finetune.sbatch
../GradientRegions <split> auto <run_name>`). FAIR tiles (500 epochs, 148,500 steps): `fair_rog_tiles500` and
`fair_rogr_tiles500`, about 6 GPU-h each (36 GPU-h); tested on FAIR chips and, through the XF rows, on ATRNet chips
and SARDet chips. Proposer jobs: CPU, well under an hour each.

## 5. Where things are

See WHAT_IS_THIS.txt. Run folders: `outputs/ssl/<stamp>_regiondino_o_gradient[_random]_ssl_mobilenetv2_split<s>_seed<s-1>/`
(SARDet pool) and `FAIR_chips/outputs/ssl/<stamp>_fair_rog[r]_tiles500_ssl_mobilenetv2_split<s>_seed<s-1>/`.
