# PoolComposition: does what the scene pool contains set the sign of plain scene SSL?

Status 2026-11-26: built, smoke-tested on the PC (lists, SSL on three subset pools, fine-tune naming), reviewed by
an agent (five documentation / design items, all applied below), registered in the transfer table and the tracker; not
run on CRC.

## 1. The question

Tracker c03 said "SSL on SARDet-100K scenes transfers to ATRNet chips worse than no SSL". Tracker c13 (2026-11-26)
showed the opposite from a second scene archive: plain DINO on FAIR-CSAR tiles transfers +1.8 / +4.1 / +2.4 above no
SSL. So the sign is a property of the pool. `RegionSSL/pool_hit_rate.py` named a candidate cause, measured with the
method's own windows and the method's own label-free bright regions: on SARDet 27 % of DINO's global views and 42 %
of its local views hold no bright content at all, driven by the ship-at-sea sources (Shipdataset 45 / 65 %, HRSID
65 / 81 %), while the land sources sit near 1 to 20 %; on the FAIR tiles the rates are 7 / 15 % (48 smoke tiles; the
full file is on CRC). Annotated objects point the other way (32 % of SARDet global views hold an object, 11 % of
FAIR's), so plain DINO seems to need structure in its views, not objects.

This study tests the cause inside one pool, with everything else held fixed: plain DINO trained on parts of the
SARDet pool that differ in how much structure a random view sees, each for the same number of steps as the full-pool
run, each tested on ATRNet chips exactly as the XA rows.

Prediction if the explanation is right: land and dense transfer at or above no SSL (and above the whole-pool DINO row,
17.8 / 41.8 / 72.3), sea and sparse transfer below it. If all four land near the whole-pool row, the view content is
not the cause and the FAIR difference must come from elsewhere (resolution, sensor, image repetition).

## 2. The five subsets (`make_pool_lists.py`, outputs in `data/`, committed)

| subset | scenes | rule | bright area median / mean | steps per epoch | matched epochs (warmup) |
|---|---|---|---|---|---|
| dense | 47,017 | bright-area fraction above the pool median (15.1 %) | 48.5 % / 51.0 % | 734 | 200 (146,800 steps; 20) |
| sparse | 47,017 | bright-area fraction at or below the median | 2.7 % / 4.1 % | 734 | 200 (146,800 steps; 20) |
| random | 47,017 | a random half of the pool (seed 0): the size control | about the pool's 15 % / 27 % | 734 | 200 (146,800 steps; 20) |
| land | 57,036 | source in OGSOD, MSAR, SAR-AIRcraft, SIVED, SADD | 34.0 % / 36.6 % | 891 | 165 (147,015 steps; 16) |
| sea | 36,998 | source in Shipdataset, HRSID, SSDD, AIR_SARShip | 3.1 % / 13.6 % | 578 | 254 (146,812 steps; 25) |

The primary test is dense vs sparse vs random: three pools of exactly 47,017 scenes, so image count and passes per
image are identical and only the content differs; the random half has the whole pool's content mix, so it also
shows what "fewer images, more passes" does on its own. Land vs sea is the interpretable pair a reader pictures
("ships on water versus airports and cities") but it changes image count as well as content, and sensor and
resolution travel with it (MSAR is HISEA-1, OGSOD and SAR-AIRcraft are Gaofen-3, Shipdataset is half Sentinel-1).

Bright-area fraction = share of the 512-px frame covered by the brightness-rule regions with extent >= 12 px (the
regions RegionDINO object-only trains on), from `RegionSSL/regions/regions.npz`. Land / sea read the pool manifest's
source column (metadata, no labels); dense / sparse read only the region file (pixels only); random reads nothing.
Dense / sparse keeps every source in both halves, but not in equal shares (Shipdataset 7,649 dense vs 24,085 sparse,
OGSOD 14,097 vs 567), so sensor and resolution still move with the split; it isolates the view content better than
land / sea, not perfectly. A per-source stratified median split would be the next refinement if the result asks for
it. `data/scene_bright_area.csv` holds the per-scene numbers for figures.

What DINO's own views see on each subset (`RegionSSL/pool_hit_rate.py --pool sardet_<subset>`, outputs in
`RegionSSL/outputs/pool_hit_rate_sardet_<subset>.json`; background-only = less than 1 % of the window bright):

| pool | global view background-only | local view background-only | near-empty scenes | global views with an annotated object |
|---|---|---|---|---|
| whole pool | 27.0 % | 42.3 % | 9.5 % | 32 % |
| land | 14.4 % | 27.0 % | 4.3 % | 35 % |
| sea | 46.2 % | 65.6 % | 17.5 % | 30 % |
| dense | 2.5 % | 11.0 % | 0.0 % | 38 % |
| sparse | 51.4 % | 73.5 % | 19.0 % | 27 % |
| random half | 27.0 % | 42.3 % | 9.4 % | 33 % |
| FAIR-CSAR tiles (48 smoke tiles) | 7.3 % | 15.1 % | 0.0 % | 11 % |

So dense is the SARDet part that looks like FAIR in this respect and sparse is the opposite; the object column barely
moves, which is the point: the subsets differ in structure per view, not in objects per view.

Matched steps: the whole-pool DINO run did 100 epochs x 1,469 = 146,900 steps with a 10-epoch (14,690-step) lr
warmup; each subset gets round(146,900 / steps_per_epoch) epochs and round(14,690 / steps_per_epoch) warmup epochs,
both recorded in `data/counts.json` and checked by `train_ssl.py` against the config (a config that disagrees refuses
to start). The cosine schedule then covers the same number of steps everywhere. One schedule item is not matched:
`freeze_last_layer` is 1 epoch, so the head's last layer stays frozen for 578 to 891 steps instead of 1,469; at
this scale that is cosmetic. Caveat, as in the FAIR 500-epoch runs: a smaller pool means more passes over each
image (land 165, sea 254, halves 200, whole pool 100); the random half measures that effect on its own.

## 3. Method and pairing

`train_ssl.py` runs RegionDINO in mode "scene" with the RegionSSL engine: whole-scene views only, global 224 px and
local 96 px with zoom 0.75 to 1.33, 8 local crops, the DINO head, loss and optimizer of `SARDet/DINO_SARDet/config_ssl.json`
(the configs agree key for key; `RegionSSL/DINO_reference` records that this mode is DINO_SARDet). The paired
reference is therefore the existing "XA DINO on SARDet" row (whole pool, 100 epochs), and `SARDet_to_ATR/compare.py`
pairs the four new rows against it per split. No new baseline run is needed. The pools are `sardet_land / sea /
dense / sparse / random` (`RegionSSL/rssl/pools.py`, `SARDetSubsetPool`): the same images, frame and exclusions as
`sardet` (the audited SSL pool: official train scenes only, as `shared/manifests.quick_audit` asserts; val and test
never read), only the scene list differs; the run's manifest records the list file and its hash.

Fine-tune: `SARDet_to_ATR/finetune_atr.py --method dino_<subset>` on the same ATRNet labeled sets and splits as every
XA row (run names `XA_dino_<subset>_sardetssl_mobilenetv2`; the run record names the subset). Three splits, three
budgets. The SARDet chip test and detection are not planned for these rows (c09: the chip test cannot separate
methods; detection would need the detector fine-tune and is a different question).

## 4. Cost and order

15 SSL chains of about 7 GPU-h each (the three whole-pool DINO_SARDet runs record 7.0 to 7.2 GPU-h for the same
146,900 steps; one slower node took 23 h), about 105 GPU-h in all, then 15 fine-tune jobs of about 20 minutes.
Order: dense, sparse and random first (the clean test, 9 chains, about 63 GPU-h), land and sea next (6 chains, about
42 GPU-h); both pairs are wanted for the paper figure. With three splits and about 1 point of split noise, the sign
of an effect and differences of 2 points or more are detectable; finer differences are not.

## 5. Where things are

See WHAT_IS_THIS.txt. Run folders `outputs/ssl/<stamp>_dino_<subset>_ssl_mobilenetv2_split<s>_seed<s-1>/`; fine-tunes
in `SARDet_to_ATR/outputs/finetune/<stamp>_XA_dino_<subset>_sardetssl_mobilenetv2[_labeled_set_<b>pct][_split<s>]_seed0/`.
