# SSL on SARDet-100K scenes: frozen-backbone map focus next to fine-tuned accuracy on ATRNet-STAR

Three-split means. Map columns: the frozen backbone on 2000 ATRNet test chips before any fine-tuning
(SARDet_to_ATR/outputs/compare/response_maps.md; box = the chip's vehicle annotation, area 17.2 % of the chip).
Accuracy columns: the same backbone after supervised fine-tuning on 1 / 3 / 10 % of ATRNet labels, official test set
(SARDet_to_ATR/outputs/compare/cross_dataset.md).

| SSL on SARDet scenes | peak in box, % chips | mass in box, % | acc 1 % | acc 3 % | acc 10 % |
|---|---|---|---|---|---|
| plain DINO | 18 | 19.2 | 17.8 | 41.8 | 72.3 |
| RegionDINO scene+object | 16 | 18.7 | 20.2 | 45.7 | 73.4 |
| scene+object, random windows (control) | 11 | 15.8 | 17.1 | 41.5 | 72.0 |
| RegionDINO object-only | 59 | 26.7 | 27.2 | 49.0 | 73.7 |
| object-only, random windows (control) | 15 | 18.0 | 20.9 | 45.4 | 72.7 |
| LeJEPA | 27 | 21.1 | 12.3 | 32.8 | 64.6 |
| SparK | 15 | 18.8 | 22.7 | 48.0 | 73.9 |
| ImageNet start, no SSL | 47 | 25.7 | 20.7 | 43.4 | 71.7 |

Reading. Within the DINO family (one recipe, one pool, only the views differ) map focus and accuracy move together:
object-only has by far the most vehicle-focused map and the best accuracy; scene+object keeps a plain-DINO map and
gains a third as much; the random-window controls keep plain-DINO maps and gain little or nothing. Across different
methods the map is not an accuracy score: SparK has one of the weakest maps and the second-best accuracy, the
ImageNet start a good map and middling accuracy. The map tells what the object views changed inside DINO; the
accuracy table is still the result.

Per-split numbers: response_maps_per_model.csv and cross_dataset.csv in this folder. Figure with the maps on six
chips: response_maps_split1.png (full size) / response_maps_split1_web.jpg.

## Added 2026-11-20: the same backbones on full SARDet-100K scenes, and the arm-I check

Scene maps (RegionSSL/outputs/scene_maps/scene_maps.md, 1500 test scenes, boxes 3.4 % of a scene): peak in an annotated box, % of scenes,
three splits: ImageNet start 36; plain DINO 8-11; LeJEPA 8-9; SparK 27-33; scene+object 17-21 (control 8-14);
object-only 29-33 (control 15-28). Same ordering inside the DINO family as on the chips; on scenes the ImageNet start
is the most object-focused backbone and every SSL method spreads its response.

Arm I (same engine on the ATRNet chip pool, XI rows): plain DINO 34.0 / 56.2 / 77.3; scene+object 33.8 / 55.8 / 77.1;
object-only 32.0 / 54.5 / 75.7 (controls 33.9 / 54.8 / 77.1 and 30.1 / 53.0 / 76.1). Maps: plain DINO already peaks in
the box on 81-84 % of chips, object-only 43-55 %. Both tables move the same way: on target-centred chips the object views
remove context and cost accuracy; on scenes they add focus and accuracy. The map still reads as a diagnostic, not a score.
