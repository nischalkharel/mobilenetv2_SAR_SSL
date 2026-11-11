# Frozen-feature probes: class versus recording condition, per backbone

Percent. Class LP 1 %: linear probe on the 652 labeled chips, scored on the full official test set (29,284);
NCM 1 %: nearest class mean of the same chips. Class LP 150/class: probe trained on a pool sample (150 per class),
scored on a test sample (100 per class). 1-NN: nearest-neighbour agreement on the test sample, own acquisition
group excluded. dec.: condition decoded by a probe trained on the pool sample, scored on the test sample.

Chance for 1-NN (sum of squared frequencies): class 2.5, scene 33.3, az_sector 8.4, depression 25.0, band 59.0.
Majority baseline for dec.: scene 44.5, az_sector 8.9, depression 26.4, band 71.2.

| Backbone | class LP 1 % | class NCM 1 % | class LP 150/class | 1-NN class | 1-NN scene | 1-NN az sector | 1-NN depression | dec. scene | dec. az sector | dec. depression | dec. band |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ImageNet | 10.2 | 8.4 | 18.9 | 12.3 | 73.5 | 29.2 | 55.0 | 83.0 | 35.4 | 63.4 | 82.9 |
| A1 | 30.3 | 18.0 | 56.1 | 32.4 | 91.3 | 57.8 | 74.8 | 96.4 | 66.3 | 86.6 | 92.7 |
| P4 | 40.0 | 28.0 | 63.4 | 46.7 | 90.0 | 40.8 | 69.5 | 95.2 | 56.9 | 83.6 | 88.4 |
| P4perm | 30.3 | 17.9 | 57.0 | 32.2 | 91.1 | 57.0 | 75.1 | 96.5 | 66.0 | 85.7 | 92.5 |
| P4d | 31.3 | 18.4 | 55.8 | 33.1 | 88.9 | 57.4 | 67.4 | 95.8 | 66.8 | 83.3 | 90.3 |

Backbones: ImageNet = IMAGENET1K_V2; A1 = backbone_teacher_epoch100.pth (20261018-192425_A1_dino_ssl_mobilenetv2_seed0); P4 = backbone_teacher_epoch100.pth (20261110-194252_P4_dino_ssl_mobilenetv2_seed0); P4perm = backbone_teacher_epoch100.pth (20261110-204010_P4perm_dino_ssl_mobilenetv2_seed0); P4d = backbone_teacher_epoch100.pth (20261110-150743_P4d_dino_ssl_mobilenetv2_seed0)
Written 2026-11-11T11:01:04.
