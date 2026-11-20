# SARMAE weights (downloaded teacher model, not ours)

| item | value |
|---|---|
| what | SARMAE, a masked-autoencoder ViT pretrained on SAR-1M (1.31 M SAR chips from 18 public datasets, Sentinel-1 / Gaofen-3 / TerraSAR-X / RadarSat-2, 0.1 to 60 m), with speckle-aware masking and an optical-alignment term |
| paper | "SARMAE", arXiv 2512.16635, CVPR 2026 |
| source | https://huggingface.co/Wenquandan777/SARMAE (file `SARMAE_vitb_checkpoint-last`), code https://github.com/mililab/sarmae |
| file here | `SARMAE_vitb_checkpoint-last.pth`, 1,357,698,663 bytes, downloaded 2026-11-20 |
| sha256 | ead8d93aaebeebbeeef43805495071738d9d63c05d7138015fdb1d9957308f6f |
| license | CC BY-NC 4.0 (research use is fine; cite the paper; no commercial use) |
| architecture | ViT-B/16, 224 x 224 input, 768-dim, 12 blocks; the checkpoint is the full pretraining state (encoder + decoder + optimizer) with keys prefixed `sar_encoder.`; only the encoder is used here |
| input expected | grayscale SAR chip, replicated to 3 channels, resized to 224, values in [0, 1] then normalised (their dataset code uses mean 0.5 / std 0.5; their transforms.py has ImageNet statistics; see TeacherRegions/DESIGN.md for which is used and why) |
| used for | TeacherRegions study: frozen, attention / token-norm saliency maps that propose object regions for the compact student (MobileNetV2). Never fine-tuned here, never used at test time |
| caveat | SAR-1M includes SARDet-100K, HRSID, SSDD, MSTAR (unlabeled MIM pretraining). ATRNet-STAR is not in it. So region proposals on SARDet scenes come from a model that has seen those scenes without labels; the ATRNet transfer test is the clean one |

The weights are git-ignored (`*.pth`). On CRC copy this file to `/ix/ageorge/nik96/data/models/SARMAE/` and point
`SARMAE_CHECKPOINT` at it (TeacherRegions/crc/env.sh). The ViT-L file (3.96 GB, `SARMAE_vitl_checkpoint-last`) was not
downloaded; fetch it the same way if ViT-B proposals turn out too coarse.
