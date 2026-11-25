# SAR self-supervised learning and foundation models: cited inventory (2019 to Nov 2026)

Scope note: ~45 tool calls (web search, page fetch, and local text extraction of four arXiv/MDPI PDFs with `pdftotext`). Numbers below are self-reported by each paper unless stated. Where a fetch truncated a page I say so. "Not found" items list the queries tried.

## Key question 1: Inventory of SAR SSL / foundation-model papers (method, pretraining data, backbone, downstream task, label budgets, gain over ImageNet or supervised baseline)

### Takeaway
Every major SAR SSL paper pretrains on a pool of **object chips and detection tiles** (MSTAR, SAMPLE, SARSim, OpenSARShip, SAR-Ship, HRSID, SSDD, MSAR, SAR-AIRcraft, OGSOD...), uses a **ViT-B / HiViT-B / iTPN-B or larger backbone (66M to 300M+ params)**, and evaluates with **N-shot (1 to 40 shots) or 30 % label budgets on MSTAR / FUSAR-Ship / SAR-ACD**, or full-label detection on SARDet-100K / SSDD. None of them uses a MobileNet-class backbone or 1 / 3 / 10 % label fractions, and most report that in-domain SAR SSL beats ImageNet initialization when the SSL pool is dominated by chips.

### Cited Findings

**SAR-JEPA (Li et al., arXiv 2311.15153, v1 Nov 2023, v6 Sep 2024; later published in ISPRS J. Photogramm. per the SARATR-X lineage; venue not printed on the arXiv page)**
- Method: LoMaR-style local masked modelling, predicts multi-scale gradient-by-ratio (GR) features with kernels r = 5, 9, 13, 17 instead of pixels; ViT-Base encoder, 4x V100, 200 epochs. — [arXiv 2311.15153v6](https://arxiv.org/html/2311.15153v6)
- Pretraining pool: MSAR 28,499 + SAR-Ship 39,729 + SARSim 21,168 + SAMPLE 5,380 = ~94.8k "image slices" (paper rounds to ~100k); all are target chips / detection tiles, not full scenes. — [arXiv 2311.15153v6](https://arxiv.org/html/2311.15153v6)
- Downstream (fine-tuning, 10/20/40-shot, avg of 10 runs): MSTAR ImageNet 16.7/32.0/40.7, MAE 50.6/61.3/69.5, FG-MAE 50.0/58.7/70.0, SAR-JEPA 70.3/82.1/91.6; FUSAR-Ship ImageNet 54.0/57.0/59.7, SAR-JEPA 81.3/83.3/85.8; SAR-ACD ImageNet 22.6/26.8/42.7, SAR-JEPA 54.8/62.6/75.5. — [arXiv 2311.15153v6](https://arxiv.org/html/2311.15153v6)
- Linear probing on FUSAR-Ship (10/20/40-shot): ImageNet-supervised ViT 74.6/79.6/83.6 **beats every SAR-chip-pretrained MIM baseline except SAR-JEPA** (MAE 74.2/78.1/80.3, FG-MAE 69.7/74.3/77.7, I-JEPA 36.5/42.2/46.8, LoMaR 70.5/77.9/80.7; SAR-JEPA 80.6/82.7/85.2). The authors do not call this negative transfer. — [arXiv 2311.15153v6](https://arxiv.org/html/2311.15153v6)
- Views: random local windows + ResizedCrop/flip/contrast; no target-aware cropping. — [arXiv 2311.15153v6](https://arxiv.org/html/2311.15153v6)

**SARATR-X v1 (Li et al., arXiv 2405.09365, v1 May 2024, v5 22 Jan 2025; arXiv page prints no venue, IEEE TPAMI could not be confirmed from the page)**
- Pool "SARDet-180K": 14 datasets, 186,600 samples (MSTAR 14,577; MiniSAR 3,927; SARSim 21,168; SAMPLE 5,380; SIVED 1,044; OpenSARShip 26,679; SAR-Ship 39,729; AIR-SARShip 801; HRSID 5,604; SSDD 1,160; SADD 883; SAR-AIRcraft 18,818; MSAR 28,499; OGSOD 18,331). Images > 1000 px were sliced; FUSAR-Ship and LS-SSDD were excluded to limit sea-scene imbalance. — [arXiv 2405.09365v5](https://arxiv.org/html/2405.09365v5)
- Backbone HiViT-B, 66M params; two-step SSL: MIM on ImageNet (open weights) then MAE-style MIM on SAR predicting multi-scale gradient features (MGF, r = 9, 13, 17); 200 epochs on 8x RTX3090 (600 epochs for final model). — [arXiv 2405.09365v5](https://arxiv.org/html/2405.09365v5)
- MSTAR SOC 10-way few-shot: 1/2/5/10/20/25-shot = 85.2/91.4/95.9/97.7/98.1/98.5 vs BIDFC (SSL, 2022) 80.7/85.3/90.3 at 1/2/5-shot. — [arXiv 2405.09365v5](https://arxiv.org/html/2405.09365v5)
- SAR-VSA 25-class linear probe (5/10/20-shot): ViT SL-ImageNet 58.6/65.7/74.2; ViT SSL-ImageNet MAE 50.7/58.0/65.5; ViT SSL-SAR MAE 54.1/61.5/68.2; HiViT SSL-SAR MAE 64.9/72.7/79.9; HiViT SSL-ImageNet&SAR MAE 71.5/78.5/84.0; HiViT SSL-ImageNet&SAR MGF 76.5/80.8/85.1. **For ViT at 5-shot, supervised-ImageNet (58.6) beat SAR-only MAE (54.1).** — [arXiv 2405.09365v5](https://arxiv.org/html/2405.09365v5)
- Detection mAP: SARDet-100K 57.3 (MSFA 56.4, ConvNeXt 55.1); OGSOD 52.0; SSDD 67.5; SAR-AIRcraft 58.7. No ImageNet-init detection baseline printed in the paper. — [arXiv 2405.09365v5](https://arxiv.org/html/2405.09365v5)
- Label budgets: only N-shot; no 1/5/10 % fractions. No saliency/target-aware cropping; "automatic target slicing" listed as future work. — [arXiv 2405.09365v5](https://arxiv.org/html/2405.09365v5)

**SARATR-X-v2 (Li, Song, Liu, Peng, Zhou, Xia, Yang, Liu, Liu, Liu; arXiv 2607.23238, v4 stamp 26 Sep 2026; "in preparation for submission", no venue)**
- Method: scale-aware structural MIM; six fixed structural extractors (3x3 blind-spot aggregator + five directional log-ratio operators, radii 3/5/9/13/17) fused by learned softmax weights; iTPN-B / iTPN-L encoder with FPN, initialised from ImageNet-1K iTPN; mask ratio 0.75; 1200 epochs on 8x A800. — [arXiv 2607.23238](https://arxiv.org/html/2607.23238)
- Pool: ~320k unlabeled single-channel SAR images from 17 datasets = SARATR-X 14 (test splits removed) + M4-SAR + FAIR-CSAR + **ATRNet-STAR SOC-40 train (68,091)**; per-dataset counts include M4-SAR 56,116, SAR-Ship 35,757, MSAR 28,638, OpenSARShip 26,677, FAIR-CSAR 23,825, MSTAR 12,092, HRSID 4,623, SSDD 1,044, SADD 839; inputs are random-resized crops to 224 of 128 to 1024 px tiles (mostly chips/tiles). — [arXiv 2607.23238](https://arxiv.org/html/2607.23238)
- ATRNet-STAR SOC-50 (linear probe) table: ViT-B 59.2; HDANet 63.7; ConvNeXt-B 81.6; SARATR-X HiViT-B 85.2; SARMAE ViT-L 81.4; AFRL-DINOv2 ViT-B 94.9; SARATR-X-v2 iTPN-B 97.8; iTPN-L 98.4 (+3.5). MSTAR SOC: iTPN-L 98.7. — [arXiv 2607.23238 (pdftotext of PDF)](https://arxiv.org/pdf/2607.23238)
- SARDet-100K HBB mAP/mAP50/mAP75: MSFA ConvNeXt-B 56.4/88.2/61.5; SARATR-X 57.2/88.5/62.6; SUMMIT 57.0/89.9/62.7; SARMAE ViT-L 63.1; SAMBA 63.3/89.1/68.0; SARATR-X-v2 iTPN-B 63.4; iTPN-L 68.7/93.9/75.3. HRSID: iTPN-L 73.5 mAP, 94.0 mAP50. FUSAR-Ship 30 % kNN: SUMMIT 71.9; SARMAE 92.9; SAMBA 85.0; v2 iTPN-B 93.0. — [arXiv 2607.23238 (PDF)](https://arxiv.org/pdf/2607.23238)
- Label budgets: 1-shot MSTAR ablation, 10-shot linear probe on ATRNet-STAR SOC-50 used for a "drift analysis" (the worst pretraining target gives 32.9 % at 10-shot). No compact backbone, no negative-transfer statement, masking is random (no CFAR/saliency). — [arXiv 2607.23238](https://arxiv.org/html/2607.23238)

**SARMAE (Liu, Wang, Wang, Chen, Jiang, Cheng, Guo, Cui, Zhang; arXiv 2512.16635, 18 Dec 2025; CVPR 2026)**
- CVPR 2026 acceptance: [CVF open access](https://openaccess.thecvf.com/content/CVPR2026/html/Liu_SARMAE_Masked_Autoencoder_for_SAR_Representation_Learning_CVPR_2026_paper.html); code and ViT-B/ViT-L weights at [github.com/MiliLab/SARMAE](https://github.com/mililab/sarmae).
- SAR-1M: 1,312,902 SAR images from 18 public datasets (MSTAR, SEN1-2, HRSID, SARDet-100K, RSAR, AIR-PolSAR-Seg...), ~1.04M paired with optical, 57 categories, 9 to 2048 px, 0.1 to 60 m, Sentinel-1/Gaofen-3/RadarSat-2/TerraSAR-X. Method: MAE + speckle-noise injection branch (SARE) + semantic-anchor constraint to a frozen DINOv3 optical teacher (SARC). — [arXiv 2512.16635v1](https://arxiv.org/html/2512.16635v1)
- Classification: FUSAR-Ship 40-shot / 30 %: SAR-JEPA 85.8 / -, SUMMIT - / 71.91, SARMAE ViT-B 89.30 / 92.92; MSTAR 40-shot / 30 %: SAR-JEPA 91.6 / -, SUMMIT - / 98.39, SARMAE-B 96.70 / 99.61; SAR-ACD 30 %: SUMMIT 84.25, SARMAE-B 95.06. — [arXiv 2512.16635v1](https://arxiv.org/html/2512.16635v1)
- Detection mAP SARDet-100K / SSDD: ImageNet baseline 52.30 / 66.40; SARATR-X 57.30 / 67.50; SARMAE-B 57.90 / 68.10; SARMAE-L 63.10 / 69.30. — [arXiv 2512.16635v1](https://arxiv.org/html/2512.16635v1)
- Ablation (ViT-B, FUSAR / SSDD / AIR-PolSAR-Seg): MAE ImageNet-1K 75.40/64.00/60.28; MAE MillionAID 80.16/63.60/62.29; **DINOv3 74.25/61.60/58.42 (below the ImageNet-MAE baseline)**; MAE on SAR-1M 82.22/64.20/64.36; +SARE 86.80/64.40/65.15; full 89.30/68.10/66.53. — [arXiv 2512.16635v1](https://arxiv.org/html/2512.16635v1)
- Label budgets: 40-shot and 30 % only; no 1 / 10 %. No distillation to a small model; no negative-transfer discussion. — [arXiv 2512.16635v1](https://arxiv.org/html/2512.16635v1)

**SUMMIT (Int. J. Applied Earth Observation and Geoinformation, 2025)**
- ViT-based MIM with auxiliary self-supervised tasks, pretrained on MuSID (>560k SAR images; a composite of HRSID, SSDD, FUSAR, SRSDD...). — [SUMMIT record](https://library.kab.ac.ug/Record/doaj-art-65c6c4310ff94207a5dccaa1b021a527/Description); [HF README](https://huggingface.co/DYunt/SUMMIT/blob/main/README.md)
- Numbers as reported by competitors: FUSAR 30 % 71.91, MSTAR 30 % 98.39, SAR-ACD 30 % 84.25 — [SARMAE](https://arxiv.org/html/2512.16635v1); SARDet-100K 57.0/89.9/62.7, SSDD 67.4/97.3 — [SARATR-X-v2 PDF](https://arxiv.org/pdf/2607.23238), [SARDet-MIM](https://mdpi-res.com/d_attachment/remotesensing/remotesensing-18-00580/article_deploy/remotesensing-18-00580.pdf).

**SAMBA (Wang, Pan, Gu, Ai, Xu, Zhao, Xiao; arXiv 2606.31668, v1 30 Jun 2026, v2 7 Jul 2026; no venue)**
- Mamba encoder ~27M params, ~8 GFLOPs; two-stage ImageNet then "186K SAR" (SARATR-X pool); scatter-guided masking (see Q3). — [arXiv 2606.31668 html](https://arxiv.org/html/2606.31668)
- MSTAR 1/2/5-shot 89.6/93.5/95.0 (SARATR-X 85.2/91.4/95.9); FUSAR-Ship 30 % 85.0 (SUMMIT 71.9, ResNet-50 58.4); SAR-ACD 30 % 81.1 (**below SUMMIT 84.3**); SARDet-100K 59.7/89.1/68.0; SSDD 71.3/97.8/85.6; SIVED 67.6; SAR-AIRcraft 60.0. ATRNet-STAR not used. — [arXiv 2606.31668 html](https://arxiv.org/html/2606.31668)

**XuanJian-DINOSAR (Wu et al., arXiv 2611.06105, 4 Nov 2026; "in preparation for submission")**
- DINOv2 ViT-B/14 (optical init) adapted with DINO + iBOT + KoLeo + a local-consistency branch (SAR intensity / Gamma-speckle perturbations, patch-token alignment at the same location); 2 global + 8 local random multi-crop views, no target-aware crop selection. — [arXiv 2611.06105 abs](https://arxiv.org/abs/2611.06105); [PDF via pdftotext](https://arxiv.org/pdf/2611.06105)
- Pool: 367,285 images = FUSAR-Ship, SAMPLE, SAR-AIRcraft-1.0, MSTAR, SIVED, FAIR-CSAR, RSAR, and the **ATRNet-STAR SOC-40 training split** ("target chips, detection images, and scene images"). — [arXiv 2611.06105 PDF](https://arxiv.org/pdf/2611.06105)
- ATRNet-STAR SOC-50 (Table IV): ViT-B 59.20; HDANet 63.70; SARMAE ViT-L 81.40; ConvNeXt-B 81.60; SARATR-X HiViT-B 85.20; AFRL-DINOv2 94.90; SARATR-X-v2 iTPN-B 97.80 / iTPN-L 98.40; DINOSAR 96.70. Table XI: official optical DINOv2 91.76 vs SAR-domain DINOv2 (no LC) 94.01 vs full 96.70 on SOC-50 (fine-tuned with LoRA). EOC group accuracies: cross-scene 35.04 to 49.28; depression 30/45/60 deg = 79.93/74.36/59.63. — [arXiv 2611.06105 PDF](https://arxiv.org/pdf/2611.06105)
- Detection: RSAR 39.90/74.30/37.10 (RoI-Trans ResNet-50 baseline 35.02/66.95/32.65; SARATR-X-v2 iTPN-L 38.70/77.30/34.20); **FAIR-CSAR AP50: RoI-Transformer ResNet-50 38.00, SSFG-Net 42.60, DINOSAR 42.40**. — [arXiv 2611.06105 PDF](https://arxiv.org/pdf/2611.06105)

**AFRL-DINOv2 / "On the Status of Foundation Models for SAR Imagery" (Inkawhich, AFRL; arXiv 2509.21722, 26 Sep 2025, no venue)**
- DINOv2 ViT-B (86M) self-supervised finetuned 100 epochs on p1 = ~110k SAR images from nine public datasets (SAR-Ships, OpenEarthMap, HRSID, FUSAR, SSDD, DualPolShips, SRSDD, CVDome, SARSIM); p3 adds the ATRNet-SOC40 train split (~68k). — [arXiv 2509.21722v1](https://arxiv.org/html/2509.21722v1)
- NN balanced accuracy averaged over 10 chip tasks (MSTAR SOC/EOC, SAMPLE, 7 ATRNet-STAR tasks): ResNet50 ImageNet-sup 26.8; DINOv2-B 29.5; DINOv3-L-Web 30.8; **DINOv3-L-Sat 25.6 (satellite pretraining worse than web)**; SARATR-X 44.6; SAR-SimCLR (11M) 60.3; AFRL-DINOv2-p1 61.6; p3/p4 79.6. — [arXiv 2509.21722v1](https://arxiv.org/html/2509.21722v1)
- Low-shot LoRA on ATRNet-SOC40 (hub DINOv2 vs AFRL-DINOv2-p1): 10-shot 14.8 vs 22.1; 100-shot 61.5 vs 60.5; 250-shot 82.7 vs 79.6; 500-shot 91.8 vs 90.1; all 98.8 vs 98.6 — **the web-pretrained model beats the SAR-finetuned one from 100 shots upward**. — [arXiv 2509.21722v1](https://arxiv.org/html/2509.21722v1)

**SAR-SimCLR / "Global Model Approach to Robust Few-Shot SAR ATR" (Inkawhich, arXiv 2303.10800, 2023)**
- SimCLR with a 7-transform SAR augmentation set; ResNet-18 backbone (the 2025 AFRL paper lists it at 11M params); ~100k unlabeled chips cut from full frames of SAR-Ships, HRSID, FUSAR, SSDD, LS-SSDD, Dual-Pol Ships, SRSDD, CVDome; ~90 % contain ship-like targets, none are MSTAR vehicles. — [ar5iv 2303.10800](https://ar5iv.labs.arxiv.org/html/2303.10800)
- MSTAR SOC 10-way 1/5-shot 28.8/60.5 vs best from-scratch 20.9/47.4; 2-way 10-shot 92.2 vs 84.3; EOC 4-way 5/25-shot 65.3/81.0 vs 56.0/76.3; no ImageNet baseline anywhere. — [ar5iv 2303.10800](https://ar5iv.labs.arxiv.org/html/2303.10800)

**BIDFC (Zhai et al., arXiv 2408.03627, 2024; "weakly contrastive" batch instance discrimination + feature clustering)**
- 91.25 % on MSTAR when fine-tuned on 3.13 % of training data; also tested on OpenSARShip; ResNet-18 backbone per SARATR-X-v2's table (90.3 at MSTAR 5-shot). — [arXiv 2408.03627](https://arxiv.org/abs/2408.03627v1); [SARATR-X-v2 PDF](https://arxiv.org/pdf/2607.23238)

**Muzeau et al. (SONDRA/ONERA, arXiv 2502.01162, Feb 2025, eess.SP)**
- DINO-style contrastive ViT-T/8 (256 prototypes, 600 epochs) trained on 199,040 patches 100x100 px from ONERA's airborne SETHI sensor (20 cm and 1 m), **never sees MSTAR**; kNN on MSTAR: 74 % with 1 label/class, 95.9 % with 10, 99.1 % with 100; ResNet-34 from scratch 52.2 % at 10 labels/class. — [arXiv 2502.01162 (pdftotext)](https://arxiv.org/pdf/2502.01162)

**PCM-SAR (Central South Univ., arXiv 2504.09502, 13 Apr 2025, no venue)**
- Contrastive learning with GLCM-based speckle sample generation, "semantic-aware local sampling" and mutual-learning fusion; ResNet-18/34/50; OpenSARShip and FUSAR-Ship only; kNN / LP: FUSAR baseline 77.45/75.32 to PCM-SAR 88.23/88.48; OpenSARShip 68.06/69.45 to 72.10/72.48; semi-supervised FUSAR 10 % / 30 % labels: ResNet-18 58.91 / 71.80, ResNet-34 59.28 / 75.03; no ImageNet-supervised baseline. — [arXiv 2504.09502v1](https://arxiv.org/html/2504.09502v1)

**FG-MAE (Wang, Hernandez, Albrecht, Zhu; arXiv 2310.18653 Oct 2023; IEEE JSTARS vol. 18, pp. 321-336, 2025)**
- MAE reconstructing HOG for SAR (HOG + NDI for multispectral); SSL4EO-S12 Sentinel-1 scene pretraining; up to +5 % over MAE on EuroSAT-SAR; scales to ViT-H. Land-cover, not ATR. — [DLR copy](https://elib.dlr.de/202303/1/2310.18653v1.pdf); [GitHub](https://github.com/zhu-xlab/fgmae)

**SAR-HUB (Yang, Kang, Liu, Liu, Huang; Remote Sensing 15(23):5534, 2023)**
- Supervised (not SSL) pretraining of CNNs and ViTs on large open SAR scene-classification datasets; seven downstream SAR tasks; claims SAR-pretrained beats ImageNet and optical-RS pretraining and explains why with SAR-KP knowledge points. — [MDPI](https://mdpi.com/2072-4292/15/23/5534); [GitHub](https://github.com/XAI4SAR/SAR-HUB)

**MAE on SARDet-100K for detection (Pu & Xu, Fudan; arXiv 2501.11249, Jan 2025, no venue)**
- ViT-B MAE pretrained on the 94,493 SARDet-100K training tiles (256x256), ViTDet + Faster R-CNN; SARDet-100K val mAP: scratch 47.2, ImageNet-MAE 53.6, SARDet-100K-MAE 54.9 (+1.3). No label-fraction runs, random masking. — [arXiv 2501.11249v1](https://arxiv.org/html/2501.11249v1)

**SARDet-MIM / SSMAE (Remote Sensing 18(4):580, accepted 12 Feb 2026)**
- MAE reconstructing multi-scale gradient + SAR-Harris response; pretrained on MSAR, SAR-Ship, SARSim, SAMPLE (the SAR-JEPA pool); "Maximally Pre-trained Detector" transfers ViT encoder and decoder. SSDD mAP/mAP50/mAP75: ViTDet 64.2/95.8; ViTDet4SAR 67.6/97.9; SARATR-X 66.2/91.4; SUMMIT 67.4/97.3; SARMAE 70.4/96.7/80.1; SARDet-MIM 72.1/98.2/87.5. SADD 55.3/95.9/59.6; SAR-AIRcraft 60.5/87.0/70.7. — [Remote Sensing PDF (pdftotext)](https://mdpi-res.com/d_attachment/remotesensing/remotesensing-18-00580/article_deploy/remotesensing-18-00580.pdf)

**SparK-SAR (Natsume, Yun, Yeo; arXiv 2610.21993, 15 Oct 2026; extends SPIE Asia-Pacific RS 2026)**
- SparK sparse-conv MIM on ConvNeXt-RS (27.8M params) over SSL4EO-S12 v1.1 Sentinel-1 GRD scenes; Copernicus-Bench gains vs no-pretraining: EuroSAT-S1 +4.7, BigEarthNet-S1 +5.0, DFC2020-S1 +5.8, Flood-S1 +1.4; beats frozen Copernicus-FM ViT (139M) on 3/4 tasks; no ImageNet baseline, no ATR. — [arXiv 2610.21993](https://arxiv.org/abs/2610.21993)

**DI3CL (Ren et al., arXiv 2511.07808, Nov 2025; accepted IEEE TIP)**
- Contrastive land-cover foundation model on SARSense (460,532 SAR images); land-cover mapping, water, roads; no ATR, no numbers on the abstract page. — [arXiv 2511.07808](https://arxiv.org/abs/2511.07808)

**Hierarchical multi-task pretext SSL (Siam, Noor, Ndoye, Khan; Sensors 26(1):122, 24 Dec 2025)**
- Nine pretext tasks on a 4-layer CNN, pretrained and evaluated only on 1,345 measured SAMPLE chips; downstream heads include MobileNet v1 (width 1.0/0.75/0.5/0.25) and EfficientNet B0-B3; k = 0.05 to 1.00 of 806 training chips; MobileNet v1-1.0 98.70 % at k = 1 and 33.77 % at k = 0.05; SimCLR ResNet-18 baseline 92.02 % / 15.77 %; **no ImageNet-pretrained baseline in the text read (first 100k chars)**. — [PMC12787659](https://pmc.ncbi.nlm.nih.gov/articles/PMC12787659/)

**Older contrastive SAR ATR work (2020 to 2023)**
- Pei et al., IEEE JSTARS 2023: two-stage CL on MSTAR, transfers to OpenSARShip. — [DOAJ](https://doaj.org/article/890344cb9692442193b83827e6b6dc8d)
- Wang et al., LPPCL, Remote Sensing 2023: InfoNCE + local-similarity constraint, MSTAR. — [DOAJ](https://doaj.org/article/f7639c6cdb1947a7bc89f2ed0e73bf88)
- Comer et al., SPIE 2020 "SAR ATR with less labels". — [SPIE](https://spiedigitallibrary.org/conference-proceedings-of-spie/11394/113940Q/SAR-automatic-target-recognition-with-less-labels/10.1117/12.2564875.short)
- DCPNet (arXiv 2311.15202, 2023): contrastive predictive pretraining on 2,296 OpenSARShip + 11,884 FUSAR-Ship chips; compares BYOL/SimSiam. — [arXiv 2311.15202](https://arxiv.org/pdf/2311.15202)

**Supervised-pretraining reference points (not SSL)**
- SARDet-100K / MSFA (Li et al., NeurIPS 2024; arXiv 2403.06534): 116,598 images, 245,653 instances, 6 classes, 10 source datasets, 512x512 slices; Faster R-CNN R-50 mAP ImageNet-only 49.0 to MSFA (ImageNet + DOTA detection + wavelet input) 51.1; ConvNeXt-B 55.1 to 56.4; Swin-B 53.8 to 55.7. Scope explicitly limited to supervised pretraining; SSL listed as future work. — [arXiv 2403.06534v3](https://arxiv.org/html/2403.06534v3); [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/e7eb8128eb26eafbe901348df1dbacdc-Abstract.html)
- Karimzadeh & de Lima (arXiv 2310.17126, 2023): ImageNet init vs random on sea-ice SAR segmentation; ImageNet mostly better, slightly worse on one winter scene (F1 0.97 vs 0.98). — [arXiv 2310.17126](https://arxiv.org/abs/2310.17126)

### Inferences
- The field has converged on chip-dominated pools + large ViT-family backbones + N-shot evaluation on MSTAR/FUSAR/SAR-ACD. The combination the user's study uses (compact MobileNetV2, 1/3/10 % label fractions, cross-dataset chips) does not appear in any paper found.
- The "ImageNet beats SAR SSL" signal does exist in the literature but only as isolated table cells (SAR-JEPA FUSAR linear probe; SARATR-X ViT 5-shot; AFRL-DINOv2 >=100-shot LoRA; DINOv3 below ImageNet-MAE in SARMAE) that the authors do not frame as negative transfer.

### Gaps
- IEEE TPAMI/ISPRS venue of SARATR-X v1 and SAR-JEPA not confirmed from the arXiv pages.
- SAR-HUB and FG-MAE exact downstream tables not fetched.
- SUMMIT's own paper was not fetched (only competitor-reported numbers).
- SARATR-X-v2 absolute SARDet-100K values for iTPN-B mAP50/75 are partly garbled in the extracted table.

## Key question 2: Has any paper pretrained on full SAR scenes and evaluated on object chips of a different dataset, and did any report scene-level SSL transferring worse than ImageNet weights?

### Takeaway
Only partial precedents exist: Muzeau et al. (SETHI airborne scene patches to MSTAR chips, no ImageNet baseline) and Inkawhich 2025 (DINOv3 satellite-scene weights worse than web weights on chip tasks; SAR-finetuned DINOv2 worse than web DINOv2 above 100 shots). No paper pretrains on SARDet-100K or FAIR-CSAR scenes and tests on ATRNet-STAR chips, and none states "scene-level SAR SSL < ImageNet init" as a finding.

### Cited Findings
- Muzeau et al. 2025 train a ViT-T/8 contrastive model on 199,040 patches from SETHI airborne stripmap scenes and evaluate by kNN on MSTAR spotlight chips without fine-tuning; 95.9 % at 10 labels/class vs ResNet-34 from scratch 52.2 %; there is no ImageNet-initialised comparison. — [arXiv 2502.01162](https://arxiv.org/pdf/2502.01162)
- Inkawhich 2025: DINOv3-ViT-L-Sat (Maxar satellite imagery) averages 25.6 vs 30.8 for DINOv3-ViT-L-Web on 10 SAR chip tasks; the author's explanation: satellite data "encourages the model to learn more 'scene' relevant features whereas the web data encourages the learning of 'object' relevant features". — [arXiv 2509.21722v1](https://arxiv.org/html/2509.21722v1)
- Inkawhich 2025: the SAR-finetuned AFRL-DINOv2-p1 trails the un-adapted web DINOv2 at 100/250/500 shots on ATRNet-SOC40 LoRA (60.5 vs 61.5; 79.6 vs 82.7; 90.1 vs 91.8). — [arXiv 2509.21722v1](https://arxiv.org/html/2509.21722v1)
- Inkawhich 2023 pretrained on ~100k chips cut from ship-detection frames (90 % ships) and transferred to MSTAR vehicles with gains over from-scratch; no ImageNet baseline. — [ar5iv 2303.10800](https://ar5iv.labs.arxiv.org/html/2303.10800)
- SARMAE ablation: DINOv3 initialisation (74.25 FUSAR) is below MAE-ImageNet (75.40); MAE on MillionAID optical RS scenes (80.16) is above ImageNet; the authors attribute differences to domain gap, not to scene vs chip. — [arXiv 2512.16635v1](https://arxiv.org/html/2512.16635v1)
- SAR-JEPA: ImageNet-supervised linear probe on FUSAR-Ship (74.6/79.6/83.6) exceeds MAE, FG-MAE, I-JEPA and LoMaR pretrained on the SAR chip pool. — [arXiv 2311.15153v6](https://arxiv.org/html/2311.15153v6)
- ATRNet-STAR benchmark: supervised SOC-40-pretrained weights transfer well to vehicle and aircraft datasets and give "little benefit" for ship (sea-scene) datasets vs ImageNet supervised (Fig. 14, numbers only in the figure). — [arXiv 2501.13354v4](https://arxiv.org/html/2501.13354v4)
- Pu & Xu pretrain MAE on SARDet-100K tiles but evaluate only detection on the same dataset. — [arXiv 2501.11249v1](https://arxiv.org/html/2501.11249v1)
- Deep-learning transfer survey defines negative transfer as source data increasing target error; depends on domain divergence and data size. — [arXiv 2205.09904](https://arxiv.org/pdf/2205.09904)

### Inferences
- The user's finding (a) is new in its explicit form: scene-archive SSL (SARDet-100K, FAIR-CSAR) measured worse than ImageNet init on another dataset's vehicle chips. The closest published hints (satellite-scene weights < web weights; SAR-adapted DINOv2 < web DINOv2 at >=100 shots) support the plausibility and give the "scene vs object features" explanation to cite.

### Gaps
- Queries tried with no direct hit: "negative transfer SAR pretraining worse than ImageNet initialization", "SAR SimCLR MoCo pretraining Sentinel-1 scenes transfer ship classification OpenSARShip FUSAR", "ATRNet-STAR few-shot self-supervised 2026 results SOC-40". No paper found that pretrains on SARDet-100K / HRSID / FAIR-CSAR full scenes and classifies chips of a different dataset.

## Key question 3: Has any SAR SSL paper used object proposals, saliency, CFAR-like bright-target detection, or region cropping to choose the SSL views?

### Takeaway
Three papers steer SSL by bright-scatterer statistics, but all steer the **mask or the reconstruction target** rather than the **crop/view location**: SAMBA (brightness-threshold scatterer map sets per-patch mask ratio), SARDet-MIM (SAR-Harris response in the target) and PCM-SAR (threshold heat-map boxes for local sampling, ~0.5 pt ablation gain). No paper crops SSL views around label-free bright regions of full scenes before DINO/JEPA-style training.

### Cited Findings
- SAMBA SG-MAE: scatterers found in the dB domain with speckle-suppression filtering, adaptive threshold T = mu + 1.2 sigma, local-contrast and neighbourhood screening; a 5x5 window counts scatterers per patch to set the per-patch mask ratio; block masks of 1/2/4 patches; no human labels described. Random-MAE vs SG-MAE (two-step, Mamba) at 5/10/20-shot: 71.5 vs 80.6, 78.5 vs 83.6, 84.0 vs 88.1. — [arXiv 2606.31668 html](https://arxiv.org/html/2606.31668)
- SARDet-MIM: reconstruction target = multi-scale gradient magnitude/orientation + multi-scale SAR-Harris response ("strong scattering centers"); masking is random; multi-scale strategy adds ~1.0 mAP on average. — [Remote Sensing 18:580](https://mdpi-res.com/d_attachment/remotesensing/remotesensing-18-00580/article_deploy/remotesensing-18-00580.pdf)
- PCM-SAR: heat map H(x) thresholded at T gives a binary mask and bounding boxes, Poisson-disk sampling places local crop centres; heat-map source unspecified; "Without SSG" ablation on ResNet-50 kNN 70.59 vs 71.10 with it. — [arXiv 2504.09502v1](https://arxiv.org/html/2504.09502v1)
- SAR-JEPA and SARATR-X use random local windows / random resized crops; SARATR-X lists "automatic target slicing" as future work. — [arXiv 2311.15153v6](https://arxiv.org/html/2311.15153v6); [arXiv 2405.09365v5](https://arxiv.org/html/2405.09365v5)
- XuanJian-DINOSAR uses 2 global + 8 local random crops with scale ranges; local consistency aligns same-location patch tokens, not target-selected crops. — [arXiv 2611.06105 PDF](https://arxiv.org/pdf/2611.06105)
- Optical precedent: SoCo (Wei et al., NeurIPS 2021) uses selective-search proposals as contrastive instances for detection pretraining; a follow-up derives proposals from the network itself. — [arXiv 2106.02637](https://arxiv.org/pdf/2106.02637); [arXiv 2504.13310](https://arxiv.org/html/2504.13310v1)
- SAR instance-level contrastive detection letter (IEEE GRSL 2023, InsRen RoI head) is the nearest SAR region-contrastive work, ship-only. — [GRSL 10.1109/LGRS.2023.3294483](https://unpaywall.org/10.1109%2FLGRS.2023.3294483)
- Physics-guided contrastive SAR work (Datcu/Huang 2022-2023) uses SAR-domain representations as guidance signals, not region selection. — [arXiv 2311.15153v3 related work](https://arxiv.org/html/2311.15153v3)

### Inferences
- "Cut views around label-free bright regions of scenes" (finding b) is not published for SAR SSL; the closest ideas (SAMBA, PCM-SAR) operate on chips or on masks and report smaller or unisolated gains.

### Gaps
- Queries with no CFAR hit: "target-aware contrastive learning SAR CFAR guided self-supervised", "scattering-guided masking SAR masked image modeling target region", "object-level region contrastive self-supervised pretraining SAR detection proposals". No "CFAR-guided" SSL paper found.

## Key question 4: Has any SAR SSL paper used a lightweight backbone (MobileNet / EfficientNet / ShuffleNet) with label-efficiency results?

### Takeaway
Only one SSL paper touches MobileNet/EfficientNet (Sensors 2025, SAMPLE-only, 4-layer CNN pretext encoder, no ImageNet baseline); the smallest backbones in the mainstream SSL line are ResNet-18 (SAR-SimCLR, BIDFC, PCM-SAR) and Mamba-27M (SAMBA). No MobileNetV2 SSL-for-SAR study with 1/3/10 % label fractions was found.

### Cited Findings
- Sensors 26(1):122 (Dec 2025): MobileNet v1 widths 1.0/0.75/0.5/0.25 reach 98.70/97.77/95.73/92.21 % on SAMPLE at full labels and 33.77/31.54/23.38 % at 5 % labels (43 chips) when fed SSL features; SimCLR ResNet-18 15.77 % at 5 %; no ImageNet-pretrained baseline in the portion read. — [PMC12787659](https://pmc.ncbi.nlm.nih.gov/articles/PMC12787659/)
- MobileNetV3 for SAR ATR under limited data (arXiv 2306.15287, 2023) is supervised only. — [arXiv 2306.15287](https://arxiv.org/pdf/2306.15287)
- SAR-SimCLR ResNet-18, 11M params (per AFRL 2025 table), few-shot MSTAR gains over scratch. — [arXiv 2509.21722v1](https://arxiv.org/html/2509.21722v1); [ar5iv 2303.10800](https://ar5iv.labs.arxiv.org/html/2303.10800)
- PCM-SAR ResNet-18: 58.91 % (10 % labels) / 71.80 % (30 %) on FUSAR-Ship semi-supervised. — [arXiv 2504.09502v1](https://arxiv.org/html/2504.09502v1)
- SAMBA Mamba encoder ~27M params, ~8 GFLOPs, "substantially fewer parameters than CNN and Transformer baselines". — [arXiv 2606.31668](https://arxiv.org/abs/2606.31668)
- SparK-SAR ConvNeXt-RS 27.8M params on Sentinel-1 land tasks. — [arXiv 2610.21993](https://arxiv.org/abs/2610.21993)
- SURGE (arXiv 2605.30380, 2026): contrastive relational distillation of a SAR ship detector into a student with >50 % fewer parameters. — [arXiv 2605.30380](https://arxiv.org/abs/2605.30380)
- Light-ResKAN: supervised lightweight model, 99.09/93.01/97.26 % on MSTAR/FUSAR-Ship/SAR-ACD, full labels. — [arXiv 2604.01903](https://arxiv.org/html/2604.01903)
- Inkawhich 2025: LoRA rank-3 adds ~1.35M trainable params vs 87M full finetune on ViT-B. — [arXiv 2509.21722v1](https://arxiv.org/html/2509.21722v1)

### Inferences
- Finding (e) (MobileNetV2 + 1/3/10 % labels, cross-dataset) has no direct comparator; the nearest label-fraction numbers are PCM-SAR's 10 %/30 % on FUSAR-Ship with ResNet-18.

### Gaps
- Queries: "lightweight MobileNet self-supervised pretraining SAR target recognition", "self-supervised SAR EfficientNet ShuffleNet label-efficient few-shot MSTAR pretraining lightweight". Nothing with ShuffleNet or EfficientNet SSL for SAR ATR.

## Key question 5: Which papers report results on ATRNet-STAR, on chips cut from SARDet-100K, or on FAIR-CSAR? Baselines and numbers

### Takeaway
ATRNet-STAR now has five SSL/foundation results (SARATR-X 85.2, SARMAE 81.4, AFRL-DINOv2 94.9, DINOSAR 96.7, SARATR-X-v2 97.8/98.4 on SOC-50; all with ViT-B-or-larger and with ATRNet-STAR train chips inside the SSL pool for v2 and DINOSAR). FAIR-CSAR appears only as a detection benchmark (AP50 38.0 to 42.6) and as an SSL pool component; no paper classifies chips cut from SARDet-100K.

### Cited Findings
- ATRNet-STAR (Liu et al., arXiv 2501.13354, 2025): 194,324 images, 40 vehicle types, 0.12-0.15 m, X/Ku, quad-pol, 128x128, five scenes (city 83,465; factory 63,597; sandstone 30,720; bare soil 8,448; woodland 8,094); SOC-40 split 68,091 / 29,284; SOC-50 18,071 / 17,603 (+ MSTAR 10 classes). — [arXiv 2501.13354v4](https://arxiv.org/html/2501.13354v4)
- ATRNet-STAR benchmark SOC-40 accuracy: VGG16 88.8; HDANet 89.1; ResNet18 90.6; ResNet34 91.7; ConvNeXt 96.0; ViT 76.4; HiViT 86.8; SARATR-X 96.4; MS-CVNet 80.9; LDSF 88.2. SOC-50: ConvNeXt 81.6, SARATR-X 85.2. EOC-Scene best 33.6 (HDANet). No MobileNet, no few-shot table (training-ratio curve only in Fig. 13). — [arXiv 2501.13354v4](https://arxiv.org/html/2501.13354v4)
- ATRNet-STAR detection mAP50 SOC-40: CenterNet 83.5; YOLOv8 86.1; YOLOv10 91.7; Faster R-CNN 88.3; DiffDet4SAR 90.9; SARATR-X 95.5. — [arXiv 2501.13354v4](https://arxiv.org/html/2501.13354v4)
- SOC-50 foundation-model table (from SARATR-X-v2 and DINOSAR): ViT-B 59.2; HDANet 63.7; ConvNeXt-B 81.6; SARMAE ViT-L 81.4; SARATR-X HiViT-B 85.2; AFRL-DINOv2 ViT-B 94.9; DINOSAR ViT-B/14 96.7; SARATR-X-v2 iTPN-B 97.8, iTPN-L 98.4. — [arXiv 2607.23238 PDF](https://arxiv.org/pdf/2607.23238); [arXiv 2611.06105 PDF](https://arxiv.org/pdf/2611.06105)
- AFRL-DINOv2 low-shot ATRNet-SOC40 (LoRA): 10-shot 22.1, 100-shot 60.5, 250-shot 79.6, 500-shot 90.1, all 98.6; hub DINOv2 14.8/61.5/82.7/91.8/98.8. — [arXiv 2509.21722v1](https://arxiv.org/html/2509.21722v1)
- SARATR-X-v2 and DINOSAR both include the ATRNet-STAR SOC-40 training split (68,091 chips) in their SSL pools. — [arXiv 2607.23238](https://arxiv.org/html/2607.23238); [arXiv 2611.06105 PDF](https://arxiv.org/pdf/2611.06105)
- FAIR-CSAR (Wu, Diao, Suo, Sun et al., IEEE TGRS vol. 63, 2025, DOI 10.1109/TGRS.2024.3519891): 175 Gaofen-3 SLC images, 32 cities + maritime regions, Spotlight (1 m nominal) and Fine Strip-Map modes, strong-scattering-point annotations. — [Journal of Radars instruction](https://radars.ac.cn/fileLDXB/cms/news/info/2025/03/e1e05c53d9de439fb8f963360e78c0d4/Instruction%20of%20FAIR-CSAR-V1.0%20Dataset.pdf); [PASTE arXiv 2603.14886](https://arxiv.org/pdf/2603.14886)
- FAIR-CSAR detection AP50 (OBB): RoI-Transformer ResNet-50 38.00; SSFG-Net 42.60; XuanJian-DINOSAR ViT-B/14 42.40. — [arXiv 2611.06105 PDF](https://arxiv.org/pdf/2611.06105)
- FAIR-CSAR contributes 23,825 tiles to the SARATR-X-v2 pool; it is not evaluated downstream there. — [arXiv 2607.23238](https://arxiv.org/html/2607.23238)
- SARDet-100K is detection-only; its only derived patches are 512x512 detection slices; no classification chips. — [arXiv 2403.06534v3](https://arxiv.org/html/2403.06534v3)

### Inferences
- There is no published chip-classification benchmark on FAIR-CSAR or on SARDet-100K chips, so finding (c) and any SARDet chip experiment have no external comparator.
- Published ATRNet-STAR SSL numbers are full-label or >=10-shot with ViT-B+; the 1/3/10 % MobileNetV2 regime is unoccupied.

### Gaps
- FAIR-CSAR TGRS paper itself (class count = 22, instance counts, its own baselines) not retrieved: aminer page returned only a header, IEEE page not fetched. The 22-class figure came from a search snippet only.
- ATRNet-STAR Fig. 14 transfer numbers are in a figure, not extractable.

## Key question 6: Has anyone used a large SAR foundation model as a frozen teacher or proposer for a small model, and compared against a cheap heuristic?

### Takeaway
No. Foundation models are used as frozen feature extractors (kNN/linear/LoRA), and the only teacher-student uses are optical DINOv3 as a semantic anchor inside SARMAE and generic detector distillation (SURGE, TerraMind). No paper uses SARMAE or any SAR FM to propose regions and benchmarks it against a brightness rule.

### Cited Findings
- SARMAE uses a frozen DINOv3 optical encoder as the "semantic anchor" teacher during pretraining; no distillation to a smaller model. — [arXiv 2512.16635v1](https://arxiv.org/html/2512.16635v1)
- Inkawhich 2025 evaluates SARATR-X and DINOv2/v3 as frozen extractors; MIM-based SARATR-X (44.6) trails DINO-based models because MIM objectives perform poorly as fixed features. — [arXiv 2509.21722v1](https://arxiv.org/html/2509.21722v1)
- SURGE: InfoNCE relational distillation from a teacher SAR ship detector to a compact student, >50 % smaller. — [arXiv 2605.30380](https://arxiv.org/abs/2605.30380)
- Generalised distillation of geospatial FMs (NeurIPS 2026): TerraMind-B (optical+radar) teacher to radar-only TerraMind-S student. — [NeurIPS 2026 listing](https://neurips.cc/virtual/2026/163753)
- Ice-FMBench proposes multi-teacher distillation into one efficient student for sea-ice segmentation. — [arXiv 2503.22516](https://arxiv.org/pdf/2503.22516.pdf)
- SAMBA's "proposer" for mask placement is a brightness threshold (mu + 1.2 sigma) with no learned model. — [arXiv 2606.31668 html](https://arxiv.org/html/2606.31668)

### Inferences
- Finding (d) (frozen SARMAE ViT-B as region proposer does not beat a pixel-brightness rule) has no published counterpart; the SAMBA brightness rule is the natural citation for "the cheap heuristic" and SARMAE/SARATR-X-as-frozen-extractor results (Inkawhich 2025) for "FM features are weak when frozen".

### Gaps
- Query "SAR foundation model knowledge distillation lightweight student" returned only the four items above; nothing with region proposals from an FM.
