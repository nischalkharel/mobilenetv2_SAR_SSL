"""The experiment registry: one plain-English record per method row, so every page can say what a row is.

A row = one SSL method (or no SSL) trained on one unlabeled pool, then fine-tuned on one labeled set for one task.
Fields:
    id        short key used in the JSON files and the pages
    label     name shown in tables and charts
    family    colour family: base | dino | lejepa | spark | cond | region | scale (the ViewScale sweep)
    ssl       unlabeled pool: none | atr (ATRNet-STAR SOC-40 chips) | sardet (SARDet-100K scenes)
    control   True for a control row (random windows, scrambled conditioning ...), drawn lighter
    ref       id of the row a paired difference is taken against (None = no paired row)
    note      one sentence on what the row is for
The classification table (ATRNet-STAR test accuracy) is keyed by the method labels of SARDet_to_ATR/compare.py;
the detection table (SARDet-100K test mAP) by the `method` column of SARDet/compare/compare.py's runs.csv.
"""

# ---- classification on ATRNet-STAR (task: 40-class chip classification, 1 / 3 / 10 % labels) ----------------
CLASSIFICATION = {
    "A0 ImageNet (no SSL)": dict(id="a0", label="ImageNet start, no SSL", family="base", ssl="none", ref=None,
                                 note="The baseline every SSL row is measured against: ImageNet weights, then supervised fine-tuning only."),
    "A1 DINO on ATRNet": dict(id="a1", label="DINO", family="dino", ssl="atr", ref="a0",
                              note="Plain DINO on the ATRNet-STAR unlabeled chips (in-domain)."),
    "A2 LeJEPA on ATRNet": dict(id="a2", label="LeJEPA", family="lejepa", ssl="atr", ref="a0",
                                note="LeJEPA on the ATRNet-STAR chips."),
    "A3 SparK on ATRNet": dict(id="a3", label="SparK", family="spark", ssl="atr", ref="a0",
                               note="SparK (masked image modelling) on the ATRNet-STAR chips."),
    "P4 CondDINO on ATRNet": dict(id="p4", label="CondDINO", family="cond", ssl="atr", ref="a1",
                                  note="DINO with per-cell centering keyed by acquisition metadata. Uses metadata fields, not class labels; counted separately for that reason."),
    "P4 scrambled (control)": dict(id="p4perm", label="CondDINO, scrambled", family="cond", ssl="atr", ref="a1", control=True,
                                   note="CondDINO with the conditioning field scrambled (split 1 only)."),
    "P4 3-field (control)": dict(id="p4d", label="CondDINO, 3 fields", family="cond", ssl="atr", ref="a1", control=True,
                                 note="CondDINO conditioned on three metadata fields (split 1 only)."),
    "XA DINO on SARDet": dict(id="xa_dino", label="DINO", family="dino", ssl="sardet", ref="a0",
                              note="Plain DINO on 94,034 SARDet-100K scenes, then fine-tuned on ATRNet chips (cross-dataset)."),
    "XA LeJEPA on SARDet": dict(id="xa_lejepa", label="LeJEPA", family="lejepa", ssl="sardet", ref="a0",
                                note="LeJEPA on SARDet scenes, tested on ATRNet chips."),
    "XA SparK on SARDet": dict(id="xa_spark", label="SparK", family="spark", ssl="sardet", ref="a0",
                               note="SparK on SARDet scenes, tested on ATRNet chips."),
    "XA RegionDINO scene+object on SARDet": dict(id="xa_rso", label="RegionDINO scene+object", family="region", ssl="sardet", ref="xa_dino",
                                                 note="DINO on the whole scene plus DINO on label-free object regions cut from it."),
    "XA RegionDINO scene+object, random windows (control)": dict(id="xa_rsor", label="scene+object, random windows", family="region", ssl="sardet", ref="xa_dino", control=True,
                                                                 note="Same crops at random places: tests whether the regions or just the small crops help."),
    "XA RegionDINO object-only on SARDet": dict(id="xa_ro", label="RegionDINO object-only", family="region", ssl="sardet", ref="xa_dino",
                                                note="DINO on the object regions only; the scene views are dropped."),
    "XA RegionDINO object-only, random windows (control)": dict(id="xa_ror", label="object-only, random windows", family="region", ssl="sardet", ref="xa_dino", control=True,
                                                                note="Object-only with the windows placed at random."),
    "XA ViewScale e32 (fixed 32-px windows)": dict(id="xa_vs32", label="ViewScale e32 (32-px windows)", family="scale", ssl="sardet", ref="xa_dino",
                                                 note="Object-only recipe with random square windows of a fixed 32-px extent (512-px frame): the window-size sweep, one point."),
    "XA ViewScale e64 (fixed 64-px windows)": dict(id="xa_vs64", label="ViewScale e64 (64-px windows)", family="scale", ssl="sardet", ref="xa_dino",
                                                 note="Object-only recipe with random square windows of a fixed 64-px extent (512-px frame): the window-size sweep, one point."),
    "XA ViewScale e128 (fixed 128-px windows)": dict(id="xa_vs128", label="ViewScale e128 (128-px windows)", family="scale", ssl="sardet", ref="xa_dino",
                                                 note="Object-only recipe with random square windows of a fixed 128-px extent (512-px frame): the window-size sweep, one point."),
    "XA ViewScale e256 (fixed 256-px windows)": dict(id="xa_vs256", label="ViewScale e256 (256-px windows)", family="scale", ssl="sardet", ref="xa_dino",
                                                 note="Object-only recipe with random square windows of a fixed 256-px extent (512-px frame): the window-size sweep, one point."),
    "XA ViewScale e512 (fixed 512-px windows)": dict(id="xa_vs512", label="ViewScale e512 (512-px windows)", family="scale", ssl="sardet", ref="xa_dino",
                                                 note="Object-only recipe with random square windows of a fixed 512-px extent (512-px frame): the window-size sweep, one point."),
    "XA RegionDINO object-only, teacher regions": dict(id="xa_rot", label="object-only, teacher regions", family="region", ssl="sardet", ref="xa_ro",
                                                        note="Object-only recipe with the windows placed on regions proposed by the frozen SARMAE teacher's attention instead of the brightness rule. Paired against the brightness-rule row."),
    "XA RegionDINO object-only, teacher-size random windows (control)": dict(id="xa_rotr", label="teacher-size random windows", family="region", ssl="sardet", ref="xa_rot", control=True,
                                                                             note="Random windows with the teacher regions' sizes (2 to 3 x larger than the brightness-rule regions): separates placement from size."),
    "XI DINO (suite recipe) on ATRNet": dict(id="xi_dino", label="DINO, RegionSSL engine", family="dino", ssl="atr", ref="a1",
                                             note="Plain DINO run by the RegionDINO engine on the ATRNet chips: the paired reference for the XI RegionDINO rows, and a check against A1."),
    "XI RegionDINO scene+object on ATRNet": dict(id="xi_rso", label="RegionDINO scene+object", family="region", ssl="atr", ref="xi_dino",
                                                 note="Scene+object views with SSL on the ATRNet chips themselves (XI rows, also called arm I)."),
    "XI RegionDINO scene+object on ATRNet, random windows (control)": dict(id="xi_rsor", label="scene+object, random windows", family="region", ssl="atr", ref="xi_dino", control=True,
                                                                           note="Its random-window control."),
    "XI RegionDINO object-only on ATRNet": dict(id="xi_ro", label="RegionDINO object-only", family="region", ssl="atr", ref="xi_dino",
                                                note="Object-only views with SSL on the ATRNet chips themselves (XI rows, also called arm I)."),
    "XI RegionDINO object-only on ATRNet, random windows (control)": dict(id="xi_ror", label="object-only, random windows", family="region", ssl="atr", ref="xi_dino", control=True,
                                                                          note="Its random-window control."),
}

# ---- detection on SARDet-100K (task: 6-class box detection, 1 / 3 / 10 % labels; SSL always on SARDet scenes) ----
DETECTION = {
    "baseline_SARDet": dict(id="d_base", label="ImageNet start, no SSL", family="base", ssl="none", ref=None,
                            note="Faster R-CNN head on ImageNet MobileNetV2, supervised only."),
    "DINO_SARDet": dict(id="d_dino", label="DINO", family="dino", ssl="sardet", ref="d_base", note="Plain DINO on SARDet scenes."),
    "LeJEPA_SARDet": dict(id="d_lejepa", label="LeJEPA", family="lejepa", ssl="sardet", ref="d_base", note="LeJEPA on SARDet scenes."),
    "SparK_SARDet": dict(id="d_spark", label="SparK", family="spark", ssl="sardet", ref="d_base", note="SparK on SARDet scenes."),
    "CondDINO_SARDet": dict(id="d_cond", label="CondDINO (source + polarization)", family="cond", ssl="sardet", ref="d_dino",
                            note="DINO with centering per source dataset and polarization."),
    "CondDINO_SARDet (permuted)": dict(id="d_condp", label="CondDINO, permuted", family="cond", ssl="sardet", ref="d_dino", control=True,
                                       note="CondDINO with the conditioning permuted."),
    "RegionDINO_object": dict(id="d_ro", label="RegionDINO object-only", family="region", ssl="sardet", ref="d_dino", note="Object-only views, detection on SARDet itself."),
    "RegionDINO_object (random)": dict(id="d_ror", label="object-only, random windows", family="region", ssl="sardet", ref="d_dino", control=True, note="Its random-window control."),
    "ViewScale e32": dict(id="d_vs32", label="ViewScale e32 (32-px windows)", family="scale", ssl="sardet", ref="d_base", note="Fixed 32-px random windows, object-only recipe (split 1 only at first)."),
    "ViewScale e64": dict(id="d_vs64", label="ViewScale e64 (64-px windows)", family="scale", ssl="sardet", ref="d_base", note="Fixed 64-px random windows, object-only recipe (split 1 only at first)."),
    "ViewScale e128": dict(id="d_vs128", label="ViewScale e128 (128-px windows)", family="scale", ssl="sardet", ref="d_base", note="Fixed 128-px random windows, object-only recipe (split 1 only at first)."),
    "ViewScale e256": dict(id="d_vs256", label="ViewScale e256 (256-px windows)", family="scale", ssl="sardet", ref="d_base", note="Fixed 256-px random windows, object-only recipe (split 1 only at first)."),
    "ViewScale e512": dict(id="d_vs512", label="ViewScale e512 (512-px windows)", family="scale", ssl="sardet", ref="d_base", note="Fixed 512-px random windows, object-only recipe (split 1 only at first)."),
    "TeacherRegions": dict(id="d_rot", label="object-only, teacher regions", family="region", ssl="sardet", ref="d_ro", note="Windows on SARMAE-proposed regions; paired against the brightness-rule row."),
    "TeacherRegions (random)": dict(id="d_rotr", label="teacher-size random windows", family="region", ssl="sardet", ref="d_rot", control=True, note="Random windows with the teacher regions' sizes."),
    "RegionDINO_scene_object": dict(id="d_rso", label="RegionDINO scene+object", family="region", ssl="sardet", ref="d_dino", note="Scene+object views, detection on SARDet itself."),
    "RegionDINO_scene_object (random)": dict(id="d_rsor", label="scene+object, random windows", family="region", ssl="sardet", ref="d_dino", control=True, note="Its random-window control."),
}

# ---- chip classification on SARDet-100K (task: 6-class chips cut from the detection boxes; SSL always on SARDet scenes) ----
CHIPS = {
    "imagenet": dict(id="c_base", label="ImageNet start, no SSL", family="base", ssl="none", ref=None,
                     note="ImageNet weights fine-tuned directly on the labeled chips."),
    "dino": dict(id="c_dino", label="DINO", family="dino", ssl="sardet", ref="c_base", note="Plain DINO on the SARDet scenes, then chips."),
    "lejepa": dict(id="c_lejepa", label="LeJEPA", family="lejepa", ssl="sardet", ref="c_base", note="LeJEPA on the scenes, then chips."),
    "spark": dict(id="c_spark", label="SparK", family="spark", ssl="sardet", ref="c_base", note="SparK on the scenes, then chips."),
    "regiondino_so": dict(id="c_rso", label="RegionDINO scene+object", family="region", ssl="sardet", ref="c_dino", note="Scene+object views on the scenes, then chips."),
    "regiondino_so_random": dict(id="c_rsor", label="scene+object, random windows", family="region", ssl="sardet", ref="c_dino", control=True, note="Its random-window control."),
    "regiondino_o": dict(id="c_ro", label="RegionDINO object-only", family="region", ssl="sardet", ref="c_dino", note="Object-only views on the scenes, then chips."),
    "regiondino_o_random": dict(id="c_ror", label="object-only, random windows", family="region", ssl="sardet", ref="c_dino", control=True, note="Its random-window control."),
    "regiondino_o_teacher": dict(id="c_rot", label="object-only, teacher regions", family="region", ssl="sardet", ref="c_ro", note="Windows on SARMAE-proposed regions (TeacherRegions backbone), then chips; paired against the brightness-rule row."),
    "regiondino_o_teacher_random": dict(id="c_rotr", label="teacher-size random windows", family="region", ssl="sardet", ref="c_rot", control=True, note="Random windows with the teacher regions' sizes, then chips."),
}

# ---- FAIR-CSAR chip classification (FAIR_chips study: 22 classes; SSL on FAIR-CSAR itself or on SARDet-100K) ----
FAIR_CLASSES = ["Airbus_A220", "Airbus_A320", "Airbus_A330", "Airfreighter", "Boeing737", "Boeing747", "Boeing767", "Boeing777", "Fokker-50",
                "Gulfstream", "Helicopter", "Other_Aircraft", "Bulk_Carrier", "Container_Ship", "General_Cargo_Ship", "Motion_Defocusing_Ship",
                "Oil_Tanker", "Other_Ship", "Warship", "Bridge", "Tank", "Tower_Crane"]
FAIR_CHIPS = {
    "imagenet": dict(id="f_base", label="ImageNet start, no SSL", family="base", ssl="none", ref=None,
                     note="ImageNet weights fine-tuned directly on the labeled FAIR-CSAR chips."),
    "fair_dino_tiles": dict(id="f_dino_t", label="DINO", family="dino", ssl="fair_tiles", ref="f_base", note="Plain DINO on the FAIR-CSAR native 1 m tiles, then the FAIR chips."),
    "fair_ro_tiles": dict(id="f_ro_t", label="RegionDINO object-only", family="region", ssl="fair_tiles", ref="f_dino_t", note="Object-only views on brightness-rule regions of the FAIR-CSAR tiles."),
    "fair_ror_tiles": dict(id="f_ror_t", label="object-only, random windows", family="region", ssl="fair_tiles", ref="f_dino_t", control=True, note="Its random-window control (same window sizes, random places)."),
    "fair_dino_tiles500": dict(id="f_dino_t5", label="DINO, 500 epochs", family="dino", ssl="fair_tiles", ref="f_base", note="Plain DINO on the FAIR-CSAR native 1 m tiles for 500 epochs (148,500 steps, the SARDet-100K runs' count; the 100-epoch runs stopped at 29,700)."),
    "fair_ro_tiles500": dict(id="f_ro_t5", label="RegionDINO object-only, 500 epochs", family="region", ssl="fair_tiles", ref="f_dino_t5", note="Object-only views on brightness-rule regions of the FAIR-CSAR tiles, 500 epochs."),
    "fair_ror_tiles500": dict(id="f_ror_t5", label="object-only, random windows, 500 epochs", family="region", ssl="fair_tiles", ref="f_dino_t5", control=True, note="Its random-window control, 500 epochs."),
    "fair_rot_tiles500": dict(id="f_rot_t5", label="object-only, teacher regions, 500 epochs", family="region", ssl="fair_tiles", ref="f_ro_t5", note="Object-only views on regions proposed by frozen SARMAE's attention (TeacherRegions) instead of the brightness rule, 500 epochs."),
    "fair_rotr_tiles500": dict(id="f_rotr_t5", label="teacher-size random windows, 500 epochs", family="region", ssl="fair_tiles", ref="f_rot_t5", control=True, note="Random windows with the teacher regions' sizes, 500 epochs (the teacher row's control)."),
    "fair_dino_down": dict(id="f_dino_d", label="DINO", family="dino", ssl="fair_down", ref="f_base", note="Plain DINO on the FAIR-CSAR slices downsampled to 512 px (2 m per pixel)."),
    "fair_ro_down": dict(id="f_ro_d", label="RegionDINO object-only", family="region", ssl="fair_down", ref="f_dino_d", note="Object-only views on the downsampled slices."),
    "fair_ror_down": dict(id="f_ror_d", label="object-only, random windows", family="region", ssl="fair_down", ref="f_dino_d", control=True, note="Its random-window control."),
    "sardet_dino": dict(id="f_sd_dino", label="DINO", family="dino", ssl="sardet", ref="f_base", note="The SARDet-100K DINO backbone (no new SSL), fine-tuned on FAIR-CSAR chips: cross-dataset transfer."),
    "sardet_regiondino_o": dict(id="f_sd_ro", label="RegionDINO object-only", family="region", ssl="sardet", ref="f_sd_dino", note="The SARDet-100K object-only backbone, fine-tuned on FAIR-CSAR chips."),
    "sardet_regiondino_o_random": dict(id="f_sd_ror", label="object-only, random windows", family="region", ssl="sardet", ref="f_sd_dino", control=True, note="The SARDet-100K random-window control, fine-tuned on FAIR-CSAR chips."),
}

POOLS = {
    "none": "no SSL",
    "atr": "ATRNet-STAR chips (128 px, target-centred, SOC-40 unlabeled pool)",
    "sardet": "SARDet-100K scenes (94,034 full scenes, shorter side 512)",
    "fair_tiles": "FAIR-CSAR SL trainval slices as 19,036 unique native 1 m tiles of 512 px (selection and test slices excluded)",
    "fair_down": "FAIR-CSAR SL trainval slices downsampled 1024 -> 512 px (10,769 images)",
}

# ---- the experiment map on the overview page: one card per (SSL pool -> labeled data / task) --------------------
SETTINGS = [
    dict(id="atr_cls", title="SSL on ATRNet-STAR chips, classify ATRNet-STAR chips", short="in-domain classification",
         table="classification", ssl="atr", task="40-class accuracy on the official ATRNet-STAR test set"),
    dict(id="sardet_cls", title="SSL on SARDet-100K scenes, classify ATRNet-STAR chips", short="cross-dataset classification",
         table="classification", ssl="sardet", task="40-class accuracy on the official ATRNet-STAR test set"),
    dict(id="sardet_det", title="SSL on SARDet-100K scenes, detect on SARDet-100K", short="in-domain detection",
         table="detection", ssl="sardet", task="mAP@[.5:.95] on the official SARDet-100K test set"),
    dict(id="sardet_chips", title="SSL on SARDet-100K scenes, classify SARDet-100K chips", short="in-domain chip classification",
         table="chips", ssl="sardet", task="mean per-class accuracy on chips cut from the official test boxes (clean set, 6 classes)"),
    dict(id="fair_chips", title="SSL on FAIR-CSAR (or on SARDet-100K), classify FAIR-CSAR chips", short="second dataset, 22 classes",
         table="fair", ssl="fair", task="mean per-class accuracy on FAIR-CSAR test chips (clean set, 22 classes)"),
]

# families in display order, with the plain name of each
FAMILIES = [("base", "no SSL"), ("dino", "DINO"), ("lejepa", "LeJEPA"), ("spark", "SparK"), ("cond", "CondDINO"), ("region", "RegionDINO")]
