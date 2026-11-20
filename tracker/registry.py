"""The experiment registry: one plain-English record per method row, so every page can say what a row is.

A row = one SSL method (or no SSL) trained on one unlabeled pool, then fine-tuned on one labeled set for one task.
Fields:
    id        short key used in the JSON files and the pages
    label     name shown in tables and charts
    family    colour family: base | dino | lejepa | spark | cond | region
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
}

POOLS = {
    "none": "no SSL",
    "atr": "ATRNet-STAR chips (128 px, target-centred, SOC-40 unlabeled pool)",
    "sardet": "SARDet-100K scenes (94,034 full scenes, shorter side 512)",
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
]

# families in display order, with the plain name of each
FAMILIES = [("base", "no SSL"), ("dino", "DINO"), ("lejepa", "LeJEPA"), ("spark", "SparK"), ("cond", "CondDINO"), ("region", "RegionDINO")]
