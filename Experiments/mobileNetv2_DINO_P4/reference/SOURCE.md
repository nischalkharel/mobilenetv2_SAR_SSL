# Reference copies (read-only, never imported)

These files are untouched copies kept so every change can be checked. Nothing in this experiment imports them at run time; only `smoke_test.py` reads the original `DINOLoss`, `DINOHead` and `trunc_normal_` source from here to compare outputs, and `tools/make_change_record.py` diffs against them.

## dino-main_original/

Source: the local folder `dino-main` next to this repository (facebookresearch/dino, Apache License 2.0, Copyright (c) Facebook, Inc. and its affiliates). It is a downloaded ZIP: it has no `.git` folder, so **no commit hash is available** and none is claimed. The files are identified by SHA-256 instead (files dated 2026-10-13 15:30 on this PC, the extraction time).

| File | SHA-256 |
|---|---|
| `LICENSE` | `50e6751797c50dedd75ef1b8a0d9e42f5f8472e9fbce91f34718e9f97b0c780a` |
| `main_dino.py` | `74ef2afb9e16ab859370e1eb18455de0797d1202851c92d8191f06ccc05b28c7` |
| `README.md` | `9d78f02b2f2770f6f6169e33b2387c5997614a1b4f8009c6fa99cc3118e31442` |
| `utils.py` | `962a97e1acda1c986dfd921275325e4083f9016dddadcf018f1ccf03fa600eab` |
| `vision_transformer.py` | `b1f998d5f49ab43666b9fc6d007c5f6540c3ead2e8645e144f74045f63ff44d7` |

## baseline_original/

Copies of `Experiments/Baseline/` files as they were when this experiment was created (2026-10-13). The latest A0 run (`20261013-150506_A0_imagenet_mobilenetv2_seed0`) used this code (Git commit 96ef57f plus uncommitted changes, as recorded in its run_info.json).

| File | SHA-256 |
|---|---|
| `config.json` | `463f47321a4e1db1581f85bc0c102ad97846b17f1a635e384cf9a67e160ee7fa` |
| `data.py` | `f93852656e4da301b8021660c665456f8ea3cef39c0799b67a56a1798d89738d` |
| `engine.py` | `e0b767766de4fb149050c1251e4c194c819d6f4b26469fa685e5a729f56dd4ed` |
| `evaluate.py` | `c121f6be5160e510d965647db7fbbb6084160a860bfd7d12507632723c1c972d` |
| `model.py` | `e5b36ae6b2b41a7d26d02bcd5637d6289d0975e6864cf6154293e32bed95bb6c` |
| `plots.py` | `97ee3b867d95a946c26672864bfb0e0658ef768baf0b6b5f160029f2265b4ebf` |
| `smoke_test.py` | `d47fa2c590be4b6acc2a2f433ea931a45b06d7cb27fc64ee028341d86963363b` |
| `train.py` | `300ac2595c252fbe8a87ff87e7aff153720e71b8bf46ed41fe92457199f58844` |
| `utils.py` | `dbaaad5ad7acbec286d7528547b0268532405f84fcfbad4fa8a4d0d0678c23d6` |
