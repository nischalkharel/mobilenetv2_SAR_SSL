# CondDINO_SARDet: DINO with the teacher center kept per acquisition-condition cell (the ATRNet-STAR "P4" method on SARDet-100K)

Same route as `DINO_SARDet` with ONE change inside the DINO loss: the teacher's anti-collapse center is kept
PER CELL and every image has its own cell's center subtracted before the sharpened softmax.

    DINO:      target = softmax( (teacher(x) - center            ) / temperature )
    CondDINO:  target = softmax( (teacher(x) - center[cell(x)]   ) / temperature )

Each cell's center is an EMA (momentum 0.9) of the teacher outputs of that cell's images in the batch; only the
cells present in a batch move; the global center is still tracked for the record but never subtracted. Views,
head, temperatures, EMA teacher, optimizer, schedules, checkpointing, export and fine-tuning are DINO's (the code
is imported from `DINO_SARDet/`, not copied: `dino_base.py`). Cells are used only inside the loss during SSL; the
detector never sees them. The ATRNet-STAR origin is `mobilenetv2_SAR_SSL/Experiments/mobileNetv2_DINO_P4/`
(`cond_center.py` there is ported here with the same arithmetic).

## The honest statement (DESIGN.md section 5)

ATRNet-STAR gave scene, band, depression and azimuth per chip: fields that cut across classes. SARDet-100K gives,
per image, only what the source name carries, and the source IS the class for most images (OGSOD = bridge /
harbor / tank, SADD and SAR-AIRcraft = aircraft, SIVED = car, HRSID / SSDD / AIR / Shipdataset = ship, MSAR
mixed). **No per-image acquisition field that is independent of the class exists in SARDet-100K.** This folder
therefore runs the mechanism with the fields that exist and MEASURES the class leak instead of hiding it
(`cell_report.py`). A gain of `source_pol` over DINO is not, by itself, evidence for "condition-aware centering";
the four variants below are what separates condition from class information.

## The four variants (`config_ssl*.json`, section `cond_center.cells`)

| variant | cells | labels in SSL | deployable | what it can show | what it cannot show |
|---|---|---|---|---|---|
| `source_pol` (default, `config_ssl.json`, run `condcenter_source_pol`) | source dataset, refined by satellite x polarisation for Shipdataset and sub-source for SIVED (manifest columns only): 19 cells of 80 to 27,159 images on the split-1 pool | none | no (the source is a curation artefact) | whether per-cell centering helps with the best "sensor bundle" cells the data has | whether the help is condition rather than class information (the cells leak class; see the report) |
| `perm` (`config_ssl_perm.json`, `condcenter_perm`) | the same cell sizes with the image -> cell assignment scrambled (`numpy.default_rng(permute_seed)`) | none | - | the size-matched control: must NOT gain over DINO. If it does, the gain is not the grouping | - |
| `oracle_class` (`config_ssl_oracle_class.json`, `condcenter_oracle_class`) | the set of classes present in the image, from the official `train.json` | **YES: labels used** | no | the LABEL-LEAK CEILING: how much per-cell centering gains when the cell is literally the label. If `source_pol` ~ `oracle_class` >> `perm`, the `source_pol` gain is class information, not condition | nothing about a method: it is never one. Marked "oracle_class control, labels used" in its config, `cells.csv`, the backbone file and every record |
| `radiometric` (`config_ssl_radiometric.json`, `condcenter_radiometric`) | k-means (k = 16, own numpy implementation, seed 0) on a 32-bin intensity histogram of the 512-resized grey image + log(native width x height), fitted on the SSL pool | none (pixels and image size only) | **yes** | the closest honest substitute for a per-image acquisition condition, available at deployment; whether label-free cells help | that its cells are class-free: its leak is measured too (histograms differ by source) |

`cond_center.enabled: false` runs one cell for every image, which equals DINO to 1e-6 (checked). `label_use` in
each config must state the truth for its variant word for word, or `train_ssl.py` refuses to start.

## Files

| file | what |
|---|---|
| `dino_base.py` | loads `../DINO_SARDet/dino.py` and `train_ssl.py` with importlib (nothing copied); exposes `dino`, `DINOMethod` |
| `cond_center.py` | `CondCenterDINOLoss(DINOLoss)` (per-cell buffers, view-major gather, per-cell EMA of present cells, global center tracked only), `assign_cells`, `permute_ids`, `cell_table` |
| `cells.py` | the four cell definitions -> one integer id per pool row + names; the radiometric feature pass (DataLoader workers) and k-means with a cache under `outputs/cache/radiometric_<pool sha256[:16]>.npz` |
| `cell_report.py` | class-vs-cell contingency, purity, majority vote vs scrambled vs chance, I(cell; class set) and I(cell; dominant class) vs scrambled, cell sizes; `outputs/cell_report/<variant>_split<s>.{md,csv,json}`; prints a WARNING when any cell is > 90 % one class (no pass/fail: the leak is expected and measured) |
| `train_ssl.py` | `CondDINOMethod(DINOMethod)`: `build` (loss swap + cell assignment + cell files), `train_epoch` (gathers the batch's cell ids from the row indices), checkpoint with the cell centers and the assignment hash, teacher export with the cell record |
| `run_finetune.py`, `config_finetune.json` | stage 2, identical protocol to the baseline (`shared/finetune.py`); run name `condcenter_mobilenetv2` (pass `--run-name <variant>_mobilenetv2` for the other variants) |
| `smoke_test.py` | DINO's checks + (a) one cell == DINO, (b) view-major gather / EMA of present cells, (c) perm keeps the histogram, (d) cell report on the smoke subset, (e) smoke SSL run + stop/resume + fine-tune + probes, (f) radiometric smoke run |
| `NOTES_FOR_MAINTAINER.md` | proposed (optional) changes to `shared/` |

## Outputs

- `outputs/ssl/<stamp>_condcenter_<variant>_split<s>_seed<k>/`: everything a DINO run writes (config, manifests,
  run_info, history.csv with the extra `cells_seen` column, charts, checkpoint.pth incl. the cell centers,
  `backbone_epoch100.pth` whose `exported` metadata records the variant, n_cells, assignment sha256, label use
  and the paths below).
- in the run folder: `cells.csv` (cell id, name, images, mapping, label_use), `cells.json` (the cell record incl.
  the assignment sha256), `cell_center_stats.csv` (visits and center norm per cell, rewritten every epoch) and
  `cells_radiometric.npz` for that variant. The engine sets `method.run_dir` before `build()`; with an engine that
  does not, the same files go to `outputs/cells/<variant>_split<s>_seed<k>[_SMOKE]_<assignment sha256[:16]>*`.
  Either way the paths are recorded in the backbone file's `exported` metadata.
- `outputs/cache/radiometric_<pool sha>.npz`: features + assignment + centroids, reused by every radiometric run
  and report on the same pool (the 94k-image histogram pass takes minutes with workers).
- `outputs/cell_report/<variant>_split<s>.{md,csv,json}`.

## Commands (from the suite root, with the suite's interpreter)

```powershell
cd C:\Users\nisch\Documents\SHREC\2026_2027\SARDet-work\SARDet
.\.venv\Scripts\python.exe CondDINO_SARDet\smoke_test.py                               # minutes; never a result
.\.venv\Scripts\python.exe CondDINO_SARDet\cell_report.py --cells source_pol           # seconds (labels read for the report only)
.\.venv\Scripts\python.exe CondDINO_SARDet\cell_report.py --cells oracle_class
.\.venv\Scripts\python.exe CondDINO_SARDet\cell_report.py --cells radiometric --workers 8   # builds the k-means cache
.\.venv\Scripts\python.exe CondDINO_SARDet\train_ssl.py                                # source_pol, 100 epochs (CRC)
.\.venv\Scripts\python.exe CondDINO_SARDet\train_ssl.py --config CondDINO_SARDet\config_ssl_perm.json
.\.venv\Scripts\python.exe CondDINO_SARDet\train_ssl.py --config CondDINO_SARDet\config_ssl_oracle_class.json   # ceiling
.\.venv\Scripts\python.exe CondDINO_SARDet\train_ssl.py --config CondDINO_SARDet\config_ssl_radiometric.json
.\.venv\Scripts\python.exe CondDINO_SARDet\train_ssl.py --resume <run folder>          # after --stop-after-epoch / a time limit
.\.venv\Scripts\python.exe CondDINO_SARDet\run_finetune.py --backbone <run>\backbone_epoch100.pth --budget 1pct
.\.venv\Scripts\python.exe CondDINO_SARDet\run_finetune.py --run-name condcenter_perm_mobilenetv2 --backbone <perm run>\backbone_epoch100.pth --budget 1pct
```

`SAR_SPLIT=2|3` selects the split (seed = split - 1), as everywhere in the suite. Equal epochs across methods is
the control (DESIGN.md section 4); per-cell centering adds one gather and one scatter-add per step, so the cost
equals DINO's within noise. Measured on the PC (RTX 4070 SUPER, batch 64, 8 workers, fp16, 2026-11-11): 0.22 s/step
= 5.4 min per epoch, about 9 h for 100 epochs (`outputs/timing_probe.log`; the probe run was deleted).

## What to compare (per split, paired, DESIGN.md section 6)

`condcenter_source_pol`, `condcenter_perm`, `condcenter_oracle_class`, `condcenter_radiometric` against
`dino_ssl_mobilenetv2` of the same split, same labeled sets, same fine-tune seed. Read the gaps together with
`outputs/cell_report/`: the cell report's purity and I(cell; class) numbers say how much of a `source_pol` gain
could be class information, `perm` says whether the grouping matters at all, `oracle_class` is the ceiling, and
`radiometric` is the only variant that could be used on new, unlabeled, source-less data.

## Known limitations

- `source_pol` and `oracle_class` cells are not deployable; only `radiometric` is.
- The radiometric k-means is fitted on the SSL pool once (seed 0); a new pool gives new cells. The histogram
  features are of the 512-resized grey image, so the descriptor matches what the detector sees; the native
  size enters through log(width x height) from the manifest.
- The cell files of a run are not in its run folder (see above); they are linked from the backbone file.
- The smoke test runs 48 images, so its cells are tiny (16 k-means cells on 48 images); it checks the mechanics,
  not the effect.
