# CHANGES: this folder versus Experiments/mobileNetv2_DINO (A1)

Copied on 2026-11-09 from `Experiments/mobileNetv2_DINO/` (A1), which was not modified. `make_changes_record.py`
regenerates `diffs/<file>.diff` and the hash table `diffs/SUMMARY.md`; this file explains every change and states
whether it can change what DINO computes. The pilot copy in `Independent_Study_For_agents/pilot/dino_pilot/`
(arms P0-P4) was the reference for the mechanism; this folder carries only the P4 part, written against the A1 code.

## 1. Files

| File | Status | Summary |
|---|---|---|
| `dino.py` | identical | DINO components (head, loss, multi-crop wrapper, augmentation, diagnostics): untouched |
| `data.py`, `model.py`, `engine.py`, `utils.py` | identical | manifests and images, MobileNetV2 construction, epoch loop, helpers |
| `reference/` | identical | untouched originals (DINO repository extracts, baseline) used by `smoke_test.py` section 4 |
| `plots.py`, `preview_views.py`, `evaluate.py` | command paths only | `Experiments/mobileNetv2_DINO/` -> `Experiments/mobileNetv2_DINO_P4/` in docstrings and printed hints; no code changed |
| `train_supervised.py` | modified | run folder created by `make_run_dir()` (2.6); command path in the docstring |
| `paths.py` | docstring only | same settings; `EXPERIMENT_DIR` is derived from the file location, so outputs land in this folder |
| `ssl_data.py` | modified | `flat_args()` carries the `cond_center` section; per-process temp name for the pixel cache; `make_run_dir()` (section 2.1) |
| `export_backbone.py` | modified | `ssl_method` says "per-condition-cell teacher centering" when the run used it; command path (2.2) |
| `train_ssl.py` | modified | section 2.3 (changes listed by topic; the unified diff is `diffs/train_ssl.py.diff`) |
| `smoke_test.py` | modified | A1 checks kept; P4 checks added (2.4) |
| `cond_center.py` | new | cells, `CellSSLDataset`, `CondCenterDINOLoss` (2.5) |
| `cell_report.py`, `make_changes_record.py` | new | analysis and record tools; not used by training |
| `config_ssl.json` | A1 values + `cond_center` | section 3 |
| `config_ssl_perm.json`, `config_ssl_3field.json`, `config_finetune*.json` | new | section 3 |
| not copied | | `DINO_CHANGES.md`, `README.md`, `tools/`, `diffs/` (A1's own records of its DINO extracts), `outputs/` |

## 2. Changes, one by one

### 2.1 `ssl_data.py` (cannot change results)
`flat_args()` adds `args.cond_center` = the config's `cond_center` section with the `_note` key dropped, defaulting to
`{"enabled": false, "cell": [], "permute_cells": false, "permute_seed": 0}` when the section is absent. Kept nested so
its keys cannot clash with the DINO argument names. Because `vars(args)` is what `checkpoint.pth` stores and what the
resume check compares, a changed cell definition is rejected on resume (checked by the smoke test).
`build_pixel_cache()` writes its temporary file as `pixels_<hash>_<N>.<pid>.tmp.npy` instead of a fixed name: two
jobs that start on the same split at the same time (P4 and its scrambled control) build the same cache from the same
image list and must not truncate each other's half-written file; the final rename is atomic (same bytes either way).
`make_run_dir(root, name)` (added 2026-11-10 after CRC jobs 4718972 and 4718977 failed) creates `root/<stamp>_<name>`
with the current second as the stamp and returns it; when that folder already exists, because another job started
the same script in the same second, it waits one second and takes the next stamp (up to 10 tries) instead of
stopping. The five P4 jobs on CRC started within one minute and ran the identical smoke test in step; three reached
its save-and-resume section in the same second and two stopped on the folder the third had just created.
`train_ssl.py`, `train_supervised.py` and the smoke test use it; the names are unchanged and a folder is still never
overwritten.

### 2.2 `export_backbone.py` (cannot change results)
The exported backbone file's `ssl_method` string says "DINO with per-condition-cell teacher centering (experiment P4)"
when `ckpt["args"]["cond_center"]["enabled"]`, else the A1 string; `ssl_config` (already exported) carries the full
`cond_center` setting. Docstring command path.

### 2.3 `train_ssl.py`
Can any hunk change the DINO computation? Only hunks 4-7, and only when `cond_center.enabled` is true: the loss class
becomes `CondCenterDINOLoss` (each teacher row centered with its own cell's center) and the dataset supplies the cell
ids. With `enabled: false` the run is A1 (`DINOLoss`, cell ids all zero and unused). Forward passes, backward,
clipping, last-layer freezing, optimizer, EMA, BatchNorm behaviour, schedules and checkpoint contents are untouched.

1. Docstring. 2. Imports: `cond_center.*`, `ssl_data.read_rows`; `SSLDataset` no longer imported here (the subclass is).
3. `validate()`: rejects a malformed `cond_center` section (non-boolean flags, unknown or empty field list, non-integer seed).
4. `build_training_parts(..., n_cells=0)`: `CondCenterDINOLoss(..., n_cells=n_cells)` when `n_cells > 0`, else the
   original `DINOLoss` with the original arguments (`**loss_kwargs` is empty then).
5. `train_one_epoch`: the loader yields `(images, cells)`; `cells` go to the device; the diagnostics use the per-row
   center `dino_loss.center_rows(cells)` (2B rows) instead of the global one; the loss call passes `cells` when the
   loss is the per-cell class. `per_cell` is decided once per epoch from the loss object.
6. `HISTORY_COLUMNS`: `cells_seen` before `seconds` (`plots.py` reads columns by name; the extra column is ignored).
7. `run()`: reads the CSV rows next to the paths and immediately drops every column except `relative_path` and the
   cell fields, so the class columns cannot be read by anything that follows (smoke subset sliced identically; a
   mismatch raises); assigns the cells (`assign_cells`, permutation for the control) or all-zero ids with `n_cells = 0`
   when disabled; the manifest
   records the cell definition, the CSV columns read, the number of cells, the numpy version (the control's
   permutation comes from numpy's generator) and a SHA-256 of the id vector.
8. New runs write `cells.csv` (cell id, name, number of images, and whether the mapping is physical or scrambled; in
   the scrambled control the names are nominal).
9. Resume: `cond_center` is added to the manifest keys that must match the saved run (so the cells are identical).
10. `CellSSLDataset(..., cell_ids)` replaces `SSLDataset` (same pixels, same transform, same random stream).
11. `build_training_parts(..., n_cells=n_cells)`.
12. Console: one `teacher centering` line (cells, fields, scrambled or not, CSV columns read, or "one global center").
13. `run_info["cond_center"]` = the manifest record.
14. Per epoch: `cells_seen`, `run_info["cond_center_stats"]` (cells reached, visits per cell = image presentations
    summed over the epochs so far, center norms) and `cell_center_stats.csv` (one row per cell, rewritten every epoch).
15. Epoch summary print: one extra line with the cell statistics.
16. Command paths in printed hints: this folder.
17. Run folder: `make_run_dir()` (2.1) replaces the inline stamp + `mkdir(exist_ok=False)`; same names; cannot change
    results (added 2026-11-10).
(Hunks are numbered by topic; the unified diff is `diffs/train_ssl.py.diff`.)

### 2.4 `smoke_test.py`
A1's sections 1-9 are unchanged. Added: `--config <file>` to test a variant's config; **1b** cells (deterministic
assignment, names equal the fields parsed independently from each image's file name, every image in one cell,
label-leak check with the class column read for the check only: median classes per cell >= 20, every cell >= 2, no
cell with a class above 50 %, cell-only majority vote <= 10 %; scrambled control keeps the histogram and lands images
in their own physical cell at chance rate; dataset item and collation); **4b** the per-cell loss (equals `DINOLoss`
exactly with one cell over 3 steps; after a warm-up step that leaves three different non-zero centers, present cells
move by `0.9 * old + 0.1 * mean of their rows` and the absent cell does not, each teacher row uses its own cell center
with view-major ids and a sample-major gather would give a different loss, wrong id count rejected, buffers in the
state_dict); **5** the real step uses 3 cells and checks the per-cell updates next to the global one, and the epoch-1
step checks the EMA rule from the now non-zero centers; **6** resume
matches on the cell centers and counts, a changed `permute_cells` is rejected, `cells.csv` / `cell_center_stats.csv` /
`cells_seen` written; **7** the export records P4; **8** the fine-tuning record carries the cell definition; **9** no
module loaded from the A1 folder. Added 2026-11-10: section 6 first checks that `make_run_dir()` skips a taken stamp
(two folders asked for within the same second come out distinct, the second with a later stamp).

### 2.5 `cond_center.py` (new)
- `CELL_FIELDS`, `cell_key()`, `assign_cells()` (sorted names, deterministic ids; `permute` shuffles the id vector with
  a seeded generator), `cell_table()`, `columns_read()`.
- `CellSSLDataset(SSLDataset)`: `__getitem__` returns `(views, cell_id)`; the default collate stacks the ids to `(B,)`.
- `CondCenterDINOLoss(DINOLoss)`: buffers `cell_center (n_cells, out_dim)` and `cell_count (n_cells,)`;
  `center_rows(cells)` repeats the ids for the two view-major global views; `forward(..., cells)` is `DINOLoss.forward`
  with `self.center` replaced by `center_rows(cells)`, then the original global center update (kept for the records,
  never subtracted) and the per-cell EMA update of the cells present in the batch (`index_add_` sums, `bincount`
  counts, momentum `center_momentum`). The pilot's `PilotDINOLoss` had the same arithmetic plus optional loss weights;
  the weights are not carried here.

### 2.6 `train_supervised.py` (cannot change results)
The run folder is created by `make_run_dir()` (2.1) instead of the inline stamp + `mkdir(exist_ok=False)`; names
unchanged (added 2026-11-10). Docstring command path. Everything else is A1's.

## 3. Config deltas (relative to A1's `config_ssl.json` / `config_finetune.json`)
`config_ssl.json`: `run_name` `P4_dino_ssl_mobilenetv2`; `_note`; new section `cond_center` = `enabled true`, `cell`
`[scene, band, depression, az_sector30]`, `permute_cells false`, `permute_seed 0`. Every other value is A1's: views
(2 x 224 at [0.4, 1.0], 8 x 96 at [0.15, 0.4], hflip, brightness/contrast 0.4 at p 0.8, blur 1.0/0.1/0.5, no
solarization), head 4096 (no BN, normalized last layer), teacher momentum 0.9995 -> 1, temperatures 0.04/0.1,
center momentum 0.9, AdamW lr 5e-4 per 256 (1.25e-4 at batch 64), min_lr 1e-6, wd 0.04 constant, clip 3.0,
freeze_last_layer 1, fp16, batch 64, 100 epochs, warm-up 10, seed 0, num_workers 2, pixel cache on.
`config_ssl_perm.json`: `run_name` `P4perm_dino_ssl_mobilenetv2`, `permute_cells true`.
`config_ssl_3field.json`: `run_name` `P4d_dino_ssl_mobilenetv2`, `cell` `[scene, band, depression]`.
`config_finetune.json`: `run_name` `P4_dino_mobilenetv2`, notes; all training values A1's (= A0's). `_perm` / `_3field`
differ in `run_name` only, so the fine-tune run folders of the variants never collide.

## 4. Fidelity statement
With `cond_center.enabled = false` this folder reproduces A1 (same code paths, `DINOLoss`, same tensors). With one
cell, `CondCenterDINOLoss` equals `DINOLoss` to 1e-6 on synthetic inputs (smoke test 4b). The dataset returns the
same views as A1's `SSLDataset` for the same index and seed (the subclass only appends the id). Verify the record with
`python Experiments/mobileNetv2_DINO_P4/make_changes_record.py` (fails if a file listed as identical differs).
