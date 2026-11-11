# Notes for the maintainer of shared/ (SparK_SARDet, 2026-11-11)

Nothing in `shared/` was edited. SparK_SARDet works with the engine as it is; the points below are proposals.

## 1. Proposed: hand the run folder to the method (`shared/ssl_engine.py`)

SparK writes `reconstruction_epoch<NNN>.png` and `param_groups.txt` into the run folder at the end of every epoch,
but `run()` never passes `run_dir` to the method (`build(args, rows, n_steps, device)` and
`train_epoch(loader, epoch, device, log_every, timing)` do not carry it). `SparKMethod._find_run_dir()` therefore
reads it from the `RunRecord` captured in the engine's `timing` closure (`record.run_dir`), after first checking
`method.run_dir`. That works but depends on an engine internal. Proposed one-line change, right after
`method = method_cls(args)` is no longer possible (the folder is made later), so right after the run folder exists:

```python
    # ssl_engine.run(), just before `try:` (both the new-run and the resume branch have run_dir by then)
    method.run_dir = run_dir   # methods that write per-epoch files (SparK previews) use it; others ignore it
```

With that line in place `SparKMethod` uses `self.run_dir` and the closure lookup is never reached. No change to
the Method interface docstring is needed beyond one sentence ("the engine sets `method.run_dir` before build()").

## 2. Observation: a kill between `checkpoint.pth` and `history.csv`

`run()` writes `checkpoint.pth` (atomic replace), then appends the epoch's row to `history.csv`, then updates
`run_info.json`. A kill in that window (CRC time limit) leaves a checkpoint at epoch E with E-1 history rows; a
resume continues from E and `history.csv` is missing row E for good (charts and the cost record lag by one epoch;
training itself is unaffected). The ATRNet-STAR A3 code stored the history row and cost inside the checkpoint and
rebuilt them on resume (`repair_after_stop`). Proposed: add `"history_row": [...]` and `"cost": cost` to the
checkpoint dict and, on resume, append the row when `history.csv` has `epoch - 1` rows. Low priority: the sbatch
scripts stop with `--stop-after-epoch`, which exits cleanly.

## 3. Observation: `shared/plots.py` `ssl_curves` schedule panel

The third panel plots `["lr", "wd", "momentum", "teacher_temp"]` on a log axis; SparK has only `lr` and `wd`, which
is fine (missing keys are skipped). SparK's `grad_norm` goes on the diagnostics panel with `feature_cosine` /
`feature_std` (different scales, same panel). If a fourth panel for method-specific extras is ever added, `grad_norm`
would belong there. No change requested.

## 4. Precision flags are process-global

`SparKMethod.build()` sets `torch.backends.cudnn.allow_tf32 = False` and `torch.backends.cuda.matmul.allow_tf32 =
False` (true fp32, see CHANGES.md section 6) and never resets them, so anything that runs later in the SAME process
also runs with TF32 off. The fine-tune protocol says "fp32 (PyTorch default TF32 flags, recorded)". Real fine-tune
runs are separate processes (`run_finetune.py`), so they are unaffected; `SparK_SARDet/smoke_test.py` restores the
two flags itself before its fine-tune step. If the engine ever wants to guarantee the defaults for every method, it
could save and restore the two flags around training in `run()`; not requested.
