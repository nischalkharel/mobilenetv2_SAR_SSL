# CondDINO_SARDet: proposed changes to shared files (nothing outside this folder was edited)

## 1. `shared/ssl_engine.py`: the run folder hook (DONE by the engine's maintainer on 2026-11-11; used)

The uncommitted engine now sets `method.run_dir = run_dir` before `method.build()`. CondDINO uses it: `cells.csv`,
`cells.json`, `cell_center_stats.csv` (and `cells_radiometric.npz`) are written into the run folder. When an
engine without that line is used, the same files fall back to
`CondDINO_SARDet/outputs/cells/<variant>_split<s>_seed<k>[_SMOKE]_<assignment sha256[:16]>*` and are linked from
the exported backbone file (`exported.cells_csv`, `exported.center_stats_csv`) either way. Please keep the hook
(and its docstring line in the Method interface) when committing.

## 2. `shared/ssl_engine.py`: method-specific manifest / run-record fields (optional)

The ATRNet-STAR P4 run record carried `run_info["cond_center"]` (variant, n_cells, assignment hash, label use,
numpy version). The engine builds `manifests.json` and `run_info.json` itself. Proposed: after `method.build()`,

```python
        extra_record = getattr(method, "record_fields", lambda: {})()
        if extra_record:
            record.update(**extra_record)
```

CondDINO would return `{"cond_center": self.cell_info | {"label_use": args.label_use}}`. The same information is
already in the backbone file's `exported` dict and in `outputs/cells/<stem>.json`, so this is a convenience.

## 3. `shared/metadata.py`: `CELL_DEFINITIONS` (no change needed; a note)

`CELL_DEFINITIONS["source_pol"]` gives 19 cells on the split-1 pool (DESIGN.md section 5 says ~17): the extra
two are `Shipdataset|unknown|unknown` (774 images whose original names carry no satellite / polarisation token)
and `SIVED|airborne-unknown|...` (80 images). DESIGN.md's "17" could be updated to 19, or the two unknown cells
documented as such. `sensor_band` is still listed in CELL_DEFINITIONS although DESIGN.md v2 dropped it; harmless.

## 4. `shared/ssl_engine.py` `merged_config`: `smoke` flag reaches the method (already true; relied upon)

`cfg["smoke"] = smoke` is a top-level non-dict key, so `flat_args()` exposes `args.smoke`. CondDINO uses it to
tag the cell files of smoke runs (`_SMOKE`). If that key is ever renamed, `train_ssl.py` falls back to no tag
(`getattr(args, "smoke", False)`).
