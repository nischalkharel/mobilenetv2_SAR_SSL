# Results tracker

A small local web app that reads every run folder in this repository and shows the study as six pages:
Overview (experiment map and headline answers), Classification (ATRNet-STAR), Detection (SARDet-100K),
Response maps, Runs (every SSL and fine-tune folder) and Ideas (an idea box saved in the repository).

## Run it

```
.venv\Scripts\python.exe tracker\serve.py
```

That collects the data (a few seconds), starts a server on http://localhost:8765/ and opens it in the default
browser. Stop it with Ctrl-C. Options: `--port 9000`, `--no-browser`, `--no-collect`.
The "refresh" link in the page header re-runs the collection after new runs are pulled.
Standard library only; nothing to install.

## What is where

| file | role |
|---|---|
| `registry.py` | one record per method row: id, label, colour family, unlabeled pool, control flag, reference row, one-sentence note |
| `collect.py` | reads run folders and the two compare outputs, writes `data/*.json` (git-ignored, regenerated on every start) |
| `serve.py` | serves `web/`, `data/`, repository images under `/file/`, and the idea-box API |
| `web/app.js`, `web/app.css` | shared chart code (plain SVG), tables, tooltip, theme |
| `web/*.html` | the pages; each has its own short script at the bottom |
| `ideas.json` | the idea box, committed with the repository so it travels with the code |

## Where the numbers come from

- Classification: `SARDet_to_ATR/compare.py`'s folder rules (`ROOTS`, `row_key`) applied to the run folders;
  accuracy from `eval_test/metrics.json`, run details from `run_info.json`, curves from `history.csv`.
- Detection: the newest `SARDet/compare/outputs/<timestamp>/` (runs.csv, per_source.csv) plus each run's
  `eval_best_test/metrics.json` for per-class AP. Re-run `SARDet/compare/compare.py` from the suite root (or
  `collect.py --run-compare`) after pulling new detection runs.
- Response maps: `SARDet_to_ATR/outputs/compare/response_maps_per_model.csv` (chips) and
  `RegionSSL/outputs/scene_maps/scene_maps_per_*.csv` (scenes); the figures are served at full resolution.
- Runs: every `outputs/ssl/*` and `outputs/finetune/*` folder with a `run_info.json`, minus smoke tests and stubs.

Adding a new method row: add its record to `registry.py` (and, for classification, its folder name to the
mapping in `SARDet_to_ATR/compare.py`); nothing in the pages needs to change.

The hand-written sentences on the Overview page ("Reading") are in `web/index.html`; update them when a result changes the story.
