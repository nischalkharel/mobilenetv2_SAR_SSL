# Notes for the maintainer of shared/ (nothing in shared/ was edited by the LeJEPA_SARDet port)

LeJEPA_SARDet works with the shared code as it is. Two small, optional observations:

1. `shared/ssl_engine.py` docstring, line 17: the Method interface lists
   `train_epoch(loader, epoch, device, log_every)` but `run()` calls
   `method.train_epoch(loader, epoch, device, args.log_every_steps, timing)` (five arguments). Proposed docstring
   line: `train_epoch(loader, epoch, device, log_every, timing)  -> dict of stats (must include every history
   column); call timing(step) once per step`.

2. `shared/ssl_engine.py`, `TIMING_FROM, TIMING_TO = 5, 25`: a smoke run has 6 steps per epoch, so the
   "measured speed" line and `measured_seconds_per_step` in run_info.json never appear in smoke mode (the
   seconds column of history.csv is the only timing). If a smoke-mode speed figure is wanted, a possible change is
   to take the marks at `TIMING_FROM = min(5, n_steps // 2)` and `TIMING_TO = min(25, n_steps)` inside `run()`.
   Full runs (1,469 steps per epoch at batch 64) are not affected.
