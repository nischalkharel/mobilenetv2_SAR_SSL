"""CondDINO SSL on the SARDet-100K pool: DINO with the teacher center kept PER CELL (DESIGN.md section 5).

    python CondDINO_SARDet/train_ssl.py                                             # source_pol cells (config_ssl.json)
    python CondDINO_SARDet/train_ssl.py --config CondDINO_SARDet/config_ssl_perm.json          # scrambled control
    python CondDINO_SARDet/train_ssl.py --config CondDINO_SARDet/config_ssl_oracle_class.json  # LABEL-LEAK CEILING
    python CondDINO_SARDet/train_ssl.py --config CondDINO_SARDet/config_ssl_radiometric.json   # label-free cells
    python CondDINO_SARDet/train_ssl.py --smoke | --resume <run folder> | --stop-after-epoch N  (as for DINO)

CondDINOMethod extends DINO_SARDet's DINOMethod (imported through dino_base.py, nothing copied): build() swaps
the loss for cond_center.CondCenterDINOLoss and assigns one cell id per pool row (cells.py); train_epoch() is the
DINO loop with the batch's cell ids gathered from the row indices the shared loader yields; the checkpoint
carries the cell centers; the TEACHER backbone is exported. Views, head, temperatures, EMA, optimizer, schedules
and everything the engine does are DINO's.

Cell files: the engine sets method.run_dir before build() (shared/ssl_engine.py), so the run folder gets
cells.csv (cell id, name, images, mapping, label use), cells.json (the cell record incl. the assignment sha256)
and cell_center_stats.csv (one row per cell: visits and center norm, rewritten after every epoch; a resume
rewrites cells.csv with the same content). If an older engine does not set run_dir, the same files go to
outputs/cells/<variant>_split<s>_seed<k>[_SMOKE]_<assignment sha256[:16]>*.csv. Their paths and the assignment
hash are recorded in the exported backbone file ("exported" metadata) and history.csv carries cells_seen per
epoch. Labels: only the oracle_class variant reads them, and it says so in its config, its cells.csv and the
backbone metadata (label_use).
"""
import csv
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from shared import paths, ssl_engine  # noqa: E402
from shared.ssl_data import to_network_input  # noqa: E402

sys.path.insert(0, str(HERE))
from dino_base import DINOMethod, DINO_SOURCE, dino  # noqa: E402
from cells import VARIANTS, assign_variant, output_dir  # noqa: E402
from cond_center import CondCenterDINOLoss, cell_table  # noqa: E402

clip_gradients, cancel_gradients_last_layer, collapse_diagnostics = (dino.clip_gradients, dino.cancel_gradients_last_layer,
                                                                     dino.collapse_diagnostics)


class CondDINOMethod(DINOMethod):
    name = "conddino"
    history_columns = DINOMethod.history_columns + ["cells_seen"]
    export_rule = "teacher backbone, final epoch (as DINO)"

    def __init__(self, args):
        super().__init__(args)
        cc_keys = ("enabled", "cells", "permute_seed", "center_momentum_cell", "label_use")
        missing = [k for k in cc_keys if not hasattr(args, k)]
        if missing:
            raise ValueError(f"config section cond_center lacks {missing}")
        if not isinstance(args.enabled, bool):
            raise ValueError("cond_center.enabled must be true or false")
        self.variant = args.cells if args.enabled else "none"
        if args.enabled and args.cells not in VARIANTS:
            raise ValueError(f"cond_center.cells must be one of {sorted(VARIANTS)}")
        if not isinstance(args.permute_seed, int):
            raise ValueError("cond_center.permute_seed must be an integer")
        if not 0 <= float(args.center_momentum_cell) < 1:
            raise ValueError("cond_center.center_momentum_cell must be in [0, 1)")
        expected = VARIANTS[args.cells]["label_use"] if args.enabled else "none"
        if args.label_use != expected:  # the config must state the truth about labels, word for word
            raise ValueError(f"cond_center.label_use must be {expected!r} for cells={self.variant!r}, got {args.label_use!r}")

    def describe(self, args):
        lab = f"; LABELS USED: {args.label_use}" if self.variant == "oracle_class" else f"; labels: {args.label_use}"
        return (super().describe(args) + f"\n  CondDINO      : cells = {self.variant} ({VARIANTS.get(self.variant, {}).get('what', 'one cell = DINO')}); "
                f"{self.loss.n_cells} cells, cell momentum {self.loss.cell_momentum}{lab}\n"
                f"  cell files    : {self.cells_csv}")

    # ---- build: DINO's networks/optimizer/schedules, then the per-cell loss and the cell assignment ----------------
    def build(self, args, rows, n_steps, device):
        super().build(args, rows, n_steps, device)
        ids, names, info = assign_variant(self.variant, rows, permute_seed=args.permute_seed, num_workers=args.num_workers)
        if len(ids) != len(rows):
            raise ValueError("one cell id per pool row expected")
        self.cell_ids = torch.as_tensor(ids, dtype=torch.long, device=device)
        self.cell_names, self.cell_info = names, info
        self.assignment_sha256 = info["cell_assignment_sha256"]
        self.loss = CondCenterDINOLoss(args.out_dim, args.local_crops_number + 2, args.warmup_teacher_temp, args.teacher_temp,
                                       args.warmup_teacher_temp_epochs, args.epochs, student_temp=args.student_temp,
                                       center_momentum=args.center_momentum, n_cells=len(names),
                                       cell_momentum=args.center_momentum_cell).to(device)
        run_dir = getattr(self, "run_dir", None)   # set by shared/ssl_engine.py before build(); else outputs/cells/
        if run_dir:
            self.cells_dir, stem = Path(run_dir), "cells"
        else:
            self.cells_dir = output_dir("cells")
            stem = f"{self.variant}_split{paths.SPLIT}_seed{args.seed}{'_SMOKE' if getattr(args, 'smoke', False) else ''}_{self.assignment_sha256[:16]}"
        self.cells_csv = self.cells_dir / f"{stem}.csv"
        self.center_stats_csv = self.cells_dir / f"{stem.replace('cells', 'cell')}_center_stats.csv"
        mapping = {"perm": "scrambled control: image -> cell is random, the names are nominal",
                   "oracle_class": "LABEL-LEAK CEILING: cell = classes present in the image (labels used)",
                   "none": "one cell (= DINO)"}.get(self.variant, "physical: every image is in the cell named by its own data")
        with open(self.cells_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["cell_id", "cell", "num_images", "mapping", "label_use", "variant"])
            w.writerows([(*row, mapping, args.label_use, self.variant) for row in cell_table(ids, names)])
        with open(self.cells_dir / f"{stem}.json", "w", encoding="utf-8") as f:
            json.dump({"cell_info": info, "label_use": args.label_use, "split": paths.SPLIT, "seed": args.seed,
                       "num_rows": len(rows), "dino_source": DINO_SOURCE, "written": datetime.now().isoformat(timespec="seconds")},
                      f, indent=2, default=str)
        if self.variant == "radiometric" and info.get("cache_file"):
            import shutil
            shutil.copy(info["cache_file"], self.cells_dir / f"{stem}_radiometric.npz")

    def _write_center_stats(self, epoch):
        norms = self.loss.cell_center.norm(dim=1).cpu().numpy()
        visits = self.loss.cell_count.cpu().numpy() // 2
        with open(self.center_stats_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["cell_id", "cell", "visits", "center_norm", "epochs_done", "written"])
            stamp = datetime.now().isoformat(timespec="seconds")
            for cid, name in enumerate(self.cell_names):
                w.writerow([cid, name, int(visits[cid]), f"{norms[cid]:.5f}", epoch + 1, stamp])

    # ---- the DINO loop with cell ids (DINO_SARDet/train_ssl.py train_epoch, plus the marked lines) -----------------
    def train_epoch(self, loader, epoch, device, log_every, timing):
        args = self.args
        self.student.train()
        self.teacher.train()
        sums, t0 = {}, time.time()
        n = len(loader)
        for it, (views, idx) in enumerate(loader):
            step = it + 1
            git = n * epoch + it
            for i, g in enumerate(self.optimizer.param_groups):
                g["lr"] = self.lr_schedule[git]
                if i == 0:
                    g["weight_decay"] = self.wd_schedule[git]
            images = [to_network_input(v, device) for v in views]
            cells = self.cell_ids[idx.to(device)]                        # CondDINO: one cell id per image of the batch
            center = self.loss.center_rows(cells)                        # CondDINO: the per-row centers this step uses
            with torch.autocast(device.type, dtype=torch.float16, enabled=self.scaler is not None):
                teacher_output = self.teacher(images[:2])
                student_output = self.student(images)
                loss = self.loss(student_output, teacher_output, epoch, cells)   # CondDINO
            if not math.isfinite(loss.item()):
                raise RuntimeError(f"loss is {loss.item()}, stopping")
            self.optimizer.zero_grad()
            if self.scaler is None:
                loss.backward()
                if args.clip_grad:
                    clip_gradients(self.student, args.clip_grad)
                cancel_gradients_last_layer(epoch, self.student, args.freeze_last_layer)
                self.optimizer.step()
            else:
                self.scaler.scale(loss).backward()
                if args.clip_grad:
                    self.scaler.unscale_(self.optimizer)
                    clip_gradients(self.student, args.clip_grad)
                cancel_gradients_last_layer(epoch, self.student, args.freeze_last_layer)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            with torch.no_grad():
                m = self.momentum_schedule[git]
                for pq, pk in zip(self.student.parameters(), self.teacher.parameters()):
                    pk.data.mul_(m).add_((1 - m) * pq.detach().data)
            diag = collapse_diagnostics(teacher_output, center, self.loss.teacher_temp_schedule[epoch],
                                        self.teacher.last_features, len(images[0]))
            for k, v in {"loss": loss.item(), **diag}.items():
                sums[k] = sums.get(k, 0.0) + v
            timing(step)
            if log_every and (step % log_every == 0 or step == n):
                print(f"    step {step:5d}/{n}  loss {sums['loss'] / step:.4f}  lr {self.optimizer.param_groups[0]['lr']:.2e}  "
                      f"teacher entropy {sums['teacher_entropy'] / step:.3f}  {time.time() - t0:.0f}s", flush=True)
        cs = self.loss.cell_stats()
        self._write_center_stats(epoch)
        print(f"  cells: {cs['cells_seen']}/{cs['n_cells']} updated; visits per cell so far min/median/max "
              f"{cs['visits_per_cell_min']}/{cs['visits_per_cell_median']:.0f}/{cs['visits_per_cell_max']}; "
              f"cell center norm median {cs['center_norm_median_seen']:.3f} (global {cs['global_center_norm']:.3f})", flush=True)
        return {k: v / n for k, v in sums.items()} | {
            "lr": self.optimizer.param_groups[0]["lr"], "wd": self.optimizer.param_groups[0]["weight_decay"], "momentum": float(m),
            "teacher_temp": float(self.loss.teacher_temp_schedule[epoch]), "center_norm": self.loss.center.norm().item(),
            "cells_seen": cs["cells_seen"]}

    # ---- checkpoint / export --------------------------------------------------------------------------------------
    def state_dict(self):
        return super().state_dict() | {"cell_variant": self.variant, "cell_assignment_sha256": self.assignment_sha256,
                                       "n_cells": self.loss.n_cells}

    def load_state_dict(self, d):
        if d.get("cell_variant") != self.variant or d.get("n_cells") != self.loss.n_cells:
            raise ValueError(f"cannot resume: cells differ ({d.get('cell_variant')}/{d.get('n_cells')} saved, "
                             f"{self.variant}/{self.loss.n_cells} rebuilt)")
        if d.get("cell_assignment_sha256") != self.assignment_sha256:
            raise ValueError("cannot resume: the cell assignment changed since the run started")
        super().load_state_dict(d)   # includes loss.state_dict(): center, cell_center, cell_count

    def export_state(self):
        state, extra = super().export_state()
        extra.update(cell_variant=self.variant, n_cells=self.loss.n_cells, cell_assignment_sha256=self.assignment_sha256,
                     label_use=self.args.label_use, cells_csv=str(self.cells_csv), center_stats_csv=str(self.center_stats_csv),
                     cell_momentum=self.loss.cell_momentum, cell_stats_final=self.loss.cell_stats(), dino_source=DINO_SOURCE,
                     what="DINO with the teacher center kept per cell (CondDINO); the global center was tracked but never subtracted")
        return state, extra


if __name__ == "__main__":
    ssl_engine.cli(CondDINOMethod, HERE)
