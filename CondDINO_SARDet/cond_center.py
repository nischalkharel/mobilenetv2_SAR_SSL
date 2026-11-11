"""Per-cell teacher centering: the ONE change of CondDINO relative to DINO (ported from the ATRNet-STAR P4 code,
mobilenetv2_SAR_SSL/Experiments/mobileNetv2_DINO_P4/cond_center.py; the loss arithmetic is unchanged).

What DINO does (DINO_SARDet/dino.py, DINOLoss): before the sharpened softmax, every teacher output has ONE running
average subtracted, the "center", so that no single output dimension can win for every image. Being far from the
average of the WHOLE pool is cheapest to achieve with the acquisition condition, so a compact backbone learns to
describe the condition first.

What CondDINO does: one running average PER CELL. Every image has its OWN cell's center subtracted; the cell
centers are updated with momentum `cell_momentum` from the teacher outputs of that cell's images in the batch
(only the cells present in the batch move). The global center is still EMA-updated for the record but is never
subtracted. Nothing else in DINO changes: views, head, temperatures, EMA teacher, optimizer, schedules.

Cells on SARDet-100K are defined in cells.py (DESIGN.md section 5): source_pol (manifest columns), perm (the
scrambled control), oracle_class (labels; a leak CEILING, never a method) and radiometric (label-free, from the
pixels). This file only knows integer cell ids.

    assign_cells(rows, fields)      deterministic: sorted cell names -> ids 0..n-1 (labels never read)
    permute_ids(ids, seed)          the size-matched scrambled control (numpy default_rng(seed).permutation)
    cell_table(ids, names)          (id, name, number of images) per cell
    CondCenterDINOLoss(DINOLoss)    forward(student_out, teacher_out, epoch, cells)
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from dino_base import dino  # noqa: E402  (DINO_SARDet/dino.py, loaded once)

DINOLoss = dino.DINOLoss


def cell_key(row: dict, fields) -> str:
    """Same rule as shared/metadata.cell_key: the field values joined with '|'."""
    return "|".join(str(row[f]) for f in fields)


def ids_from_keys(keys):
    """Deterministic ids: the sorted distinct names get ids 0..n-1. Returns (int64 ids, names)."""
    names = sorted(set(keys))
    index = {k: i for i, k in enumerate(names)}
    return np.array([index[k] for k in keys], dtype=np.int64), names


def assign_cells(rows, fields):
    """One cell id per row from the manifest columns `fields` (no labels). Returns (ids, names)."""
    if not fields:
        raise ValueError("at least one cell field is needed")
    missing = [f for f in fields if rows and f not in rows[0]]
    if missing:
        raise KeyError(f"cell fields {missing} are not manifest columns ({sorted(rows[0])})")
    return ids_from_keys([cell_key(r, fields) for r in rows])


def permute_ids(ids: np.ndarray, seed: int) -> np.ndarray:
    """The scrambled control: the SAME id vector in a fixed random order (cell sizes unchanged, image -> cell random)."""
    return ids[np.random.default_rng(int(seed)).permutation(len(ids))]


def cell_table(ids: np.ndarray, names) -> list:
    """(cell id, cell name, number of images) for every cell, including empty ones."""
    counts = np.bincount(ids, minlength=len(names))
    return [(i, names[i], int(counts[i])) for i in range(len(names))]


class CondCenterDINOLoss(DINOLoss):
    """DINOLoss with one EMA center per cell.

    forward(student_output, teacher_output, epoch, cells): cells holds one cell id per IMAGE of the batch (B,).
    The teacher output holds the two global views VIEW-MAJOR (first all images' view 1, then all images' view 2;
    see MultiCropWrapper), so the ids are repeated once for the second view (center_rows). Each teacher row has
    its own cell's center subtracted before the sharpened softmax; everything after that is DINOLoss.forward.

    After the loss: the ORIGINAL global center is still updated (never subtracted here; kept so the history can
    show it), and every cell present in the batch has its center moved with momentum cell_momentum towards the
    mean of its teacher rows. Cells start at zero, like the global center. cell_center and cell_count are
    buffers, so checkpoint.pth carries them and a resumed run continues exactly.

    With n_cells = 1 (every image in the same cell) and cell_momentum = center_momentum this loss equals
    DINOLoss to 1e-6 (checked by smoke_test.py).
    """

    def __init__(self, *args, n_cells: int, cell_momentum: float = None, **kwargs):
        super().__init__(*args, **kwargs)
        if int(n_cells) < 1:
            raise ValueError("CondCenterDINOLoss needs at least one cell")
        self.n_cells = int(n_cells)
        self.cell_momentum = float(self.center_momentum if cell_momentum is None else cell_momentum)
        out_dim = self.center.shape[1]
        self.register_buffer("cell_center", torch.zeros(self.n_cells, out_dim))
        self.register_buffer("cell_count", torch.zeros(self.n_cells, dtype=torch.long))

    def center_rows(self, cells: torch.Tensor) -> torch.Tensor:
        """(2B, out_dim): the center subtracted from each teacher row (view-major, so the ids are repeated)."""
        return self.cell_center[torch.cat([cells, cells])]

    def forward(self, student_output, teacher_output, epoch, cells):
        if cells is None or 2 * len(cells) != len(teacher_output):
            raise ValueError("cells must hold one id per image of the batch (teacher output = 2 global views)")
        if cells.min() < 0 or cells.max() >= self.n_cells:
            raise ValueError(f"cell id outside 0..{self.n_cells - 1}")
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering (per cell) and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center_rows(cells)) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)  # the global center: tracked for the records, never subtracted here
        self.update_cell_centers(teacher_output, cells)
        return total_loss

    @torch.no_grad()
    def update_cell_centers(self, teacher_output, cells):
        ids = torch.cat([cells, cells])
        sums = torch.zeros_like(self.cell_center).index_add_(0, ids, teacher_output.float())
        counts = torch.bincount(ids, minlength=self.n_cells)
        present = counts > 0
        batch_mean = sums[present] / counts[present].unsqueeze(1)
        self.cell_center[present] = self.cell_center[present] * self.cell_momentum + batch_mean * (1 - self.cell_momentum)
        self.cell_count += counts

    def cell_stats(self) -> dict:
        """Cells updated so far, VISITS per cell (image presentations summed over all epochs so far: two teacher
        rows per image, halved) and the sizes of the centers."""
        cnt = self.cell_count.cpu().numpy() // 2
        norms = self.cell_center.norm(dim=1).cpu().numpy()
        seen = cnt > 0
        return {"n_cells": self.n_cells, "cells_seen": int(seen.sum()),
                "visits_per_cell_min": int(cnt.min()), "visits_per_cell_median": float(np.median(cnt)),
                "visits_per_cell_max": int(cnt.max()),
                "center_norm_min_seen": float(norms[seen].min()) if seen.any() else 0.0,
                "center_norm_median_seen": float(np.median(norms[seen])) if seen.any() else 0.0,
                "center_norm_max": float(norms.max()), "global_center_norm": float(self.center.norm())}
