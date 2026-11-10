"""Condition-conditional teacher centering: the one change of experiment P4 relative to A1 (DINO).

What DINO does (dino.py, DINOLoss): before the sharpened softmax, the teacher's output for every image has ONE
running average subtracted, the "center", so that no single output dimension can win for every image (the
anti-collapse device). Being far from the average of the WHOLE pool is cheapest to achieve with the recording
condition (scene, band, depression angle), so a compact backbone learns to describe the condition first.

What P4 does: one running average PER CONDITION CELL. A cell is the combination of sensor-recorded fields of a
chip (default: scene | band | depression | 30-degree azimuth sector). Every image has its OWN cell's center
subtracted; the centers are updated with the same momentum from the teacher outputs of that cell's images in the
batch. Inside a cell the condition is constant, so it can no longer make an image "unusual"; what remains is the
target and its pose. Nothing else in DINO changes: views, head, temperatures, EMA teacher, optimizer, schedules.

Labels are never read here. The fields come from the metadata columns of the SSL CSV (written from the dataset's
file names: band, polarization, depression, azimuth, scene); the class columns are not used for training. A
scrambled control (permute_cells = true, the "P4perm" arm) keeps the cell histogram but assigns cells to images
at random; it must NOT gain anything, otherwise the gain is not coming from the physical grouping.

Pilot evidence (../Independent_Study_For_agents/pilot/RESULTS.md next to this repository, 2026-11-09,
sector-holdout split, 25 epochs):
frozen linear probe on held-out azimuth sectors +4.5 (16-shot) / +5.4 (10 %) over the unchanged recipe; the
scrambled control equalled the unchanged recipe. This folder runs the same mechanism on the full pools for
the real comparison with A1 (100 epochs, splits 1-3, fine-tuning at 1 / 3 / 10 %).
"""
import numpy as np
import torch
import torch.nn.functional as F

from dino import DINOLoss
from ssl_data import SSLDataset

# Fields a cell may be built from, and the CSV column each one reads.
CELL_FIELDS = {
    "scene": "scene",                # urban, factory, sandstone, bare_soil, woodland
    "band": "band",                  # KU or X
    "depression": "depression",      # degrees, as recorded
    "polarization": "polarization",  # HH, HV, VH, VV
    "azimuth": "azimuth",            # exact azimuth in degrees (very small cells; for analysis only)
    "az_sector30": "azimuth",        # azimuth // 30 -> 12 sectors
    "az_sector60": "azimuth",        # azimuth // 60 -> 6 sectors
}


def columns_read(fields) -> list[str]:
    """The SSL CSV columns the cell definition reads (recorded in run_info.json)."""
    return sorted({CELL_FIELDS[f] for f in fields})


def cell_key(row: dict, fields) -> str:
    parts = []
    for f in fields:
        if f == "az_sector30":
            parts.append(str(int(row["azimuth"]) % 360 // 30))
        elif f == "az_sector60":
            parts.append(str(int(row["azimuth"]) % 360 // 60))
        else:
            parts.append(str(row[CELL_FIELDS[f]]))
    return "|".join(parts)


def assign_cells(rows: list[dict], fields, permute: bool = False, seed: int = 0):
    """Returns (cell id per row as an int64 array, cell names in id order).

    Deterministic: the same CSV rows and fields always give the same ids (names are sorted). With permute, the id
    vector is shuffled by a fixed random permutation (seed): every image still has a cell and the histogram of
    cell sizes is unchanged, but image -> cell is random (the P4perm control)."""
    unknown = [f for f in fields if f not in CELL_FIELDS]
    if unknown:
        raise ValueError(f"unknown cell field(s) {unknown}; allowed: {sorted(CELL_FIELDS)}")
    if len(fields) == 0:
        raise ValueError("at least one cell field is needed")
    keys = [cell_key(r, fields) for r in rows]
    names = sorted(set(keys))
    index = {k: i for i, k in enumerate(names)}
    ids = np.array([index[k] for k in keys], dtype=np.int64)
    if permute:
        ids = ids[np.random.default_rng(seed).permutation(len(ids))]
    return ids, names


def cell_table(ids: np.ndarray, names: list[str]) -> list[tuple[int, str, int]]:
    """(cell id, cell name, number of images) for every cell."""
    counts = np.bincount(ids, minlength=len(names))
    return [(i, names[i], int(counts[i])) for i in range(len(names))]


class CellSSLDataset(SSLDataset):
    """SSLDataset that returns (views, cell id) instead of views. The id is an integer tensor; the default
    collate function stacks the ids into a (batch,) tensor and the views into the usual list of view batches."""

    def __init__(self, relative_paths: list[str], transform, cache_file, cell_ids):
        super().__init__(relative_paths, transform, cache_file)
        if len(cell_ids) != len(relative_paths):
            raise ValueError(f"{len(cell_ids)} cell ids for {len(relative_paths)} images")
        self.cell_ids = torch.as_tensor(np.asarray(cell_ids), dtype=torch.long)

    def __getitem__(self, i: int):
        return super().__getitem__(i), self.cell_ids[i]


class CondCenterDINOLoss(DINOLoss):
    """DINOLoss with one EMA center per condition cell.

    forward(student_output, teacher_output, epoch, cells): cells holds one cell id per IMAGE of the batch (B,);
    the teacher output holds the two global views view-major (first all images' view 1, then all images' view 2;
    see MultiCropWrapper), so the ids are repeated once for the second view. Each teacher row has its own cell's
    center subtracted before the sharpened softmax; everything after that is DINOLoss.forward unchanged.

    After the loss: the ORIGINAL global center is still updated (it is never subtracted here; it is kept so the
    diagnostics and the history file can show it), and every cell present in the batch has its center moved with
    momentum center_momentum towards the mean of its teacher rows. Cells start at zero, like the global center.
    cell_center and cell_count are buffers, so checkpoint.pth carries them and a resumed run continues exactly.

    With n_cells = 1 (every image in the same cell) this loss equals DINOLoss exactly (checked by smoke_test.py)."""

    def __init__(self, *args, n_cells: int, **kwargs):
        super().__init__(*args, **kwargs)
        if int(n_cells) < 1:
            raise ValueError("CondCenterDINOLoss needs at least one cell")
        self.n_cells = int(n_cells)
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
        # ema update, same momentum as the global center
        self.cell_center[present] = self.cell_center[present] * self.center_momentum + batch_mean * (1 - self.center_momentum)
        self.cell_count += counts

    def cell_stats(self) -> dict:
        """How many cells have been updated so far, VISITS per cell (image presentations summed over all epochs so
        far: two teacher rows per image, halved; after E full epochs a cell with n images has n x E visits), and
        the sizes of the centers. The number of distinct images per cell is in the run's cells.csv."""
        cnt = self.cell_count.cpu().numpy() // 2
        norms = self.cell_center.norm(dim=1).cpu().numpy()
        seen = cnt > 0
        return {"n_cells": self.n_cells, "cells_seen": int(seen.sum()),
                "visits_per_cell_min": int(cnt.min()), "visits_per_cell_median": float(np.median(cnt)),
                "visits_per_cell_max": int(cnt.max()),
                "center_norm_min_seen": float(norms[seen].min()) if seen.any() else 0.0,
                "center_norm_median_seen": float(np.median(norms[seen])) if seen.any() else 0.0,
                "center_norm_max": float(norms.max()), "global_center_norm": float(self.center.norm())}
