#!/bin/bash
# Cut the SARDet-100K chips once on CRC (CPU job): stage the dataset tar to node disk, run make_chips.py, write the
# array + manifest to SARDET_CHIPS_ROOT on /ix. Submit from the repository root:
#   sbatch --job-name=make_chips SARDet_chips/crc/make_chips.sbatch
# About 10 min (staging dominates). Output: /ix/ageorge/nik96/data/SARDet_chips/{chips_u8.npy,chips.csv,chips_meta.json,summary.md,previews/}
#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=0-02:00:00
#SBATCH --output=crc/logs/%x_%j.out

set -eo pipefail
source SARDet_chips/crc/env.sh
mkdir -p crc/logs "$SARDET_CHIPS_ROOT"
stage_dataset_from_repo_root          # SARDet/crc/env.sh stage_dataset: one sequential tar read; sets SARDET100K_ROOT to the node copy
python SARDet_chips/make_chips.py --workers 16
ls -la "$SARDET_CHIPS_ROOT"
crc-job-stats
