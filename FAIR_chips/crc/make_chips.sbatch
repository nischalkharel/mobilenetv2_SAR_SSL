#!/bin/bash
# Cut the FAIR-CSAR chips once on CRC (CPU job): stage the extracted dataset to node disk, run make_chips.py, write the
# array + manifest to FAIR_CHIPS_ROOT on /ix. Submit from the repository root:
#   sbatch --job-name=make_fair_chips FAIR_chips/crc/make_chips.sbatch
# Output: /ix/ageorge/nik96/data/FAIR_chips/{chips_u8.npy (1.4 GB),chips.csv,chips_meta.json,summary.md,previews/}
#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=0-02:00:00
#SBATCH --output=crc/logs/%x_%j.out

set -eo pipefail
source FAIR_chips/crc/env.sh
mkdir -p crc/logs "$FAIR_CHIPS_ROOT"
stage_fair
python FAIR_chips/make_chips.py --workers 16
ls -la "$FAIR_CHIPS_ROOT"
crc-job-stats
