#!/bin/bash
# Gradient-content regions (DINOSAR's content map) for the FAIR-CSAR tiles pool, once, on a CPU node. Submit from the
# repository root:
#   sbatch --job-name=regions_grad_fair GradientRegions/crc/regions_fair.sbatch
# Writes GradientRegions/outputs/regions_fair_tiles/regions_fair_tiles.npz (not committed) + the summary, box-agreement
# and preview files (commit those). 19,036 tiles: a few minutes with 16 workers.
#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=0-03:00:00
#SBATCH --output=crc/logs/%x_%j.out

set -eo pipefail
source FAIR_chips/crc/env.sh
mkdir -p crc/logs
stage_fair
python GradientRegions/make_regions.py --pool fair_tiles --diagnose --preview 12 --workers "$SLURM_CPUS_PER_TASK"
ls -la GradientRegions/outputs/regions_fair_tiles/
crc-job-stats
