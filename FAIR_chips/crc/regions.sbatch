#!/bin/bash
# Propose the label-free regions for one FAIR-CSAR pool, once (CPU job). Submit from the repository root:
#   sbatch --job-name=regions_fair_tiles FAIR_chips/crc/regions.sbatch fair_tiles
#   sbatch --job-name=regions_fair_down  FAIR_chips/crc/regions.sbatch fair_down
# Writes RegionSSL/regions/regions_<pool>.npz (not committed) + regions_<pool>_summary.json (commit it).
# The tiles pool is 43,076 images at 512 px: about 15 min with 16 workers; the down pool a quarter of that.
#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=0-03:00:00
#SBATCH --output=crc/logs/%x_%j.out

set -eo pipefail
POOL=${1:?fair_tiles | fair_down}
source FAIR_chips/crc/env.sh
mkdir -p crc/logs
stage_fair
python RegionSSL/make_regions.py --pool "$POOL" --workers "$SLURM_CPUS_PER_TASK"
ls -la RegionSSL/regions/ | grep fair
crc-job-stats
