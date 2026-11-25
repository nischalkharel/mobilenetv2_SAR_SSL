#!/bin/bash
# Gradient-content regions (DINOSAR's content map) for the whole SARDet-100K pool, once, on a CPU node. Submit from
# the SARDet suite root, like RegionSSL/crc/make_regions.sbatch:
#   cd /ihome/ageorge/nik96/mobilenetv2_SAR_SSL/SARDet && sbatch --job-name=regions_grad ../GradientRegions/crc/regions_sardet.sbatch
# Writes ../GradientRegions/outputs/regions_sardet/regions.npz (not committed) + regions_summary.json,
# regions_box_agreement.json and regions_preview.png (commit those). About 94k scenes; the map is cheap (two box
# filters on a 128-px grid), expect well under an hour with 16 workers.
#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=0-06:00:00
#SBATCH --output=crc/logs/%x_%j.out

set -eo pipefail
source crc/env.sh
mkdir -p crc/logs
export STAGE_DATA=0          # one pass over the files; no node-local copy needed
python ../GradientRegions/make_regions.py --pool sardet --diagnose --preview 12 --workers "$SLURM_CPUS_PER_TASK"
ls -la ../GradientRegions/outputs/regions_sardet/
crc-job-stats
