#!/bin/bash
# Propose the label-free region boxes for the whole SSL pool on a CPU node (once; the file is reused by every
# RegionDINO run of every split). Submit from the SARDet suite root, like every other job:
#   cd /ihome/ageorge/nik96/mobilenetv2_SAR_SSL/SARDet && sbatch --job-name=regions ../RegionSSL/crc/make_regions.sbatch
# Writes $REGIONSSL_REGIONS_DIR/regions.npz (default RegionSSL/regions/) + regions_summary.json (commit the json).
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
python ../RegionSSL/make_regions.py --workers "$SLURM_CPUS_PER_TASK"
ls -la ../RegionSSL/regions/
crc-job-stats
