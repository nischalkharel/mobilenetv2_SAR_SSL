#!/bin/bash
# Propose the label-free region boxes for the ATRNet-STAR SSL pool (union of the three splits' pools, ~66k chips at
# their native 128 px) on a CPU node, once. Submit from the suite root:
#   cd /ihome/ageorge/nik96/mobilenetv2_SAR_SSL/SARDet && sbatch --job-name=regions_atr ../RegionSSL/crc/make_regions_atr.sbatch
# Writes RegionSSL/regions/regions_atrnet.npz + regions_atrnet_summary.json (commit the json).
#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=0-03:00:00
#SBATCH --output=crc/logs/%x_%j.out

set -eo pipefail
export STAGE_DATA=0
source crc/env.sh
source ../crc/env.sh
mkdir -p crc/logs
python ../RegionSSL/make_regions.py --pool atrnet --workers "$SLURM_CPUS_PER_TASK"
ls -la ../RegionSSL/regions/
crc-job-stats
