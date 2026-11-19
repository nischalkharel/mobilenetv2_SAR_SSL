#!/bin/bash
# Response maps of the frozen SARDet-pool backbones on SARDet-100K test scenes (RegionSSL/scene_maps.py), all three
# splits, one figure for split 1. Submit from the suite root:
#   cd /ihome/ageorge/nik96/mobilenetv2_SAR_SSL/SARDet && sbatch --job-name=scene_maps ../RegionSSL/crc/scene_maps.sbatch
# Reads 1500 test scenes from /ix (one pass, no staging needed) and writes RegionSSL/outputs/scene_maps/.
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=0-01:00:00
#SBATCH --qos=short
#SBATCH --output=crc/logs/%x_%j.out

set -eo pipefail
export STAGE_DATA=0
source crc/env.sh
mkdir -p crc/logs
python ../RegionSSL/scene_maps.py --images 1500 --rows 6 --splits 1,2,3 --figure-split 1 --device cuda
crc-job-stats
