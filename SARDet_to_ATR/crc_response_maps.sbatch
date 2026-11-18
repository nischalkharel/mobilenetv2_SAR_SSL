#!/bin/bash
# Frozen-backbone response maps on ATRNet-STAR test chips for every SARDet-trained backbone (no fine-tuning).
# Submit from the repository root after all SSL runs of the methods in find_backbone.py have finished:
#   sbatch --job-name=xa_maps SARDet_to_ATR/crc_response_maps.sbatch
# Writes SARDet_to_ATR/outputs/compare/response_maps.{md,png,pdf} + the two CSVs (commit them; small).
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-01:00:00
#SBATCH --qos=short
#SBATCH --output=crc/logs/%x_%j.out

set -eo pipefail
source crc/env.sh
mkdir -p crc/logs
python SARDet_to_ATR/response_maps.py --chips 2000 --rows 6 --splits 1,2,3 --figure-split 1 --device cuda
ls -la SARDet_to_ATR/outputs/compare/
crc-job-stats
