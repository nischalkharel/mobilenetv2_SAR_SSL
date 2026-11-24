#!/bin/bash
# Teacher (frozen SARMAE) region proposals for the FAIR-CSAR tiles pool: one GPU job, once. Submit from the repository root:
#   sbatch --job-name=fair_teacher_regions FAIR_chips/crc/teacher_regions.sbatch
# Writes TeacherRegions/outputs/regions_cls_fair_tiles/regions_fair_tiles.npz (+ the norm-signal twin folder and the
# summary / box-agreement json), same proposer setting as the SARDet run (zoom 2, top 2 %). 19,036 tiles: about 1 h.
# The 500-epoch SSL configs config_ssl_rot_tiles500.json / config_ssl_rotr_tiles500.json read that folder (regions_dir).
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-04:00:00
#SBATCH --qos=short
#SBATCH --output=crc/logs/%x_%j.out

set -eo pipefail
source FAIR_chips/crc/env.sh
source TeacherRegions/crc/env.sh          # SARMAE_CHECKPOINT (and its existence check)
mkdir -p crc/logs
stage_fair
python TeacherRegions/teacher_saliency.py --pool fair_tiles --device cuda --diagnose --preview 12 \
    --out-dir TeacherRegions/outputs/regions_cls_fair_tiles "$@"
crc-job-stats
