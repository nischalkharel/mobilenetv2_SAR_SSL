#!/bin/bash
# Teacher region proposals for the whole SARDet-100K pool (GPU job, once per setting). Submit from the repository root:
#   sbatch --job-name=teacher_regions TeacherRegions/crc/propose.sbatch            # default setting: zoom 2, top 2 % (DESIGN.md section 3)
#   sbatch --job-name=teacher_regions TeacherRegions/crc/propose.sbatch --zoom 1 --top-q 5   # another setting, own output folder
# Writes TeacherRegions/outputs/regions_cls/regions.npz (+ the norm-signal twin and the summary json), about 3 to 4 h on one A100.
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-06:00:00
#SBATCH --qos=short
#SBATCH --output=crc/logs/%x_%j.out

set -eo pipefail
source TeacherRegions/crc/env.sh
mkdir -p crc/logs
stage_dataset_from_repo_root
python TeacherRegions/teacher_saliency.py --device cuda --diagnose "$@"
crc-job-stats
