#!/bin/bash
# Chip study S1 on CRC: one backbone -> chip classification fine-tunes on the 1 %, 3 % and 10 % labeled sets of one
# split, each scored once on test and test_clean. 24 jobs cover the study (8 methods x 3 splits).
# Submit from the repository root:
#   sbatch --job-name=ch_imagenet_s1 SARDet_chips/crc/finetune.sbatch imagenet 1
#   sbatch --job-name=ch_dino_s2 SARDet_chips/crc/finetune.sbatch dino 2
#   methods: imagenet dino lejepa spark regiondino_so regiondino_so_random regiondino_o regiondino_o_random
#            regiondino_o_teacher regiondino_o_teacher_random (TeacherRegions backbones); splits 1 2 3
# The SSL backbone is the scene SSL run of the same split (SARDet_to_ATR/find_backbone.py); its SHA-256 goes into
# run_info.json. Budgets run largest first so the long one starts early; a budget whose run folder already has
# eval_test/metrics.json is skipped (safe to resubmit).
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-03:00:00
#SBATCH --qos=short
#SBATCH --output=crc/logs/%x_%j.out

set -eo pipefail
METHOD=${1:?imagenet | dino | lejepa | spark | regiondino_so[_random] | regiondino_o[_random] | regiondino_o_teacher[_random]}
SPLIT=${2:?split 1, 2 or 3}
source SARDet_chips/crc/env.sh
mkdir -p crc/logs
stage_chips
for BUDGET in 10pct 3pct 1pct; do
    echo "================ $METHOD split $SPLIT $BUDGET ================"
    RUN=$(ls -td SARDet_chips/outputs/finetune/*_S1_${METHOD}_mobilenetv2_${BUDGET}_split${SPLIT}_seed0 2>/dev/null | head -1) || true
    if [ -n "$RUN" ] && [ -f "$RUN/eval_test/metrics.json" ]; then echo "already done: $RUN"; continue; fi
    python SARDet_chips/finetune_chips.py --method "$METHOD" --budget "$BUDGET" --split "$SPLIT"
done
crc-job-stats
