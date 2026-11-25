#!/bin/bash
# FAIR-CSAR chip study on CRC: one backbone -> chip classification fine-tunes on the 1 %, 3 % and 10 % labeled sets of
# one split, each scored once on test and test_clean. Submit from the repository root:
#   sbatch --job-name=fch_imagenet_s1 FAIR_chips/crc/finetune.sbatch imagenet 1
#   sbatch --job-name=fch_ro_t_s2     FAIR_chips/crc/finetune.sbatch fair_ro_tiles 2
#   methods: imagenet | fair_{dino,ro,ror}_{tiles,down} | fair_{dino,ro,ror,rot,rotr,rog,rogr}_tiles500 | sardet_dino | sardet_regiondino_o
#            | sardet_regiondino_o_random; splits 1 2 3
# The SSL backbone is the completed SSL run of the same split (SARDet_to_ATR/find_backbone.py); its SHA-256 goes into
# run_info.json. Budgets run largest first; a budget whose run folder already has eval_test/metrics.json is skipped.
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-03:00:00
#SBATCH --qos=short
#SBATCH --output=crc/logs/%x_%j.out

set -eo pipefail
METHOD=${1:?imagenet | fair_<dino|ro|ror>_<tiles|down> | fair_<dino|ro|ror|rot|rotr|rog|rogr>_tiles500 | sardet_dino | sardet_regiondino_o | sardet_regiondino_o_random}
SPLIT=${2:?split 1, 2 or 3}
source FAIR_chips/crc/env.sh
mkdir -p crc/logs
stage_fair_chips
for BUDGET in 10pct 3pct 1pct; do
    echo "================ $METHOD split $SPLIT $BUDGET ================"
    RUN=$(ls -td FAIR_chips/outputs/finetune/*_F1_${METHOD}_mobilenetv2_${BUDGET}_split${SPLIT}_seed0 2>/dev/null | head -1) || true
    if [ -n "$RUN" ] && [ -f "$RUN/eval_test/metrics.json" ]; then echo "already done: $RUN"; continue; fi
    python FAIR_chips/finetune_chips.py --method "$METHOD" --budget "$BUDGET" --split "$SPLIT"
done
crc-job-stats
