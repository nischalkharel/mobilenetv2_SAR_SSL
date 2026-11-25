#!/bin/bash
# Cross-dataset study, arm A, on Pitt CRC: one SARDet-100K SSL backbone -> ATRNet-STAR fine-tunes on the 1 %, 3 %
# and 10 % labeled sets of one split, validation + test for each. Nine jobs cover the study (3 methods x 3 splits).
# Submit from the repository root:
#   sbatch --job-name=xa_dino_s1 SARDet_to_ATR/crc.sbatch dino 1
#   sbatch --job-name=xa_lejepa_s2 SARDet_to_ATR/crc.sbatch lejepa 2      (methods: dino lejepa spark; splits 1 2 3)
#   sbatch --job-name=xf_ro5_s1 SARDet_to_ATR/crc.sbatch fair_ro_tiles500 1   (arm F: the FAIR_chips 500-epoch backbones)
# The backbone is the SARDet SSL run of the same split number (find_backbone.py); its SHA-256 goes into every run
# record. Test-set rule as always: decisions on validation only; test scores recorded, never used to choose.
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-04:00:00
#SBATCH --qos=short
#SBATCH --output=crc/logs/%x_%j.out

set -eo pipefail
METHOD=${1:?dino | lejepa | spark | regiondino_o[_random] | regiondino_so[_random] | atr_dino | atr_regiondino_* | fair_<dino|ro|ror|rot|rotr>_tiles500}
export SAR_SPLIT=${2:?split 1, 2 or 3}
source crc/env.sh
mkdir -p crc/logs
stage_atrnet                   # copy the SOC-40 chips to the node disk: the test pass reads them per batch
BACKBONE=$(python SARDet_to_ATR/find_backbone.py "$METHOD" "$SAR_SPLIT")
echo "backbone: $BACKBONE  sha256 $(sha256sum "$BACKBONE" | cut -d' ' -f1)"
TAG=""; [ "$SAR_SPLIT" = 1 ] || TAG="_split$SAR_SPLIT"
# Run-name stem written by finetune_atr.py: XA_<m>_sardetssl_ for the SARDet backbones, XI_<m>_atrssl_ for the
# atr_* methods (arm I: the RegionSSL engine on the ATRNet pool; see finetune_atr.py).
case "$METHOD" in atr_*) STEM="XI_${METHOD#atr_}_atrssl_mobilenetv2" ;; fair_*) STEM="XF_${METHOD#fair_}_fairssl_mobilenetv2" ;; *) STEM="XA_${METHOD}_sardetssl_mobilenetv2" ;; esac

newest() { local m; m=$(ls -td $1 2>/dev/null | head -1) || true; [ -n "$m" ] || { echo "ERROR: no run matches $1" >&2; exit 1; }; echo "$m"; }
score() { python Experiments/mobileNetv2_DINO/evaluate.py --run "$1" --split validation; python Experiments/mobileNetv2_DINO/evaluate.py --run "$1" --split test; }

for BUDGET in labeled_set_1pct labeled_set labeled_set_10pct; do
    echo "================ $METHOD split $SAR_SPLIT $BUDGET ================"
    if [ "$BUDGET" = labeled_set ]; then CSV=(); SUFFIX=; else CSV=(--train-csv "$BUDGET.csv"); SUFFIX=_$BUDGET; fi
    RUN=$(ls -td SARDet_to_ATR/outputs/finetune/*_${STEM}${SUFFIX}${TAG}_seed0 2>/dev/null | head -1) || true
    if [ -n "$RUN" ] && [ -f "$RUN/eval_test/metrics.json" ]; then echo "already done: $RUN"; continue; fi
    python SARDet_to_ATR/finetune_atr.py --method "$METHOD" --backbone "$BACKBONE" "${CSV[@]}"
    score "$(newest "SARDet_to_ATR/outputs/finetune/*_${STEM}${SUFFIX}${TAG}_seed0")"
done
crc-job-stats
