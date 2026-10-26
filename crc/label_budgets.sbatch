#!/bin/bash
# Label budgets on split 1 (seed 0): fine-tune A0, A1, A2 on the 1% and 10% labeled sets, reusing the
# existing 100-epoch SSL backbones (no new SSL). Scores validation and the official test set for every
# run, and adds test scores to the three existing 3% runs.
# Submit from the repository root:   sbatch crc/label_budgets.sbatch
# Rule: every decision is made on validation only; test scores are recorded, never used to choose settings.
#SBATCH --job-name=label_budgets
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-06:00:00
#SBATCH --qos=short
#SBATCH --output=crc/logs/%x_%j.out

set -eo pipefail
source crc/env.sh

DINO_SSL=Experiments/mobileNetv2_DINO/outputs/ssl/20261018-192425_A1_dino_ssl_mobilenetv2_seed0
LEJEPA_SSL=Experiments/mobileNetv2_LeJEPA/outputs/ssl/20261018-200757_A2_lejepa_ssl_mobilenetv2_seed0
DINO_BACKBONE=$DINO_SSL/backbone_teacher_epoch100.pth
LEJEPA_BACKBONE=$LEJEPA_SSL/backbone_epoch100.pth

# The SSL backbones must be exactly the files the 3% fine-tunes used (SHA-256 from their run_info.json).
check_hash() {  # file, expected sha256
    actual=$(sha256sum "$1" | cut -d' ' -f1)
    if [ "$actual" != "$2" ]; then echo "ERROR: $1 sha256 $actual, expected $2" >&2; exit 1; fi
    echo "backbone unchanged: $1"
}
check_hash "$DINO_BACKBONE"   50214da5d552186e090ed7c485ef28c863aedd1976f2838c31568c3a84c55c2c
check_hash "$LEJEPA_BACKBONE" 1412edff352b7189b79277d9b60ad88fe98a360d6e8a227e2822e242f8f6a369

newest() { ls -td $1 | head -1; }  # newest run folder matching a pattern

score() {  # evaluate script, run folder: validation and test
    python "$1" --run "$2" --split validation
    python "$1" --run "$2" --split test
}

for BUDGET in labeled_set_1pct labeled_set_10pct; do
    echo "================ $BUDGET ================"
    python Experiments/Baseline/train.py --train-csv $BUDGET.csv
    score Experiments/Baseline/evaluate.py "$(newest "Experiments/Baseline/Baseline_Output/*_A0_imagenet_mobilenetv2_${BUDGET}_seed0")"

    python Experiments/mobileNetv2_DINO/train_supervised.py --backbone "$DINO_BACKBONE" --train-csv $BUDGET.csv
    score Experiments/mobileNetv2_DINO/evaluate.py "$(newest "Experiments/mobileNetv2_DINO/outputs/finetune/*_A1_dino_mobilenetv2_${BUDGET}_seed0")"

    python Experiments/mobileNetv2_LeJEPA/train_supervised.py --backbone "$LEJEPA_BACKBONE" --train-csv $BUDGET.csv
    score Experiments/mobileNetv2_LeJEPA/evaluate.py "$(newest "Experiments/mobileNetv2_LeJEPA/outputs/finetune/*_A2_lejepa_mobilenetv2_${BUDGET}_seed0")"
done

echo "================ test scores for the existing 3% runs ================"
python Experiments/Baseline/evaluate.py      --run Experiments/Baseline/Baseline_Output/20261018-190525_A0_imagenet_mobilenetv2_seed0 --split test
python Experiments/mobileNetv2_DINO/evaluate.py   --run Experiments/mobileNetv2_DINO/outputs/finetune/20261019-032810_A1_dino_mobilenetv2_seed0 --split test
python Experiments/mobileNetv2_LeJEPA/evaluate.py --run Experiments/mobileNetv2_LeJEPA/outputs/finetune/20261019-065634_A2_lejepa_mobilenetv2_seed0 --split test

crc-job-stats
