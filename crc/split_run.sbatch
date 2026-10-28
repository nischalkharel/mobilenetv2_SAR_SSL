#!/bin/bash
# One route on a re-drawn data split (2 or 3): SSL (if the route has one), then fine-tune on the 1%, 3%
# and 10% labeled sets of THAT split and score validation and test for each. Same recipe and settings
# as the split 1 runs; only the split (validation, SSL pool, labeled sets) changes. Training seed stays 0.
# Submit from the repository root, one job per route and split:
#   sbatch crc/split_run.sbatch baseline 2      (routes: baseline, dino, lejepa, spark; splits: 2, 3)
# If an SSL part stops early, resume it inside a job with SAR_SPLIT set:
#   SAR_SPLIT=2 python Experiments/<route folder>/train_ssl.py --resume "<SSL run folder>"
# Test-set rule: decisions are made on validation only; test scores are recorded, never used to choose settings.
#SBATCH --job-name=split_run
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-12:00:00
#SBATCH --qos=normal
#SBATCH --output=crc/logs/%x_%j.out

set -eo pipefail
ROUTE=$1
export SAR_SPLIT=$2
case "$SAR_SPLIT" in 2|3) ;; *) echo "ERROR: split must be 2 or 3, got '$SAR_SPLIT'" >&2; exit 1 ;; esac
if [ ! -f "preprocessing_dataset/split$SAR_SPLIT/labeled_set_10pct.csv" ]; then
    echo "ERROR: preprocessing_dataset/split$SAR_SPLIT/ is missing its CSVs (git pull?)" >&2; exit 1
fi
source crc/env.sh
echo "route $ROUTE  split $SAR_SPLIT"
TAG=_split$SAR_SPLIT

newest() { ls -td $1 | head -1; }  # newest run folder matching a pattern

budget() {  # labeled set name -> CSV option and run-name suffix (3% = the configs' default, no option, as in split 1)
    if [ "$1" = labeled_set ]; then CSV=(); SUFFIX=; else CSV=(--train-csv "$1.csv"); SUFFIX=_$1; fi
}

score() {  # evaluate script, run folder: validation and test
    python "$1" --run "$2" --split validation
    python "$1" --run "$2" --split test
}

case "$ROUTE" in
    baseline)
        for BUDGET in labeled_set_1pct labeled_set labeled_set_10pct; do
            budget $BUDGET
            python Experiments/Baseline/train.py "${CSV[@]}"
            score Experiments/Baseline/evaluate.py "$(newest "Experiments/Baseline/Baseline_Output/*_A0_imagenet_mobilenetv2${SUFFIX}${TAG}_seed0")"
        done
        ;;
    dino|lejepa|spark)
        case "$ROUTE" in
            dino)   DIR=Experiments/mobileNetv2_DINO;   NAME=A1_dino;   BACKBONE_FILE=backbone_teacher_epoch100.pth ;;
            lejepa) DIR=Experiments/mobileNetv2_LeJEPA; NAME=A2_lejepa; BACKBONE_FILE=backbone_epoch100.pth ;;
            spark)  DIR=Experiments/mobileNetv2_SparK;  NAME=A3_spark;  BACKBONE_FILE=backbone_epoch100.pth ;;
        esac
        python $DIR/smoke_test.py            # stops the job here if any check fails
        python $DIR/train_ssl.py
        SSL=$(newest "$DIR/outputs/ssl/*_${NAME}_ssl_mobilenetv2${TAG}_seed0")
        echo "SSL run folder: $SSL"
        for BUDGET in labeled_set_1pct labeled_set labeled_set_10pct; do
            budget $BUDGET
            python $DIR/train_supervised.py --backbone "$SSL/$BACKBONE_FILE" "${CSV[@]}"
            score $DIR/evaluate.py "$(newest "$DIR/outputs/finetune/*_${NAME}_mobilenetv2${SUFFIX}${TAG}_seed0")"
        done
        ;;
    *) echo "ERROR: route must be baseline, dino, lejepa or spark, got '$ROUTE'" >&2; exit 1 ;;
esac

crc-job-stats
