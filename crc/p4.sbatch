#!/bin/bash
# P4 (DINO with per-condition-cell teacher centering) on Pitt CRC: smoke test, cell report, SSL 100 epochs, then
# fine-tune on the 1 %, 3 % and 10 % labeled sets of the split and score validation + test for each. Same recipe,
# settings and seed as the A1 runs (Experiments/mobileNetv2_DINO); only the SSL centering differs.
# Submit from the repository root, one job per split and variant:
#   sbatch crc/p4.sbatch 1             # split 1, P4 (config_ssl.json)
#   sbatch crc/p4.sbatch 2             # split 2;  sbatch crc/p4.sbatch 3  for split 3
#   sbatch --job-name=P4perm_dino crc/p4.sbatch 1 perm     # scrambled-cell control (config_ssl_perm.json); split 1 is enough
#   sbatch --job-name=P4d_dino crc/p4.sbatch 1 3field      # deployable cells, scene | band | depression (config_ssl_3field.json)
# (--job-name on the command line overrides the directive below, so the logs of the variants get their own names.)
# Jobs on the same split share the pixel cache file in outputs/cache/ (built once, about 15 min, from the same image
# list). Each job writes it under a temporary per-process name, so starting them together is safe; copying A1's cache
# first (README.md, "Commands on CRC") skips the build.
# If the SSL part stops early, resume it inside a job with the same SAR_SPLIT:
#   SAR_SPLIT=<n> python Experiments/mobileNetv2_DINO_P4/train_ssl.py --resume "<SSL run folder>"
# Test-set rule: decisions are made on validation only; test scores are recorded, never used to choose settings.
#SBATCH --job-name=P4_dino
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-20:00:00
#SBATCH --qos=short
#SBATCH --output=crc/logs/%x_%j.out

set -eo pipefail
export SAR_SPLIT=${1:-1}
VARIANT=${2:-p4}
case "$SAR_SPLIT" in 1|2|3) ;; *) echo "ERROR: split must be 1, 2 or 3, got '$SAR_SPLIT'" >&2; exit 1 ;; esac
case "$VARIANT" in
    p4)     CFG=config_ssl.json;        FT=config_finetune.json;        NAME=P4_dino ;;
    perm)   CFG=config_ssl_perm.json;   FT=config_finetune_perm.json;   NAME=P4perm_dino ;;
    3field) CFG=config_ssl_3field.json; FT=config_finetune_3field.json; NAME=P4d_dino ;;
    *) echo "ERROR: variant must be p4, perm or 3field, got '$VARIANT'" >&2; exit 1 ;;
esac
if [ "$SAR_SPLIT" != 1 ] && [ ! -f "preprocessing_dataset/split$SAR_SPLIT/labeled_set_10pct.csv" ]; then
    echo "ERROR: preprocessing_dataset/split$SAR_SPLIT/ is missing its CSVs (git pull?)" >&2; exit 1
fi
source crc/env.sh
DIR=Experiments/mobileNetv2_DINO_P4
TAG=""
[ "$SAR_SPLIT" = 1 ] || TAG="_split$SAR_SPLIT"
echo "P4 route: split $SAR_SPLIT  variant $VARIANT  config $DIR/$CFG  fine-tune config $DIR/$FT"

newest() {  # newest run folder matching a pattern; stops the job with a message when nothing matches
    local m
    m=$(ls -td $1 2>/dev/null | head -1) || true
    if [ -z "$m" ]; then echo "ERROR: no run folder matches $1" >&2; exit 1; fi
    echo "$m"
}

budget() {  # labeled set name -> CSV option and run-name suffix (3% = the configs' default, no option, as for A1)
    if [ "$1" = labeled_set ]; then CSV=(); SUFFIX=; else CSV=(--train-csv "$1.csv"); SUFFIX=_$1; fi
}

score() {  # evaluate script, run folder: validation and test
    python "$1" --run "$2" --split validation
    python "$1" --run "$2" --split test
}

python $DIR/smoke_test.py                       # stops the job here if any check fails
python $DIR/cell_report.py --config $DIR/$CFG   # CPU, seconds: the cells of this split (label-leak numbers for the records)
python $DIR/train_ssl.py --config $DIR/$CFG
SSL=$(newest "$DIR/outputs/ssl/*_${NAME}_ssl_mobilenetv2${TAG}_seed0")
echo "SSL run folder: $SSL"
for BUDGET in labeled_set_1pct labeled_set labeled_set_10pct; do
    budget $BUDGET
    python $DIR/train_supervised.py --config $DIR/$FT --backbone "$SSL/backbone_teacher_epoch100.pth" "${CSV[@]}"
    score $DIR/evaluate.py "$(newest "$DIR/outputs/finetune/*_${NAME}_mobilenetv2${SUFFIX}${TAG}_seed0")"
done

crc-job-stats
