#!/bin/bash
# A3 on Pitt CRC: smoke test, SparK SSL (100 epochs, batch 64, full float32), then fine-tune the FINAL backbone
# on the 3%, 1% and 10% labeled sets and score validation and test for each.
# Submit from the repository root:   sbatch crc/spark.sbatch
# If it stops early, resume the SSL part with:
#   python Experiments/mobileNetv2_SparK/train_ssl.py --resume "<SSL run folder>"   (inside a job)
# Test-set rule: decisions are made on validation only; test scores are recorded, never used to choose settings.
# SSL time on an A100 is not measured yet (train_ssl.py prints it after 25 steps), so this uses QoS normal
# (up to 3 days) with a 36 h limit, like the LeJEPA job.
#SBATCH --job-name=A3_spark
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-12:00:00
#SBATCH --qos=normal
#SBATCH --output=crc/logs/%x_%j.out

set -eo pipefail
source crc/env.sh

python Experiments/mobileNetv2_SparK/smoke_test.py          # stops the job here if any check fails

python Experiments/mobileNetv2_SparK/train_ssl.py
SSL=$(ls -td Experiments/mobileNetv2_SparK/outputs/ssl/*_A3_spark_ssl_* | head -1)
echo "SSL run folder: $SSL"
BACKBONE="$SSL/backbone_epoch100.pth"

score() {  # run folder: validation and test
    python Experiments/mobileNetv2_SparK/evaluate.py --run "$1" --split validation
    python Experiments/mobileNetv2_SparK/evaluate.py --run "$1" --split test
}

# 3% (labeled_set.csv, the config's default; same run name as the A0/A1/A2 main runs)
python Experiments/mobileNetv2_SparK/train_supervised.py --backbone "$BACKBONE"
score "$(ls -td Experiments/mobileNetv2_SparK/outputs/finetune/*_A3_spark_mobilenetv2_seed0 | head -1)"

for BUDGET in labeled_set_1pct labeled_set_10pct; do
    python Experiments/mobileNetv2_SparK/train_supervised.py --backbone "$BACKBONE" --train-csv $BUDGET.csv
    score "$(ls -td Experiments/mobileNetv2_SparK/outputs/finetune/*_A3_spark_mobilenetv2_${BUDGET}_seed0 | head -1)"
done

crc-job-stats
