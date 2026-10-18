#!/bin/bash
# A0 baseline on Pitt CRC: fine-tune ImageNet MobileNetV2 on labeled_set.csv, then score validation.
# Submit from the repository root:   sbatch crc/baseline.sbatch
#SBATCH --job-name=A0_baseline
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-02:00:00
#SBATCH --qos=short
#SBATCH --output=crc/logs/%x_%j.out

set -eo pipefail
source crc/env.sh

python Experiments/Baseline/train.py
RUN=$(ls -td Experiments/Baseline/Baseline_Output/*_A0_* | head -1)
echo "run folder: $RUN"
python Experiments/Baseline/evaluate.py --run "$RUN" --split validation

crc-job-stats
