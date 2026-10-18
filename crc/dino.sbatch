#!/bin/bash
# A1 on Pitt CRC: smoke test, DINO SSL (100 epochs), fine-tune from the teacher backbone, score validation.
# Submit from the repository root:   sbatch crc/dino.sbatch
# If it stops early, resume the SSL part with:
#   python Experiments/mobileNetv2_DINO/train_ssl.py --resume "<SSL run folder>"   (inside a job)
#SBATCH --job-name=A1_dino
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --qos=short
#SBATCH --output=crc/logs/%x_%j.out

set -eo pipefail
source crc/env.sh

python Experiments/mobileNetv2_DINO/smoke_test.py            # stops the job here if any check fails

python Experiments/mobileNetv2_DINO/train_ssl.py
SSL=$(ls -td Experiments/mobileNetv2_DINO/outputs/ssl/*_A1_dino_ssl_* | head -1)
echo "SSL run folder: $SSL"

python Experiments/mobileNetv2_DINO/train_supervised.py --backbone "$SSL/backbone_teacher_epoch100.pth"
FT=$(ls -td Experiments/mobileNetv2_DINO/outputs/finetune/*_A1_dino_* | head -1)
echo "fine-tuning run folder: $FT"
python Experiments/mobileNetv2_DINO/evaluate.py --run "$FT" --split validation

crc-job-stats
