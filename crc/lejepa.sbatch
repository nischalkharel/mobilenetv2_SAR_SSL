#!/bin/bash
# A2 on Pitt CRC: smoke test, LeJEPA SSL (100 epochs, batch 64), fine-tune from the final backbone, score validation.
# Submit from the repository root:   sbatch crc/lejepa.sbatch
# If it stops early, resume the SSL part with:
#   python Experiments/mobileNetv2_LeJEPA/train_ssl.py --resume "<SSL run folder>"   (inside a job)
# Expected SSL time on an A100 is about 14 h, so this uses QoS normal (up to 3 days) with a 36 h limit.
#SBATCH --job-name=A2_lejepa
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

python Experiments/mobileNetv2_LeJEPA/smoke_test.py          # stops the job here if any check fails

python Experiments/mobileNetv2_LeJEPA/train_ssl.py
SSL=$(ls -td Experiments/mobileNetv2_LeJEPA/outputs/ssl/*_A2_lejepa_ssl_* | head -1)
echo "SSL run folder: $SSL"

python Experiments/mobileNetv2_LeJEPA/train_supervised.py --backbone "$SSL/backbone_epoch100.pth"
FT=$(ls -td Experiments/mobileNetv2_LeJEPA/outputs/finetune/*_A2_lejepa_* | head -1)
echo "fine-tuning run folder: $FT"
python Experiments/mobileNetv2_LeJEPA/evaluate.py --run "$FT" --split validation

crc-job-stats
