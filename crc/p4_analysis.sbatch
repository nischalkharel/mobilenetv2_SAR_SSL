#!/bin/bash
# Analyses of the finished P4 / A1 backbones on Pitt CRC (the checkpoints live here): frozen-feature probes (what the
# features encode: class versus scene / azimuth sector / depression / band) and the Grad-CAM + response-map figure with
# the share of map mass on the vehicle box, A1 versus P4. GPU, about 15 minutes. Submit from the repository root:
#   sbatch crc/p4_analysis.sbatch
# Outputs (small; commit them): Experiments/mobileNetv2_DINO_P4/outputs/compare/feature_probe.{md,csv},
# saliency_summary.md, saliency_per_chip.csv, saliency_figure.{png,pdf}.
#SBATCH --job-name=P4_analysis
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-02:00:00
#SBATCH --qos=short
#SBATCH --output=crc/logs/%x_%j.out

set -eo pipefail
export SAR_SPLIT=1
source crc/env.sh
DIR=Experiments/mobileNetv2_DINO_P4
python $DIR/feature_probe.py
python $DIR/saliency_figure.py
crc-job-stats
