#!/bin/bash
# Fine-tune seed noise: repeat the baseline at 10 %, split 1 with seed 1 (DESIGN.md section 6). The gap between
# this run and the seed-0 run is the number quoted next to the method differences.
#   sbatch crc/finetune_repeat.sbatch [experiment folder] [backbone]   (default baseline_SARDet imagenet)
#SBATCH --job-name=ft_repeat
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --time=0-06:00:00
#SBATCH --qos=short
#SBATCH --output=crc/logs/%x_%j.out

set -eo pipefail
DIR=${1:-baseline_SARDet}
BACKBONE=${2:-imagenet}
export SAR_SPLIT=1
source crc/env.sh
mkdir -p crc/logs
stage_dataset
RUN_NAME=$(python -c "import json,os; f='$DIR/config_finetune.json'; f=f if os.path.exists(f) else '$DIR/config.json'; print(json.load(open(f))['run_name'])")
python -m shared.finetune --experiment-dir $DIR --run-name "$RUN_NAME" --budget 10pct --backbone "$BACKBONE" --seed 1
crc-job-stats
