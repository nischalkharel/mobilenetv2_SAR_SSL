#!/bin/bash
# One-time learning-rate check for the SHARED fine-tune lr (DESIGN.md section 3): 1e-4 / 2e-4 / 5e-4 on
# val_select, 10 % labels, split 1, for the ImageNet baseline and (optionally) one SSL backbone.
#   sbatch crc/lr_check.sbatch [path to an exported SSL backbone]
# Runs are marked protocol_deviation (lr override) so compare.py ignores them; read best_val_select_map in
# each run_info.json, pick the lr, write it into shared/finetune_config.json ONCE, commit, and only then
# launch the real fine-tunes. Decision on val_select only (never on test).
#SBATCH --job-name=lr_check
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --time=0-20:00:00
#SBATCH --qos=short
#SBATCH --output=crc/logs/%x_%j.out

set -eo pipefail
export SAR_SPLIT=1
source crc/env.sh
mkdir -p crc/logs
stage_dataset
SSL_BACKBONE=${1:-}
for LR in 0.0001 0.0002 0.0005; do
    python -m shared.finetune --experiment-dir baseline_SARDet --run-name lrcheck_baseline_lr$LR --budget 10pct \
        --backbone imagenet --lr $LR --no-final-eval
    if [ -n "$SSL_BACKBONE" ]; then
        python -m shared.finetune --experiment-dir DINO_SARDet --run-name lrcheck_ssl_lr$LR --budget 10pct \
            --backbone "$SSL_BACKBONE" --lr $LR --no-final-eval
    fi
done
echo "best val_select mAP per run:"
for f in baseline_SARDet/outputs/finetune/*lrcheck*/run_info.json DINO_SARDet/outputs/finetune/*lrcheck*/run_info.json; do
    [ -f "$f" ] && python -c "import json; r=json.load(open('$f')); print(r['config']['run_name'], r.get('best_val_select_map'))"
done
crc-job-stats
