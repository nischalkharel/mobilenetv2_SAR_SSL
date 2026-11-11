#!/bin/bash
# Fine-tune + score one method on the 1 %, 3 % and 10 % labeled sets of one split, then probes and panels.
# Submit from the suite root:
#   sbatch --job-name=base_ft_s1 crc/finetune.sbatch baseline_SARDet 1 imagenet
#   sbatch --job-name=dino_ft_s1 crc/finetune.sbatch DINO_SARDet 1 auto          # auto = newest completed SSL run of that split
#   sbatch --job-name=cond_ft_s1 crc/finetune.sbatch CondDINO_SARDet 1 auto condcenter_source_pol   # 4th arg = SSL run_name to match
#   sbatch --job-name=dino_ft_s1 crc/finetune.sbatch DINO_SARDet 1 DINO_SARDet/outputs/ssl/<run>/backbone_epoch100.pth
# Chain after an SSL job:  sbatch --dependency=afterok:<ssl job id> ... (the ssl chain's LAST job id)
# Budgets run largest first so the long run starts early; each budget = train + val_select selection + full
# val / test / test_clean scoring of best.pt and test scoring of last.pt (shared/finetune.py).
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --time=0-12:00:00
#SBATCH --qos=short
#SBATCH --output=crc/logs/%x_%j.out

set -eo pipefail
DIR=${1:?experiment folder}
export SAR_SPLIT=${2:?split}
BACKBONE=${3:?imagenet | auto | path to backbone_epochN.pth}
MATCH=${4:-}
source crc/env.sh
SEED=$((SAR_SPLIT - 1))

if [ "$BACKBONE" = auto ]; then
    for d in $(ls -td $DIR/outputs/ssl/*${MATCH}*_split${SAR_SPLIT}_seed${SEED} 2>/dev/null); do
        status=$(python -c "import json; print(json.load(open('$d/run_info.json'))['status'])")
        if [ "$status" = "completed" ]; then BACKBONE=$(ls $d/backbone_epoch*.pth | head -1); break; fi
    done
    [ "$BACKBONE" = auto ] && { echo "ERROR: no completed SSL run for $DIR split $SAR_SPLIT ($MATCH)" >&2; exit 1; }
fi
echo "backbone: $BACKBONE"
PROBE_SPEC=$BACKBONE
RUN_NAME=$(python -c "import json,os; f='$DIR/config_finetune.json'; f=f if os.path.exists(f) else '$DIR/config.json'; print(json.load(open(f))['run_name'])")
[ -n "$MATCH" ] && RUN_NAME="${MATCH}_mobilenetv2"     # CondDINO variants get their own fine-tune run names
mkdir -p crc/logs

for BUDGET in 10pct 3pct 1pct; do
    DONE=""
    for d in $(ls -td $DIR/outputs/finetune/*_${RUN_NAME}_${BUDGET}_split${SAR_SPLIT}_seed0 2>/dev/null); do
        if [ "$(python -c "import json; r=json.load(open('$d/run_info.json')); print(r['status'], r.get('protocol_deviation'))")" = "completed None" ]; then DONE=$d; break; fi
    done
    if [ -n "$DONE" ]; then echo "budget $BUDGET already completed in $DONE; skipping"; continue; fi
    python -m shared.finetune --experiment-dir $DIR --run-name "$RUN_NAME" --budget $BUDGET --backbone "$BACKBONE"
    RUN=$(ls -td $DIR/outputs/finetune/*_${RUN_NAME}_${BUDGET}_split${SAR_SPLIT}_seed0 | head -1)
    echo "run folder: $RUN"
    python -m shared.visualize --run "$RUN" --split test --per-source 2
    python -m shared.probes --backbone "finetuned:$RUN/best.pt" --out "$RUN/probes_finetuned"
done
# frozen-feature probes of the starting backbone (once per split)
python -m shared.probes --backbone "$PROBE_SPEC" --out "$DIR/outputs/probes/${RUN_NAME}_split${SAR_SPLIT}"
crc-job-stats
