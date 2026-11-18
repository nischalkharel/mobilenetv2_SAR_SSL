#!/bin/bash
# One SSL run of the RegionSSL engine on the ATRNet-STAR pool (arm I), one split, self-resuming across the time limit.
# Same chain logic as the suite's crc/ssl.sbatch; differences: both env files are sourced (SARDet paths for the suite
# modules, ATRNet paths for the pool), nothing is staged to the node disk (the chips are small and read from /ix).
# Submit from the suite root, like every other job (the region file regions_atrnet.npz must exist first):
#   cd /ihome/ageorge/nik96/mobilenetv2_SAR_SSL/SARDet
#   sbatch --job-name=adino_s1 ../RegionSSL/crc/ssl_atr.sbatch ../RegionSSL/DINO_reference 1 config_ssl_atr.json
#   sbatch --job-name=aro_s1   ../RegionSSL/crc/ssl_atr.sbatch ../RegionSSL/RegionDINO_object 1 config_ssl_atr.json
#   sbatch --job-name=aror_s1  ../RegionSSL/crc/ssl_atr.sbatch ../RegionSSL/RegionDINO_object 1 config_ssl_atr_random.json
#   sbatch --job-name=arso_s1  ../RegionSSL/crc/ssl_atr.sbatch ../RegionSSL/RegionDINO_scene_object 1 config_ssl_atr.json
#   sbatch --job-name=arsor_s1 ../RegionSSL/crc/ssl_atr.sbatch ../RegionSSL/RegionDINO_scene_object 1 config_ssl_atr_random.json
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=0-23:30:00
#SBATCH --qos=short
#SBATCH --output=crc/logs/%x_%j.out

set -eo pipefail
DIR=${1:?experiment folder, e.g. ../RegionSSL/RegionDINO_object}
export SAR_SPLIT=${2:?split 1, 2 or 3}
CFG=${3:?config file, e.g. config_ssl_atr.json}
case "$SAR_SPLIT" in 1|2|3) ;; *) echo "ERROR: split must be 1, 2 or 3" >&2; exit 1 ;; esac
export STAGE_DATA=0
source crc/env.sh              # suite env: python, SARDET100K_ROOT, SAR_SPLIT handling
source ../crc/env.sh           # repository env: ATRNET_GROUND_RANGE (the pool), MOBILENETV2_IMAGENET_WEIGHTS
SEED=$((SAR_SPLIT - 1))
RUN_NAME=$(python -c "import json; print(json.load(open('$DIR/$CFG'))['run_name'])")
SELF=../RegionSSL/crc/ssl_atr.sbatch

UNFINISHED=""; NEWEST_STATUS=""
for d in $(ls -td $DIR/outputs/ssl/*_${RUN_NAME}_split${SAR_SPLIT}_seed${SEED} 2>/dev/null); do
    status=$(python -c "import json; print(json.load(open('$d/run_info.json'))['status'])")
    [ -z "$NEWEST_STATUS" ] && NEWEST_STATUS=$status
    if [ "$status" != "completed" ] && [ "$status" != "failed" ]; then UNFINISHED=$d; break; fi
    if [ "$status" = "completed" ] && [ -z "$FORCE_NEW" ]; then
        echo "run $d is complete; nothing to do (set FORCE_NEW=1 to start another)"; exit 0
    fi
done
if [ -z "$UNFINISHED" ] && [ "$NEWEST_STATUS" = failed ] && [ -z "$FORCE_NEW" ]; then
    echo "ERROR: the newest run of $RUN_NAME split $SAR_SPLIT FAILED (see its run_info.json); fix the cause, then resubmit with FORCE_NEW=1" >&2
    exit 1
fi
if [ -n "$SLURM_JOB_ID" ] && [ -z "$NO_CHAIN" ]; then
    if [ -z "$(squeue -M gpu -h --name="$SLURM_JOB_NAME" -t PD 2>/dev/null | grep -v '^CLUSTER')" ]; then
        sbatch --dependency=afterany:$SLURM_JOB_ID --job-name="$SLURM_JOB_NAME" $SELF "$DIR" "$SAR_SPLIT" "$CFG"
    else
        echo "a continuation of $SLURM_JOB_NAME is already pending; not submitting another"
    fi
fi
mkdir -p crc/logs
if [ -n "$UNFINISHED" ]; then
    echo "resuming $UNFINISHED"
    python $DIR/train_ssl.py --resume "$UNFINISHED"
else
    python $DIR/train_ssl.py --config "$DIR/$CFG"
fi
crc-job-stats
