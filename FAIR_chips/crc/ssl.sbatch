#!/bin/bash
# One FAIR-CSAR SSL run (one config, one split), self-resuming across the QoS time limit, like SARDet/crc/ssl.sbatch.
# Submit from the repository root (cancel a whole chain with: scancel -M gpu --name=<job-name>):
#   sbatch --job-name=fdino_t_s1 FAIR_chips/crc/ssl.sbatch config_ssl_dino_tiles.json 1
#   sbatch --job-name=fro_t_s1   FAIR_chips/crc/ssl.sbatch config_ssl_ro_tiles.json 1
#   sbatch --job-name=fror_t_s1  FAIR_chips/crc/ssl.sbatch config_ssl_ror_tiles.json 1
#   (and the *_down.json configs for the resolution arm)
# The engine checkpoints every epoch; at its start this job submits a copy of itself (afterany) that resumes an
# unfinished run or exits at once when the run is complete, so the chain ends by itself. The object-only configs
# need RegionSSL/regions/regions_<pool>.npz (FAIR_chips/crc/regions.sbatch) and stop if it is missing or partial.
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=0-23:30:00
#SBATCH --qos=short
#SBATCH --output=crc/logs/%x_%j.out

set -eo pipefail
CFG=${1:?config file name in FAIR_chips/, e.g. config_ssl_ro_tiles.json}
export SAR_SPLIT=${2:?split 1, 2 or 3}
case "$SAR_SPLIT" in 1|2|3) ;; *) echo "ERROR: split must be 1, 2 or 3" >&2; exit 1 ;; esac
DIR=FAIR_chips
source FAIR_chips/crc/env.sh
SEED=$((SAR_SPLIT - 1))
RUN_NAME=$(python -c "import json; print(json.load(open('$DIR/$CFG'))['run_name'])")

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
        sbatch --dependency=afterany:$SLURM_JOB_ID --job-name="$SLURM_JOB_NAME" FAIR_chips/crc/ssl.sbatch "$CFG" "$SAR_SPLIT"
    else
        echo "a continuation of $SLURM_JOB_NAME is already pending; not submitting another"
    fi
fi
mkdir -p crc/logs
stage_fair

if [ -n "$UNFINISHED" ]; then
    echo "resuming $UNFINISHED"
    python $DIR/train_ssl.py --resume "$UNFINISHED"
else
    python $DIR/train_ssl.py --config $DIR/$CFG
fi
crc-job-stats
