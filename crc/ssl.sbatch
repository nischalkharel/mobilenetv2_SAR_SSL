#!/bin/bash
# One SSL run (DINO / LeJEPA / SparK / CondDINO) on one split, self-resuming across the QoS time limit.
# Submit from the suite root (cancel a whole chain with: scancel -M gpu --name=<job-name>):
#   sbatch --job-name=dino_s1 crc/ssl.sbatch DINO_SARDet 1
#   sbatch --job-name=lejepa_s1 crc/ssl.sbatch LeJEPA_SARDet 1
#   sbatch --job-name=spark_s1 crc/ssl.sbatch SparK_SARDet 1
#   sbatch --job-name=cond_s1 crc/ssl.sbatch CondDINO_SARDet 1 [config_ssl.json | config_ssl_perm.json | ...]
# How the time limit is survived: the engine checkpoints every epoch. At its start this job submits a copy of
# itself that runs after it ends (afterany). If this job is killed at the limit, the copy finds the unfinished
# run folder and resumes it; if the run completed, the copy exits at once. So the chain ends by itself.
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=0-23:30:00
#SBATCH --qos=short
#SBATCH --output=crc/logs/%x_%j.out

set -eo pipefail
DIR=${1:?experiment folder, e.g. DINO_SARDet}
export SAR_SPLIT=${2:?split 1, 2 or 3}
CFG=${3:-config_ssl.json}
case "$SAR_SPLIT" in 1|2|3) ;; *) echo "ERROR: split must be 1, 2 or 3" >&2; exit 1 ;; esac
source crc/env.sh
SEED=$((SAR_SPLIT - 1))
RUN_NAME=$(python -c "import json; print(json.load(open('$DIR/$CFG'))['run_name'])")

# an unfinished run of this method / config / split? (status running = killed at the limit, or stopped)
UNFINISHED=""; NEWEST_STATUS=""
for d in $(ls -td $DIR/outputs/ssl/*_${RUN_NAME}_split${SAR_SPLIT}_seed${SEED} 2>/dev/null); do
    status=$(python -c "import json; print(json.load(open('$d/run_info.json'))['status'])")
    [ -z "$NEWEST_STATUS" ] && NEWEST_STATUS=$status
    if [ "$status" != "completed" ] && [ "$status" != "failed" ]; then UNFINISHED=$d; break; fi
    if [ "$status" = "completed" ] && [ -z "$FORCE_NEW" ]; then
        echo "run $d is complete; nothing to do (set FORCE_NEW=1 to start another)"; exit 0
    fi
done
# a FAILED newest run (NaN loss, OOM, export error) must not be restarted from scratch by the chain:
if [ -z "$UNFINISHED" ] && [ "$NEWEST_STATUS" = failed ] && [ -z "$FORCE_NEW" ]; then
    echo "ERROR: the newest run of $RUN_NAME split $SAR_SPLIT FAILED (see its run_info.json); fix the cause, then resubmit with FORCE_NEW=1" >&2
    exit 1
fi

# submit the continuation only now that this job will train (it exits at once if the run completes), and only
# if no continuation of this chain is already pending (a second manual submit must not create a second chain)
if [ -n "$SLURM_JOB_ID" ] && [ -z "$NO_CHAIN" ]; then
    if [ -z "$(squeue -M gpu -h --name="$SLURM_JOB_NAME" -t PD 2>/dev/null | grep -v '^CLUSTER')" ]; then
        sbatch --dependency=afterany:$SLURM_JOB_ID --job-name="$SLURM_JOB_NAME" crc/ssl.sbatch "$DIR" "$SAR_SPLIT" "$CFG"
    else
        echo "a continuation of $SLURM_JOB_NAME is already pending; not submitting another"
    fi
fi
mkdir -p crc/logs

if [ -n "$UNFINISHED" ]; then
    echo "resuming $UNFINISHED"
    python $DIR/train_ssl.py --resume "$UNFINISHED"
else
    python $DIR/train_ssl.py --config $DIR/$CFG
fi
crc-job-stats
