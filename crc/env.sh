#!/bin/bash
# Shared setup for every job on Pitt CRC; sourced by the .sbatch scripts in this folder.
# Everything CRC-specific lives here. On the PC nothing here is used (paths.py defaults apply).

module purge
module load anaconda3/2025.7.0-2-python_3.11
source activate /ix/ageorge/nik96/envs/sarssl     # CRC manual: "source activate", never "conda init"

# Dataset on CRC (read by every paths.py); contains Amplitude_8bit/<encoded name>/SOC_40classes
export ATRNET_GROUND_RANGE=/ix/ageorge/nik96/data/ATRNet-STAR/Ground_Range
# ImageNet MobileNetV2 start weights on CRC (sha256 1f19491f...; same file as on the PC)
export MOBILENETV2_IMAGENET_WEIGHTS=/ix/ageorge/nik96/data/mobilenet_v2_imagenet1k_v2.pth

# Record where and with what the job runs (top of the job's log file)
echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  start $(date)"
echo "repo $(pwd)  commit $(git rev-parse --short HEAD)  branch $(git rev-parse --abbrev-ref HEAD)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
python -c "import sys, torch, torchvision; print('python', sys.version.split()[0], '| torch', torch.__version__, '| torchvision', torchvision.__version__, '| cuda available', torch.cuda.is_available())"
