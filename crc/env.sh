#!/bin/bash
# Shared setup for every job on Pitt CRC; sourced by the .sbatch scripts in this folder.
# Everything CRC-specific lives here. On the PC nothing here is used (paths.py defaults apply).

# Python environment (conda-forge Python 3.13 + pip packages). It is put first on PATH directly instead of
# "source activate": Slurm copies the submitting shell's settings into the job, and when the environment is
# already active there, conda's activate skips its work and the base Anaconda Python (no torch) wins.
ENV_DIR=/ix/ageorge/nik96/envs/sarssl
export PATH="$ENV_DIR/bin:$PATH"
export CONDA_PREFIX="$ENV_DIR"
if [ "$(command -v python)" != "$ENV_DIR/bin/python" ]; then
    echo "ERROR: python is $(command -v python), expected $ENV_DIR/bin/python" >&2
    exit 1
fi

# Dataset on CRC (read by every paths.py); contains Amplitude_8bit/<encoded name>/SOC_40classes
export ATRNET_GROUND_RANGE=/ix/ageorge/nik96/data/ATRNet-STAR/Ground_Range
# ImageNet MobileNetV2 start weights on CRC (sha256 1f19491f...; same file as on the PC)
export MOBILENETV2_IMAGENET_WEIGHTS=/ix/ageorge/nik96/data/mobilenet_v2_imagenet1k_v2.pth

# Record where and with what the job runs (top of the job's log file)
echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  start $(date)"
echo "repo $(pwd)  commit $(git rev-parse --short HEAD)  branch $(git rev-parse --abbrev-ref HEAD)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
python -c "import sys, torch, torchvision; print('python', sys.executable, sys.version.split()[0], '| torch', torch.__version__, '| torchvision', torchvision.__version__, '| cuda available', torch.cuda.is_available())"
