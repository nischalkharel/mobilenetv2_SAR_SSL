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

# Stage the SOC-40 chips (train + test, .tif + .xml, about 97k small files) on the node-local scratch disk.
# Measured 2026-11-18: with 15 arm-I SSL chains plus 6 ATRNet fine-tunes reading the chips from the shared /ix at
# once, jobs sharing a node ran 3-10x slower than jobs alone on a node (0.45 vs 4.7 s per SSL step; a fine-tune test
# batch 2 vs 30 s), because every job on a node shares that node's one link to /ix. Same remedy as the SARDet suite
# (SARDet/crc/env.sh stage_dataset): one sequential read of a tar file per job, wiped when the job ends.
# Source: /ix/.../ATRNet-STAR/Ground_Range_SOC40.tar when it exists (make it once, on a login node:
#   cd /ix/ageorge/nik96/data/ATRNet-STAR && tar -cf Ground_Range_SOC40.tar "Ground_Range/Amplitude_8bit/<encoded name>/SOC_40classes"
# ), otherwise a tar pipe over the files (slower). Set STAGE_ATRNET=0 to read from /ix directly.
# Not run automatically: the job scripts call `stage_atrnet` right before training, after they know there is
# something to train. Only ATRNET_GROUND_RANGE changes; the CSVs (repo) and the weights are untouched.
stage_atrnet() {
    if [ -z "$SLURM_SCRATCH" ] || [ "${STAGE_ATRNET:-1}" != 1 ]; then return 0; fi
    local repo soc rel src="$ATRNET_GROUND_RANGE" dst="$SLURM_SCRATCH/ATRNet-STAR/Ground_Range" t0
    repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
    soc=$(python -c "import sys; sys.path.insert(0, '$repo/Experiments/mobileNetv2_DINO'); import paths; print(paths.SOC40_DIR.as_posix())")
    rel=${soc#"$src"/}                      # Amplitude_8bit/<encoded name>/SOC_40classes
    if [ ! -f "$dst/.staged" ]; then
        t0=$(date +%s)
        if [ -f "$src/../Ground_Range_SOC40.tar" ]; then
            echo "staging ATRNet SOC-40 from $src/../Ground_Range_SOC40.tar -> $dst ..."
            mkdir -p "$dst" && tar -C "$(dirname "$dst")" -xf "$src/../Ground_Range_SOC40.tar" || { echo "ERROR: staging failed" >&2; exit 1; }
        else
            echo "staging ATRNet SOC-40 $soc -> $dst/$rel (file by file; create Ground_Range_SOC40.tar to make this fast) ..."
            mkdir -p "$dst/$(dirname "$rel")" && tar -C "$(dirname "$soc")" -cf - SOC_40classes | tar -C "$dst/$(dirname "$rel")" -xf - || { echo "ERROR: staging failed" >&2; exit 1; }
        fi
        touch "$dst/.staged"
        echo "staged in $(( $(date +%s) - t0 )) s: $(find "$dst/$rel" -name '*.tif' | wc -l) chips"
    fi
    export ATRNET_GROUND_RANGE="$dst"
    echo "ATRNET_GROUND_RANGE=$ATRNET_GROUND_RANGE"
}

# Record where and with what the job runs (top of the job's log file)
echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  start $(date)"
echo "repo $(pwd)  commit $(git rev-parse --short HEAD)  branch $(git rev-parse --abbrev-ref HEAD)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
python -c "import sys, torch, torchvision; print('python', sys.executable, sys.version.split()[0], '| torch', torch.__version__, '| torchvision', torchvision.__version__, '| cuda available', torch.cuda.is_available())"
