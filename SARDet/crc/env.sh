#!/bin/bash
# Shared setup for every job of the SARDet suite on Pitt CRC; sourced by the .sbatch scripts in this folder.
# Everything CRC-specific lives here. On the PC nothing here is used (shared/paths.py defaults apply).
# CONFIRM the three paths below with `ls` on CRC after the transfer, before the first job (memory: setup before edits).

# Python environment: the same conda-forge env as mobilenetv2_SAR_SSL plus pycocotools
#   /ix/ageorge/nik96/envs/sarssl/bin/pip install pycocotools   (once)
ENV_DIR=/ix/ageorge/nik96/envs/sarssl
export PATH="$ENV_DIR/bin:$PATH"
export CONDA_PREFIX="$ENV_DIR"
if [ "$(command -v python)" != "$ENV_DIR/bin/python" ]; then
    echo "ERROR: python is $(command -v python), expected $ENV_DIR/bin/python" >&2
    exit 1
fi

# Dataset on CRC (read by shared/paths.py): folder containing JPEGImages/, Annotations/, test.json, mapping.json.
# Large data belongs in /ix/<group> or /vast/<group>, never a shared drive (CRC notice of 2026-11-02).
export SARDET100K_ROOT=/ix/ageorge/nik96/data/SARDet_100K
# ImageNet MobileNetV2 start weights (same file as the ATRNet-STAR runs; sha256 1f19491f...)
export MOBILENETV2_IMAGENET_WEIGHTS=/ix/ageorge/nik96/data/mobilenet_v2_imagenet1k_v2.pth
# Stage the dataset on the node-local scratch disk. Measured 2026-11-12: with six jobs reading the 116,598 small
# files from the shared /ix at once, epochs took 9-21 min instead of the 4-5 min the GPU needs (I/O bound), and
# identical jobs on different nodes differed 3x. The copy (7.2 GB) takes a few minutes per job and is wiped when
# the job ends. Set STAGE_DATA=0 to read from /ix directly (login-node checks, or a scratch disk that is too small).
# Not run automatically: the job scripts call `stage_dataset` right before training, AFTER they have decided
# that there is something to train (a continuation that finds a completed run must not spend 40 min copying).
# Source: the single tar file next to the dataset folder when it exists (one 7.2 GB sequential read; make it once
# with `tar -cf SARDet_100K.tar SARDet_100K` in the data folder), otherwise a tar pipe over the 116k files (slow).
stage_dataset() {
    if [ -z "$SLURM_SCRATCH" ] || [ "${STAGE_DATA:-1}" != 1 ]; then return 0; fi
    local src="$SARDET100K_ROOT" dst="$SLURM_SCRATCH/SARDet_100K" t0
    if [ ! -f "$dst/.staged" ]; then
        t0=$(date +%s)
        if [ -f "$src.tar" ]; then
            echo "staging dataset from $src.tar -> $dst ..."
            mkdir -p "$SLURM_SCRATCH" && tar -C "$SLURM_SCRATCH" -xf "$src.tar" || { echo "ERROR: staging failed" >&2; exit 1; }
        else
            echo "staging dataset $src -> $dst (file by file; create $src.tar to make this fast) ..."
            mkdir -p "$dst" && tar -C "$src" -cf - . | tar -C "$dst" -xf - || { echo "ERROR: staging failed" >&2; exit 1; }
        fi
        touch "$dst/.staged"
        echo "staged in $(( $(date +%s) - t0 )) s: $(find "$dst/JPEGImages" -type f | wc -l) images"
    fi
    export SARDET100K_ROOT="$dst"
    python shared/paths.py
}
# Which data split (1, 2, 3); the job scripts export it from their first argument
export SAR_SPLIT=${SAR_SPLIT:-1}
# Fewer, fatter data-loader threads
export OMP_NUM_THREADS=1

echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  start $(date)"
echo "repo $(pwd)  commit $(git rev-parse --short HEAD)  branch $(git rev-parse --abbrev-ref HEAD)  split $SAR_SPLIT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
python -c "import sys, torch, torchvision, pycocotools; print('python', sys.executable, sys.version.split()[0], '| torch', torch.__version__, '| torchvision', torchvision.__version__, '| cuda', torch.cuda.is_available())"
python shared/paths.py
