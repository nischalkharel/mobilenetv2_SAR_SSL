# CRC environment for the chip study: the detection suite's environment (conda env, SARDET100K_ROOT, stage_dataset)
# plus where the chips live and a helper that copies the chip array to the node's local disk.
#   source SARDet_chips/crc/env.sh      (from the repository root, inside a job script)
_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
pushd "$_REPO/SARDet" >/dev/null && source crc/env.sh && popd >/dev/null      # SARDet/crc/env.sh expects the suite root
export SARDET_CHIPS_ROOT=${SARDET_CHIPS_ROOT:-/ix/ageorge/nik96/data/SARDet_chips}

# The suite's stage_dataset runs `python shared/paths.py` relative to the suite root; call it from there.
stage_dataset_from_repo_root() { pushd "$_REPO/SARDet" >/dev/null && stage_dataset && popd >/dev/null; }

# Copy chips_u8.npy (4 GB) + chips.csv + chips_meta.json to $SLURM_SCRATCH once per job and point SARDET_CHIPS_ROOT
# there. Two sequential file reads; nothing else in a fine-tune job touches /ix per step. STAGE_CHIPS=0 disables it.
stage_chips() {
    if [ -z "$SLURM_SCRATCH" ] || [ "${STAGE_CHIPS:-1}" != 1 ]; then return 0; fi
    local src="$SARDET_CHIPS_ROOT" dst="$SLURM_SCRATCH/SARDet_chips" t0
    if [ ! -f "$dst/.staged" ]; then
        t0=$(date +%s)
        mkdir -p "$dst"
        cp "$src/chips_u8.npy" "$src/chips.csv" "$src/chips_meta.json" "$dst/" || { echo "ERROR: chip staging failed" >&2; exit 1; }
        touch "$dst/.staged"
        echo "chips staged in $(( $(date +%s) - t0 )) s: $(du -sh "$dst" | cut -f1)"
    fi
    export SARDET_CHIPS_ROOT="$dst"
    echo "SARDET_CHIPS_ROOT=$SARDET_CHIPS_ROOT"
}
