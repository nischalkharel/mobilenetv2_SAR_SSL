# CRC environment for the FAIR-CSAR chip study: the detection suite's environment (conda env, ImageNet weights,
# SAR_SPLIT) plus where FAIR-CSAR and its chips live, and helpers that copy them to the node's local disk.
#   source FAIR_chips/crc/env.sh      (from the repository root, inside a job script)
_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
pushd "$_REPO/SARDet" >/dev/null && source crc/env.sh && popd >/dev/null      # SARDet/crc/env.sh expects the suite root
export FAIR_CSAR_ROOT=${FAIR_CSAR_ROOT:-/ix/ageorge/nik96/data/FAIR_CSAR/extracted}
export FAIR_CHIPS_ROOT=${FAIR_CHIPS_ROOT:-/ix/ageorge/nik96/data/FAIR_chips}
# region files (regions_fair_tiles.npz, regions_fair_down.npz) go where the SARDet ones are: RegionSSL/regions/ in
# the clone (the REGIONSSL_REGIONS_DIR default), not committed; the *_summary.json next to them is committed.

# Copy the extracted dataset (12 GB: PNGs + the small folders, no .mat) to $SLURM_SCRATCH once per job and point
# FAIR_CSAR_ROOT there. Source: the tar next to the extracted folder when it exists (one sequential read; make it
# once with `cd /ix/ageorge/nik96/data/FAIR_CSAR && tar -cf extracted.tar extracted`), else a tar pipe over the files.
# STAGE_DATA=0 disables it (reads from /ix directly).
stage_fair() {
    if [ -z "$SLURM_SCRATCH" ] || [ "${STAGE_DATA:-1}" != 1 ]; then return 0; fi
    local src="$FAIR_CSAR_ROOT" dst="$SLURM_SCRATCH/FAIR_CSAR/extracted" t0
    if [ ! -f "$dst/.staged" ]; then
        t0=$(date +%s)
        mkdir -p "$SLURM_SCRATCH/FAIR_CSAR"
        if [ -f "$src.tar" ]; then
            tar -C "$SLURM_SCRATCH/FAIR_CSAR" -xf "$src.tar" || { echo "ERROR: staging failed" >&2; exit 1; }
        else
            mkdir -p "$dst" && tar -C "$src" -cf - . | tar -C "$dst" -xf - || { echo "ERROR: staging failed" >&2; exit 1; }
        fi
        touch "$dst/.staged"
        echo "FAIR-CSAR staged in $(( $(date +%s) - t0 )) s: $(find "$dst" -name '*.png' | wc -l) PNGs"
    fi
    export FAIR_CSAR_ROOT="$dst"
    echo "FAIR_CSAR_ROOT=$FAIR_CSAR_ROOT"
}

# Copy chips_u8.npy (1.4 GB) + chips.csv + chips_meta.json to $SLURM_SCRATCH and point FAIR_CHIPS_ROOT there.
stage_fair_chips() {
    if [ -z "$SLURM_SCRATCH" ] || [ "${STAGE_CHIPS:-1}" != 1 ]; then return 0; fi
    local src="$FAIR_CHIPS_ROOT" dst="$SLURM_SCRATCH/FAIR_chips" t0
    if [ ! -f "$dst/.staged" ]; then
        t0=$(date +%s)
        mkdir -p "$dst"
        cp "$src/chips_u8.npy" "$src/chips.csv" "$src/chips_meta.json" "$dst/" || { echo "ERROR: chip staging failed" >&2; exit 1; }
        touch "$dst/.staged"
        echo "chips staged in $(( $(date +%s) - t0 )) s: $(du -sh "$dst" | cut -f1)"
    fi
    export FAIR_CHIPS_ROOT="$dst"
    echo "FAIR_CHIPS_ROOT=$FAIR_CHIPS_ROOT"
}
python FAIR_chips/fair_paths.py
