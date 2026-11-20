# CRC environment for TeacherRegions: the suite's env (python, SARDET100K_ROOT, stage_dataset) plus the teacher weights.
#   source TeacherRegions/crc/env.sh      (from the repository root, inside a job script)
_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
pushd "$_REPO/SARDet" >/dev/null && source crc/env.sh && popd >/dev/null
# the SARMAE ViT-B checkpoint (models/SARMAE/README.md): copy it once with
#   scp models/SARMAE/SARMAE_vitb_checkpoint-last.pth nik96@h2p.crc.pitt.edu:/ix/ageorge/nik96/data/models/SARMAE/
export SARMAE_CHECKPOINT=${SARMAE_CHECKPOINT:-/ix/ageorge/nik96/data/models/SARMAE/SARMAE_vitb_checkpoint-last.pth}
if [ ! -f "$SARMAE_CHECKPOINT" ]; then echo "ERROR: teacher checkpoint not found: $SARMAE_CHECKPOINT" >&2; [ "${BASH_SOURCE[0]}" != "$0" ] && return 1 || exit 1; fi
stage_dataset_from_repo_root() { pushd "$_REPO/SARDet" >/dev/null && stage_dataset && popd >/dev/null; }
