"""Print the path of the SSL backbone (SARDet pool, or the ATRNet pool for the atr_* methods) to use for one method and one ATRNet split.

    python SARDet_to_ATR/find_backbone.py dino 2      ->  SARDet/DINO_SARDet/outputs/ssl/<run>_split2_seed1/backbone_epoch100.pth

Pairing rule (fixed in advance): ATRNet split s uses the SARDet SSL run of split s, i.e. SSL seed s - 1. The
SARDet SSL pool is the same for all three runs, so the three pairs vary the SSL seed and the ATRNet labeled
draw together, exactly as the A1/A2/A3 rows did (one SSL run per split). The newest COMPLETED run is used; its
file SHA-256 is recorded by the fine-tune in run_info.json (init.sha256).
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FOLDERS = {"dino": ("SARDet/DINO_SARDet", "dino_ssl_mobilenetv2"), "lejepa": ("SARDet/LeJEPA_SARDet", "lejepa_ssl_mobilenetv2"),
           "spark": ("SARDet/SparK_SARDet", "spark_ssl_mobilenetv2"),
           # RegionDINO (RegionSSL study): scene + object, object only, and their random-window controls
           "regiondino_so": ("RegionSSL/RegionDINO_scene_object", "regiondino_so_ssl_mobilenetv2"),
           "regiondino_so_random": ("RegionSSL/RegionDINO_scene_object", "regiondino_so_random_ssl_mobilenetv2"),
           "regiondino_o": ("RegionSSL/RegionDINO_object", "regiondino_o_ssl_mobilenetv2"),
           "regiondino_o_random": ("RegionSSL/RegionDINO_object", "regiondino_o_random_ssl_mobilenetv2"),
           # ViewScale sweep (ViewScale/): DINO on random square windows of one fixed extent, SARDet pool
           **{f"viewscale_e{e}": ("ViewScale", f"viewscale_e{e}_ssl_mobilenetv2") for e in (32, 64, 128, 256, 512)},
           # TeacherRegions: regions proposed by the frozen SARMAE teacher, and the size-matched random control
           "regiondino_o_teacher": ("TeacherRegions", "regiondino_o_teacher_ssl_mobilenetv2"),
           "regiondino_o_teacher_random": ("TeacherRegions", "regiondino_o_teacher_random_ssl_mobilenetv2"),
           # arm I: the same engine on the ATRNet-STAR SSL pool (in-domain); atr_dino = plain DINO of that engine, the paired reference
           "atr_dino": ("RegionSSL/DINO_reference", "dino_suite_atr_ssl_mobilenetv2"),
           "atr_regiondino_so": ("RegionSSL/RegionDINO_scene_object", "regiondino_so_atr_ssl_mobilenetv2"),
           "atr_regiondino_so_random": ("RegionSSL/RegionDINO_scene_object", "regiondino_so_atr_random_ssl_mobilenetv2"),
           "atr_regiondino_o": ("RegionSSL/RegionDINO_object", "regiondino_o_atr_ssl_mobilenetv2"),
           "atr_regiondino_o_random": ("RegionSSL/RegionDINO_object", "regiondino_o_atr_random_ssl_mobilenetv2")}


def find(method: str, split: int) -> Path:
    folder, run_name = FOLDERS[method]
    pattern = f"*_{run_name}_split{split}_seed{split - 1}"
    runs = sorted((REPO / folder / "outputs" / "ssl").glob(pattern), reverse=True)
    for run in runs:
        info = json.loads((run / "run_info.json").read_text(encoding="utf-8"))
        file = run / f"backbone_epoch{info['config']['common']['epochs']}.pth"
        if info.get("status") == "completed" and file.exists():
            return file
    raise SystemExit(f"no completed SARDet SSL run with an exported backbone for {method} split {split} ({pattern})")


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in FOLDERS or sys.argv[2] not in ("1", "2", "3"):
        raise SystemExit(f"usage: find_backbone.py {'|'.join(FOLDERS)} 1|2|3")
    print(find(sys.argv[1], int(sys.argv[2])))
