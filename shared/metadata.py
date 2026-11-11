"""Per-image metadata of SARDet-100K, recovered from the ORIGINAL file names in mapping.json.

The annotation JSON carries no metadata, but mapping.json maps every original file name (e.g.
`7Shipdataset_Gao_ship_hh_02017022546020508.jpg`) to the anonymised name (`0000004.jpg`). The original name
gives the source dataset, and for some sources the satellite, polarisation and the parent image a patch was
cut from. Sensor, band and resolution per source come from Table 2 of the SARDet-100K paper (Li et al. 2024,
arXiv 2403.06534); values the paper lists as a mixture are recorded as "mixed" rather than guessed.

Fields produced for every image (see parse_original_name):
    source       AIR_SARShip, HRSID, MSAR, SADD, SAR-AIRcraft, Shipdataset, SSDD, OGSOD, SIVED
    satellite    per image where the name gives it (Shipdataset: Sentinel-1 / Gaofen-3; SIVED: MSTAR / FARAD),
                 else the paper's value for the source ("mixed(...)" when the paper lists several)
    polarization per image for Shipdataset (HH/HV/VH/VV); else the paper's value, "mixed" or "single(unknown)"
    band         C, X, mixed(...) (paper, per source; SIVED per sub-source)
    resolution   paper's per-source class (string)
    parent       id of the large image a patch was cut from, where the name shows it (SAR-AIRcraft, AIR_SARShip-1:
                 512 px patches with 200 px overlap); otherwise the file itself. Grouping unit for subset drawing.
    scene        HRSID scene id (800 px tiles, <= 25 % overlap; recorded, not a grouping unit) or the parent
                 for the patch sources; "" elsewhere.
"""
import csv
import json
import re
from pathlib import Path

from . import paths

# Paper Table 2 (per source): satellite, band, polarisations, resolution. "mixed(...)" = several listed.
SOURCE_TABLE = {
    "AIR_SARShip":  dict(satellite="Gaofen-3", band="C", polarization="VV", resolution="1m,3m"),
    "HRSID":        dict(satellite="mixed(S-1B,TerraSAR-X,TanDEM-X)", band="mixed(C,X)", polarization="mixed", resolution="0.5-3m"),
    "MSAR":         dict(satellite="HISEA-1", band="C", polarization="mixed", resolution="<=1m"),
    "SADD":         dict(satellite="TerraSAR-X", band="X", polarization="HH", resolution="0.5-3m"),
    "SAR-AIRcraft": dict(satellite="Gaofen-3", band="C", polarization="single(unknown)", resolution="1m"),
    "Shipdataset":  dict(satellite="per-image", band="C", polarization="per-image", resolution="3-25m"),
    "SSDD":         dict(satellite="mixed(S-1,RadarSat-2,TerraSAR-X)", band="mixed(C,X)", polarization="mixed", resolution="1-15m"),
    "OGSOD":        dict(satellite="Gaofen-3", band="C", polarization="mixed(VV,VH)", resolution="3m"),
    "SIVED":        dict(satellite="per-image", band="per-image", polarization="mixed(VV,HH)", resolution="0.1m,0.3m"),
}
SIVED_SUBSOURCE = {"MSTAR": dict(satellite="airborne-MSTAR", band="X"),
                   "FARAD": dict(satellite="airborne-FARAD", band="mixed(Ka,Ku)")}
SHIPDATASET_SATELLITE = {"Sen": "Sentinel-1", "Gao": "Gaofen-3"}

_SOURCE_RE = re.compile(r"^\d*(AIR|HRSID|MSAR|SADD|SAR-AIRcraft|Shipdataset|SSDD|OGSOD|SIVED)")


def parse_original_name(original: str) -> dict:
    """Metadata fields from one original file name. Raises ValueError for an unknown pattern."""
    m = _SOURCE_RE.match(original)
    if not m:
        raise ValueError(f"unknown source in original name {original!r}")
    src = m.group(1)
    stem = Path(original).stem
    if src == "AIR":
        # 1AIR_SARShip_1_3__512__1872___1248 (patch of large image AIR_SARShip_1_3) / 2AIR_SARShip_2_265 (no patching)
        src = "AIR_SARShip"
        parent = stem.split("__")[0].lstrip("0123456789")
    elif src == "SAR-AIRcraft":
        parent = "SAR-AIRcraft_" + stem.split("_")[1]            # 6SAR-AIRcraft_0001872__512__488___312
    elif src == "HRSID":
        # 3HRSID_P0047_3600_4400_0_800: an 800 px tile of scene P0047. Tiles overlap neighbours by <= 25 % and
        # HRSID's own split mixes scenes, so the scene is RECORDED (scene column) but is not the grouping unit:
        # grouping by scene would exclude 3,638 of the 3,642 train tiles (audit, 2026-11-11).
        parent = stem.lstrip("0123456789")
        scene = "HRSID_" + stem.split("_")[1]
    else:
        parent = stem.lstrip("0123456789")                        # the file itself
    info = dict(SOURCE_TABLE[src])
    info.update(source=src, parent=parent, original=original)
    info["scene"] = scene if src == "HRSID" else (parent if src in ("SAR-AIRcraft", "AIR_SARShip") and parent != stem.lstrip("0123456789") else "")
    if src == "Shipdataset":
        parts = stem.split("_")                                   # 7Shipdataset_Gao_ship_hh_0201...
        if len(parts) >= 4 and parts[1] in SHIPDATASET_SATELLITE:
            info["satellite"] = SHIPDATASET_SATELLITE[parts[1]]
            info["polarization"] = parts[3].upper()
        else:                                                     # a few names like 7Shipdataset_newship0150707
            info["satellite"] = "unknown"
            info["polarization"] = "unknown"
    elif src == "SIVED":
        sub = stem.split("_")[1]                                  # 10SIVED_MSTAR_89
        info.update(SIVED_SUBSOURCE.get(sub, dict(satellite="airborne-unknown", band="unknown")))
    return info


def load_mapping() -> dict:
    """anonymised file name -> original file name."""
    with open(paths.MAPPING_JSON, encoding="utf-8") as f:
        original_to_anon = json.load(f)
    inv = {v: k for k, v in original_to_anon.items()}
    if len(inv) != len(original_to_anon):
        raise ValueError("mapping.json is not one-to-one")
    return inv


METADATA_COLUMNS = ["file_name", "official_split", "image_id", "width", "height", "source", "satellite",
                    "polarization", "band", "resolution", "parent", "scene", "group", "md5", "pixel_md5", "original"]


def read_metadata(csv_path=None) -> list:
    """Rows of data/image_metadata.csv as dicts (one per image of the whole release)."""
    with open(csv_path or paths.METADATA_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["image_id"] = int(r["image_id"])
        r["width"] = int(r["width"])
        r["height"] = int(r["height"])
    return rows


# Condition-cell definitions for CondDINO (fields that exist for every image; labels never used)
CELL_DEFINITIONS = {
    "source_pol": ["source", "satellite", "polarization"],   # Shipdataset split by satellite x pol, SIVED by sub-source (19 cells)
    "source": ["source"],                                    # 9 cells
    # "sensor_band" (v1 of the design) was dropped: sensor and band are not per-image for 7 of 9 sources.
}


def cell_key(row: dict, fields) -> str:
    return "|".join(str(row[f]) for f in fields)
