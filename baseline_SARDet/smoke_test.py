"""baseline_SARDet smoke test: paths, data, detector, a 1-epoch fine-tune on 8 images, probes and panels.

    python baseline_SARDet/smoke_test.py [--cpu]
Pipeline check only (outputs go to baseline_SARDet/outputs/smoke/); never a result.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from shared import smoke_common as sc  # noqa: E402

if __name__ == "__main__":
    device = sc.device_arg()
    sc.check_paths()
    sc.check_detection_data()
    sc.check_detector(device or ("cuda" if __import__("torch").cuda.is_available() else "cpu"))
    run_dir = sc.run_finetune_smoke(HERE, "baseline_imagenet", "imagenet", device=device)
    sc.run_analysis_smoke(run_dir, "imagenet", device=device)
    print("\nSMOKE TEST PASSED (baseline_SARDet)")
