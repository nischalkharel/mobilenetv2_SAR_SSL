"""The SparK authors' pre-training code (keyu-tian/SparK, commit a63e386, MIT licence; see LICENSE here).

    lamb.py, lr_control.py      byte-identical copies of pretrain/utils/lamb.py and pretrain/utils/lr_control.py
    encoder.py, spark.py,       copies of pretrain/encoder.py, spark.py, decoder.py in which ONLY the import lines
    decoder.py                  marked "CHANGED (A3)" differ (timm is not installed; package-relative imports)

All six files are byte-identical to mobilenetv2_SAR_SSL/Experiments/mobileNetv2_SparK/spark_lib/ (the ATRNet-STAR
A3 copies, whose smoke test compared them line by line with the authors' originals); SparK_SARDet/smoke_test.py
re-checks the hashes when that folder is present. Everything specific to MobileNetV2 and SAR is in
../spark_model.py, never in this folder.
"""
