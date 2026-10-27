"""The SparK authors' pre-training code (keyu-tian/SparK, commit a63e386, MIT licence; see LICENSE here).

    lamb.py, lr_control.py      byte-identical copies of pretrain/utils/lamb.py and pretrain/utils/lr_control.py
    encoder.py, spark.py,       copies of pretrain/encoder.py, spark.py, decoder.py in which ONLY the import lines
    decoder.py                  marked "CHANGED (A3)" differ (timm is not installed; package-relative imports)

smoke_test.py checks both statements against reference/SparK-main_original/. Everything specific to
MobileNetV2 and SAR is in ../spark_model.py, never in this folder.
"""
