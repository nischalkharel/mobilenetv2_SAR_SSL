"""Shared package of the SARDet-100K detection suite: data, splits, detector, evaluation, records, plots.

Every experiment folder imports from here, so the fine-tuning and evaluation protocol is identical across
methods. Scripts inside an experiment folder put the suite root on sys.path first:
    import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
"""
