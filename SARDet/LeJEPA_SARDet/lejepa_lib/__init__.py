"""Unchanged copies of the three LeJEPA library files this experiment needs (SIGReg).

Source: the official repository rbalestr-lab/lejepa, downloaded as a ZIP (lejepa-main), files
    lejepa/univariate/base.py, lejepa/univariate/epps_pulley.py, lejepa/multivariate/slicing.py
They are byte-identical to the originals in reference/lejepa-main_original/ (smoke_test.py checks the
SHA-256 hashes). Only these package __init__ files are new: the original lejepa/__init__.py also
imports other statistical tests that need scipy, which this experiment does not use.
Licence: CC BY-NC 4.0 (reference/lejepa-main_original/LICENSE).
"""
from . import multivariate, univariate

__all__ = ["univariate", "multivariate"]
