# New file (see lejepa_lib/__init__.py). The original lejepa/univariate/__init__.py also imports tests needing scipy.
from .base import UnivariateTest
from .epps_pulley import EppsPulley

__all__ = ["UnivariateTest", "EppsPulley"]
