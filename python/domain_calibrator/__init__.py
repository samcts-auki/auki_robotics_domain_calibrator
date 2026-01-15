"""
Domain Calibrator Package

A Python package for calibrating cameras to domain coordinate systems using QR codes.
"""

import logging
from .domain_calibrator import DomainCalibratorPy, load_config, rpy_from_matrix_scipy
from .domain import Domain, transformation_matrix
from .marker_calibrator import Calibrator, average_transforms

logger = logging.getLogger(__name__)

__all__ = [
    # Main classes
    'DomainCalibratorPy',
    'Domain',
    'Calibrator',
    # Utility functions
    'average_transforms',
    'load_config',
    'rpy_from_matrix_scipy',
    'transformation_matrix',
]

__version__ = '0.1.0'
