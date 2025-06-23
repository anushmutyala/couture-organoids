"""
ECG Analysis Package

This package provides comprehensive ECG analysis tools including:
- Signal generation and processing
- Morphology correlation analysis
- Interval analysis and measurements
- PID control for signal convergence
- Real-time ECG reading and analysis
"""

from .ecg_morphology_correlator import ECGMorphologyCorrelator
from .ecg_analyzer import ECGAnalyzer
from .synthetic_ecg_generator import SyntheticECGGenerator
from .ecg_pid_control import ECGPIDController
from .ecg_reader import ECGReader

__version__ = "1.0.0"
__author__ = "Couture Organoids Team"

__all__ = [
    "ECGMorphologyCorrelator",
    "ECGAnalyzer", 
    "SyntheticECGGenerator",
    "ECGPIDController",
    "ECGReader"
] 