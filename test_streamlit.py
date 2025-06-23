#!/usr/bin/env python3
"""
Test script for Streamlit ECG Dashboard
Verifies imports and basic functionality
"""

import sys
from pathlib import Path

def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    
    try:
        import numpy as np
        print("✓ numpy imported successfully")
    except ImportError as e:
        print(f"✗ numpy import failed: {e}")
        return False
    
    try:
        import streamlit as st
        print("✓ streamlit imported successfully")
    except ImportError as e:
        print(f"✗ streamlit import failed: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✓ matplotlib imported successfully")
    except ImportError as e:
        print(f"✗ matplotlib import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✓ pandas imported successfully")
    except ImportError as e:
        print(f"✗ pandas import failed: {e}")
        return False
    
    try:
        import scipy
        print("✓ scipy imported successfully")
    except ImportError as e:
        print(f"✗ scipy import failed: {e}")
        return False
    
    # Test ECG module imports
    ecg_path = Path(__file__).parent / "ecg"
    sys.path.append(str(ecg_path))
    
    try:
        from ecg_morphology_correlator import ECGMorphologyCorrelator
        print("✓ ECGMorphologyCorrelator imported successfully")
    except ImportError as e:
        print(f"✗ ECGMorphologyCorrelator import failed: {e}")
        return False
    
    try:
        from ecg_analyzer import ECGAnalyzer
        print("✓ ECGAnalyzer imported successfully")
    except ImportError as e:
        print(f"✗ ECGAnalyzer import failed: {e}")
        return False
    
    try:
        from synthetic_ecg_generator import SyntheticECGGenerator
        print("✓ SyntheticECGGenerator imported successfully")
    except ImportError as e:
        print(f"✗ SyntheticECGGenerator import failed: {e}")
        return False
    
    try:
        from ecg_pid_control import ECGPIDController
        print("✓ ECGPIDController imported successfully")
    except ImportError as e:
        print(f"✗ ECGPIDController import failed: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality"""
    print("\nTesting basic functionality...")
    
    try:
        import numpy as np
        from collections import deque
        
        # Test data generation
        def generate_realistic_ecg(t: float, base_freq: float = 1.2, 
                                 noise_level: float = 0.05, 
                                 irregularity: float = 0.0) -> float:
            fundamental = np.sin(2 * np.pi * base_freq * t)
            harmonic1 = 0.3 * np.sin(4 * np.pi * base_freq * t)
            harmonic2 = 0.1 * np.sin(6 * np.pi * base_freq * t)
            noise = noise_level * np.random.randn()
            if irregularity > 0:
                irregularity_factor = 1 + irregularity * np.sin(3 * np.pi * base_freq * t)
                fundamental *= irregularity_factor
            return fundamental + harmonic1 + harmonic2 + noise
        
        # Test signal generation
        t = 1.0
        signal = generate_realistic_ecg(t)
        print(f"✓ Signal generation works: {signal:.3f}")
        
        # Test deviation calculation
        def calculate_deviation_area(signal1: np.ndarray, signal2: np.ndarray) -> float:
            min_len = min(len(signal1), len(signal2))
            sig1 = signal1[:min_len]
            sig2 = signal2[:min_len]
            time_axis = np.arange(min_len)
            return np.trapz(np.abs(sig1 - sig2), time_axis)
        
        sig1 = np.array([1, 2, 3, 4, 5])
        sig2 = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        deviation = calculate_deviation_area(sig1, sig2)
        print(f"✓ Deviation calculation works: {deviation:.3f}")
        
        # Test ECG objects initialization
        from ecg_morphology_correlator import ECGMorphologyCorrelator
        from ecg_analyzer import ECGAnalyzer
        from synthetic_ecg_generator import SyntheticECGGenerator
        from ecg_pid_control import ECGPIDController
        
        correlator = ECGMorphologyCorrelator(sampling_rate=500, window_size=0.8)
        analyzer = ECGAnalyzer(sampling_rate=500)
        generator = SyntheticECGGenerator(sampling_rate=500)
        pid_controller = ECGPIDController(kp=0.8, ki=0.2, kd=0.1)
        
        print("✓ ECG objects initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

def main():
    """Main test function"""
    print("Streamlit ECG Dashboard - Import and Functionality Test")
    print("=" * 60)
    
    # Test imports
    imports_ok = test_imports()
    
    if imports_ok:
        # Test functionality
        functionality_ok = test_basic_functionality()
        
        if functionality_ok:
            print("\n🎉 All tests passed! The Streamlit app should work correctly.")
            print("\nTo run the dashboard:")
            print("1. Install requirements: pip install -r requirements.txt")
            print("2. Run the app: streamlit run streamlit-app.py")
        else:
            print("\n❌ Functionality tests failed. Check the error messages above.")
    else:
        print("\n❌ Import tests failed. Install missing dependencies:")
        print("pip install -r requirements.txt")

if __name__ == "__main__":
    main() 