#!/usr/bin/env python3
"""
Simple test script to verify ECG module imports
"""

import sys
from pathlib import Path

def test_imports():
    """Test ECG module imports"""
    print("Testing ECG module imports...")
    
    # Add ECG folder to path
    ecg_path = Path(__file__).parent / "ecg"
    sys.path.append(str(ecg_path))
    
    # Test package import
    try:
        import ecg
        print("✓ ecg package imported successfully")
        print(f"  Version: {ecg.__version__}")
        print(f"  Available classes: {ecg.__all__}")
    except ImportError as e:
        print(f"✗ ecg package import failed: {e}")
        return False
    
    # Test individual class imports
    try:
        from ecg import ECGMorphologyCorrelator
        correlator = ECGMorphologyCorrelator(sampling_rate=500)
        print("✓ ECGMorphologyCorrelator imported and instantiated")
    except Exception as e:
        print(f"✗ ECGMorphologyCorrelator failed: {e}")
    
    try:
        from ecg import ECGAnalyzer
        analyzer = ECGAnalyzer(sampling_rate=500)
        print("✓ ECGAnalyzer imported and instantiated")
    except Exception as e:
        print(f"✗ ECGAnalyzer failed: {e}")
    
    try:
        from ecg import SyntheticECGGenerator
        generator = SyntheticECGGenerator(sampling_rate=500)
        print("✓ SyntheticECGGenerator imported and instantiated")
    except Exception as e:
        print(f"✗ SyntheticECGGenerator failed: {e}")
    
    try:
        from ecg import ECGPIDController
        pid = ECGPIDController(kp=0.8, ki=0.2, kd=0.1)
        print("✓ ECGPIDController imported and instantiated")
    except Exception as e:
        print(f"✗ ECGPIDController failed: {e}")
    
    return True

if __name__ == "__main__":
    test_imports() 