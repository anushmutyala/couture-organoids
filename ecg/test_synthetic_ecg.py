#!/usr/bin/env python3
"""
Simple test script for Synthetic ECG Generator
"""

from synthetic_ecg_generator import SyntheticECGGenerator

def test_basic_functionality():
    print("Testing Synthetic ECG Generator...")
    
    # Initialize generator
    generator = SyntheticECGGenerator(sampling_rate=500, duration=3.0)
    
    try:
        # Test 1: Simple mathematical model
        print("✓ Testing simple mathematical model...")
        time, ecg = generator.generate_simple_mathematical_ecg(heart_rate=72)
        print(f"   Generated {len(ecg)} samples")
        
        # Test 2: Physiological model
        print("✓ Testing physiological model...")
        time, ecg = generator.generate_physiological_ecg()
        print(f"   Generated {len(ecg)} samples")
        
        # Test 3: Abnormal rhythms
        print("✓ Testing abnormal rhythms...")
        time, ecg = generator.generate_abnormal_rhythms('tachycardia', {'heart_rate': 120})
        print(f"   Generated {len(ecg)} samples")
        
        # Test 4: Noise addition
        print("✓ Testing noise addition...")
        clean_ecg = generator.generate_physiological_ecg()[1]
        noisy_ecg = generator.add_realistic_noise(clean_ecg)
        print(f"   Added noise to {len(noisy_ecg)} samples")
        
        print("\n🎉 All tests passed! The synthetic ECG generator is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    test_basic_functionality() 