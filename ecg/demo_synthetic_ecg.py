#!/usr/bin/env python3
"""
Demo script for Synthetic ECG Generator
Quick test of different ECG generation methods
"""

from synthetic_ecg_generator import SyntheticECGGenerator
import matplotlib.pyplot as plt

def main():
    print("Synthetic ECG Generator - Quick Demo")
    print("=" * 40)
    
    # Initialize generator
    generator = SyntheticECGGenerator(sampling_rate=500, duration=5.0)
    
    # Create subplots for comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Synthetic ECG Signals Comparison', fontsize=16)
    
    # 1. Normal sinus rhythm
    print("Generating normal sinus rhythm...")
    time, normal_ecg = generator.generate_physiological_ecg()
    normal_ecg = generator.add_realistic_noise(normal_ecg)
    axes[0, 0].plot(time, normal_ecg, 'b-', linewidth=1)
    axes[0, 0].set_title('Normal Sinus Rhythm (72 BPM)')
    axes[0, 0].set_ylabel('Amplitude (mV)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Tachycardia
    print("Generating tachycardia...")
    time, tachy_ecg = generator.generate_abnormal_rhythms('tachycardia', {'heart_rate': 120})
    tachy_ecg = generator.add_realistic_noise(tachy_ecg)
    axes[0, 1].plot(time, tachy_ecg, 'r-', linewidth=1)
    axes[0, 1].set_title('Tachycardia (120 BPM)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Bradycardia
    print("Generating bradycardia...")
    time, brady_ecg = generator.generate_abnormal_rhythms('bradycardia', {'heart_rate': 45})
    brady_ecg = generator.add_realistic_noise(brady_ecg)
    axes[1, 0].plot(time, brady_ecg, 'g-', linewidth=1)
    axes[1, 0].set_title('Bradycardia (45 BPM)')
    axes[1, 0].set_xlabel('Time (seconds)')
    axes[1, 0].set_ylabel('Amplitude (mV)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Simple mathematical model
    print("Generating simple mathematical model...")
    time, simple_ecg = generator.generate_simple_mathematical_ecg(heart_rate=80, noise_level=0.1)
    axes[1, 1].plot(time, simple_ecg, 'm-', linewidth=1)
    axes[1, 1].set_title('Simple Mathematical Model (80 BPM)')
    axes[1, 1].set_xlabel('Time (seconds)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("Demo completed! Check the plots above.")
    print("\nNext steps:")
    print("1. Run 'python synthetic_ecg_generator.py' for full demo with data saving")
    print("2. Modify parameters in the generator for different heart rates and rhythms")
    print("3. Add more noise types or abnormal rhythms as needed")

if __name__ == "__main__":
    main() 