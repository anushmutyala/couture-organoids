#!/usr/bin/env python3
"""
Test script for ECG PID Control Loop
Shows chemical modulation to match normal ECG to afib patterns
"""

from ecg_pid_control import ECGPIDVisualizer
import os

def main():
    print("ECG PID Control Loop Test")
    print("=" * 50)
    
    # Check if datasets exist
    datasets_path = "./datasets"
    if not os.path.exists(datasets_path):
        print(f"Error: Datasets folder not found at {datasets_path}")
        return
    
    # Look for normal and afib files
    csv_files = [f for f in os.listdir(datasets_path) if f.endswith('.csv')]
    
    normal_file = None
    afib_file = None
    
    for file in csv_files:
        if 'normal' in file.lower():
            normal_file = os.path.join(datasets_path, file)
        elif 'afib' in file.lower():
            afib_file = os.path.join(datasets_path, file)
    
    if not normal_file or not afib_file:
        print("Error: Need both normal and afib ECG files")
        print(f"Available files: {csv_files}")
        return
    
    print(f"Normal ECG file: {os.path.basename(normal_file)}")
    print(f"Afib ECG file: {os.path.basename(afib_file)}")
    
    print("\nStarting PID control loop visualization...")
    print("This demonstrates:")
    print("1. Real-time ECG comparison (Normal vs Afib vs Controlled)")
    print("2. PID control block diagram")
    print("3. Chemical concentration modulation over time")
    print("4. Error signal (deviation area) minimization")
    print("5. Current chemical levels as gauges")
    print("6. Performance metrics")
    print("\nChemical A (Activator): Increases ECG activity")
    print("Chemical B (Inhibitor): Decreases ECG activity")
    print("Goal: Minimize deviation area between controlled and target signals")
    
    # Create and run visualizer
    visualizer = ECGPIDVisualizer(normal_file, afib_file)
    visualizer.start_animation()

if __name__ == "__main__":
    main() 