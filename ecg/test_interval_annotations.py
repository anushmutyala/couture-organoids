#!/usr/bin/env python3
"""
Test script for ECG interval annotations
Shows PR, QRS, and QT intervals with vertical bars on heartbeat comparison
"""

from ecg_morphology_correlator import ECGMorphologyCorrelator
import os

def main():
    print("Testing ECG Interval Annotations")
    print("=" * 50)
    
    # Check if datasets folder exists
    datasets_path = "./datasets"
    if not os.path.exists(datasets_path):
        print(f"Error: Datasets folder not found at {datasets_path}")
        return
    
    # Get CSV files
    csv_files = [f for f in os.listdir(datasets_path) if f.endswith('.csv')]
    
    if len(csv_files) < 2:
        print("Need at least 2 CSV files for comparison")
        return
    
    print(f"Found {len(csv_files)} ECG files: {csv_files}")
    
    # Initialize correlator
    correlator = ECGMorphologyCorrelator(sampling_rate=500, window_size=0.8)
    
    # Compare first two files
    file1 = os.path.join(datasets_path, csv_files[0])
    file2 = os.path.join(datasets_path, csv_files[1])
    
    print(f"\nComparing: {csv_files[0]} vs {csv_files[1]}")
    print("This will show:")
    print("- Both ECG recordings overlaid on the same graph")
    print("- Purple shaded areas showing deviation between recordings")
    print("- Individual heartbeat comparison with PR, QRS, QT interval annotations")
    print("- Vertical bars marking P, Q, R, S, and T waves")
    print("- Interval measurements in milliseconds")
    print("- Summary statistics including ECG intervals")
    
    # Generate the comparison plot with interval annotations
    correlator.plot_simple_comparison(file1, file2)
    
    print("\nVisualization complete!")
    print("The vertical bars show:")
    print("- Green dashed lines: P-wave and R-peak (PR interval)")
    print("- Orange dashed lines: Q-wave and S-wave (QRS duration)")
    print("- Red dashed line: T-wave (QT interval)")
    print("The annotations show the measured intervals in milliseconds.")

if __name__ == "__main__":
    main() 