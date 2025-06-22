#!/usr/bin/env python3
"""
Quick script to run ECG analysis on datasets
"""

import os
import sys
from ecg_analyzer import ECGAnalyzer

def main():
    print("ECG Analysis - Extracting Key Metrics")
    print("=" * 50)
    
    # Check if datasets folder exists
    datasets_path = "./datasets"
    if not os.path.exists(datasets_path):
        print(f"Error: Datasets folder not found at {datasets_path}")
        print("Please ensure the datasets folder contains CSV files with Time,ECG columns")
        return
    
    # Initialize analyzer
    analyzer = ECGAnalyzer(sampling_rate=500)
    
    # Analyze all datasets
    print("Analyzing ECG datasets...")
    results_df = analyzer.analyze_all_datasets(datasets_path)
    
    if not results_df.empty:
        print("\n" + "="*80)
        print("ANALYSIS RESULTS")
        print("="*80)
        print(results_df.to_string(index=False))
        
        # Save results
        output_file = "ecg_analysis_results.csv"
        results_df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
        
        # Display summary
        print("\n" + "="*50)
        print("SUMMARY STATISTICS")
        print("="*50)
        
        # Filter out rows with errors
        valid_results = results_df[results_df['error'].isna()]
        
        if not valid_results.empty:
            print(f"Files analyzed: {len(valid_results)}")
            print(f"Average Heart Rate: {valid_results['heart_rate'].mean():.1f} BPM")
            print(f"Average PR Interval: {valid_results['pr_interval'].mean():.1f} ms")
            print(f"Average QTc Interval: {valid_results['qtc_interval'].mean():.1f} ms")
            print(f"Average HRV RMSSD: {valid_results['hrv_rmssd'].mean():.1f} ms")
            
            # Show individual file results
            print("\n" + "="*50)
            print("INDIVIDUAL FILE RESULTS")
            print("="*50)
            for _, row in valid_results.iterrows():
                print(f"\n{row['filename']}:")
                print(f"  Heart Rate: {row['heart_rate']} BPM")
                print(f"  PR Interval: {row['pr_interval']} ms")
                print(f"  QTc Interval: {row['qtc_interval']} ms")
                print(f"  HRV RMSSD: {row['hrv_rmssd']} ms")
                print(f"  Beats detected: {row['num_beats']}")
        else:
            print("No valid results found. Check for errors in the analysis.")
        
        # Show files with errors
        error_results = results_df[results_df['error'].notna()]
        if not error_results.empty:
            print("\n" + "="*50)
            print("FILES WITH ERRORS")
            print("="*50)
            for _, row in error_results.iterrows():
                print(f"{row['filename']}: {row['error']}")
    
    else:
        print("No CSV files found in the datasets folder.")
        print("Please ensure your CSV files have 'Time' and 'ECG' columns.")

if __name__ == "__main__":
    main() 