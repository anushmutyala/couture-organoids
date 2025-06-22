#!/usr/bin/env python3
"""
Demo script for ECG Morphology Correlation
Tests the correlation system with different ECG recordings
"""

import os
import sys
from ecg_morphology_correlator import ECGMorphologyCorrelator

def main():
    print("ECG Morphology Correlation Demo")
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
    
    # Compare all pairs
    results = []
    
    for i in range(len(csv_files)):
        for j in range(i + 1, len(csv_files)):
            file1 = os.path.join(datasets_path, csv_files[i])
            file2 = os.path.join(datasets_path, csv_files[j])
            
            print(f"\n{'='*60}")
            print(f"Comparing: {csv_files[i]} vs {csv_files[j]}")
            print(f"{'='*60}")
            
            # Compare recordings
            comparison_result = correlator.compare_ecg_recordings(file1, file2)
            
            if comparison_result.get('error'):
                print(f"Error: {comparison_result['error']}")
                continue
            
            # Display results
            print(f"\nComparison Summary:")
            print(f"Recording 1: {comparison_result['recording1']}")
            print(f"Recording 2: {comparison_result['recording2']}")
            print(f"Heartbeats 1: {comparison_result['num_heartbeats1']}")
            print(f"Heartbeats 2: {comparison_result['num_heartbeats2']}")
            print(f"Total comparisons: {comparison_result['total_comparisons']}")
            
            print(f"\nSimilarity Scores:")
            print("-" * 40)
            for key, value in comparison_result['similarity_scores'].items():
                print(f"{key:25s}: {value:.3f}")
            
            # Store results
            results.append({
                'file1': csv_files[i],
                'file2': csv_files[j],
                'overall_similarity': comparison_result['similarity_scores'].get('mean_overall_similarity', 0.0),
                'signal_correlation': comparison_result['similarity_scores'].get('mean_signal_correlation', 0.0),
                'feature_similarity': comparison_result['similarity_scores'].get('mean_feature_similarity', 0.0),
                'cosine_similarity': comparison_result['similarity_scores'].get('mean_cosine_similarity', 0.0)
            })
            
            # Plot comparison for first pair
            if i == 0 and j == 1:
                print(f"\nGenerating simple morphology comparison plot...")
                correlator.plot_simple_comparison(file1, file2)
    
    # Summary table
    if results:
        print(f"\n{'='*80}")
        print("MORPHOLOGY CORRELATION SUMMARY")
        print(f"{'='*80}")
        print(f"{'File 1':<20s} {'File 2':<20s} {'Overall':<8s} {'Signal':<8s} {'Feature':<8s} {'Cosine':<8s}")
        print("-" * 80)
        
        for result in results:
            print(f"{result['file1']:<20s} {result['file2']:<20s} "
                  f"{result['overall_similarity']:<8.3f} {result['signal_correlation']:<8.3f} "
                  f"{result['feature_similarity']:<8.3f} {result['cosine_similarity']:<8.3f}")
        
        # Find most similar pair
        most_similar = max(results, key=lambda x: x['overall_similarity'])
        print(f"\nMost Similar Pair: {most_similar['file1']} vs {most_similar['file2']}")
        print(f"Overall Similarity: {most_similar['overall_similarity']:.3f}")
        
        # Find least similar pair
        least_similar = min(results, key=lambda x: x['overall_similarity'])
        print(f"Least Similar Pair: {least_similar['file1']} vs {least_similar['file2']}")
        print(f"Overall Similarity: {least_similar['overall_similarity']:.3f}")
        
        # Save results
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv("morphology_correlation_results.csv", index=False)
        print(f"\nResults saved to: morphology_correlation_results.csv")

if __name__ == "__main__":
    main() 