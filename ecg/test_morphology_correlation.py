#!/usr/bin/env python3
"""
Simple test script for ECG Morphology Correlator
"""

from ecg_morphology_correlator import ECGMorphologyCorrelator
import os

def test_basic_functionality():
    print("Testing ECG Morphology Correlator...")
    
    # Initialize correlator
    correlator = ECGMorphologyCorrelator(sampling_rate=500, window_size=0.8)
    
    # Check if datasets exist
    datasets_path = "./datasets"
    if not os.path.exists(datasets_path):
        print("❌ Datasets folder not found")
        return False
    
    csv_files = [f for f in os.listdir(datasets_path) if f.endswith('.csv')]
    if len(csv_files) < 2:
        print("❌ Need at least 2 CSV files for testing")
        return False
    
    try:
        # Test with first two files
        file1 = os.path.join(datasets_path, csv_files[0])
        file2 = os.path.join(datasets_path, csv_files[1])
        
        print(f"✓ Testing with: {csv_files[0]} vs {csv_files[1]}")
        
        # Test comparison
        results = correlator.compare_ecg_recordings(file1, file2)
        
        if results.get('error'):
            print(f"❌ Error: {results['error']}")
            return False
        
        print(f"✓ Successfully compared recordings")
        print(f"  Heartbeats 1: {results['num_heartbeats1']}")
        print(f"  Heartbeats 2: {results['num_heartbeats2']}")
        print(f"  Total comparisons: {results['total_comparisons']}")
        
        # Check if similarity scores were calculated
        if results['similarity_scores']:
            print(f"✓ Similarity scores calculated successfully")
            for key, value in list(results['similarity_scores'].items())[:5]:  # Show first 5
                print(f"  {key}: {value:.3f}")
        else:
            print("❌ No similarity scores calculated")
            return False
        
        print("\n🎉 ECG Morphology Correlator is working correctly!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    test_basic_functionality() 