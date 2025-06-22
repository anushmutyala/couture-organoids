#!/usr/bin/env python3
"""
ECG Analyzer
Extracts key ECG metrics from datasets:
- Resting Heart Rate (BPM)
- QTc Interval (ms)
- PR Interval (ms)
- HRV RMSSD (ms)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks
from typing import Dict, List, Tuple, Optional
import os
import glob

class ECGAnalyzer:
    """
    Comprehensive ECG analysis tool for extracting key cardiac metrics
    """
    
    def __init__(self, sampling_rate: int = 500):
        """
        Initialize ECG analyzer
        
        Args:
            sampling_rate (int): Sampling rate in Hz
        """
        self.sampling_rate = sampling_rate
        self.dt = 1.0 / sampling_rate  # Time step
        
    def load_ecg_data(self, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load ECG data from CSV file
        
        Args:
            filepath (str): Path to CSV file
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Time and ECG signal arrays
        """
        df = pd.read_csv(filepath)
        time = np.array(df['Time'].values)
        ecg = np.array(df['ECG'].values)
        return time, ecg
    
    def preprocess_ecg(self, ecg: np.ndarray) -> np.ndarray:
        """
        Preprocess ECG signal (filtering, baseline correction)
        
        Args:
            ecg (np.ndarray): Raw ECG signal
            
        Returns:
            np.ndarray: Preprocessed ECG signal
        """
        # Remove baseline wander with high-pass filter
        result = signal.butter(3, 0.5, 'high', fs=self.sampling_rate)
        b, a = result[0], result[1]
        filtered_ecg = signal.filtfilt(b, a, ecg)
        
        # Remove powerline interference (50/60 Hz)
        result = signal.butter(4, [45, 65], 'bandstop', fs=self.sampling_rate)
        b, a = result[0], result[1]
        filtered_ecg = signal.filtfilt(b, a, filtered_ecg)
        
        return filtered_ecg
    
    def detect_r_peaks(self, ecg: np.ndarray, prominence: float = 0.5) -> np.ndarray:
        """
        Detect R-peaks in ECG signal
        
        Args:
            ecg (np.ndarray): ECG signal
            prominence (float): Minimum prominence for peak detection
            
        Returns:
            np.ndarray: Indices of R-peaks
        """
        # Find peaks with minimum prominence
        peaks, _ = find_peaks(ecg, prominence=prominence, distance=int(0.3 * self.sampling_rate))
        return peaks
    
    def calculate_heart_rate(self, r_peaks: np.ndarray, time: np.ndarray) -> float:
        """
        Calculate resting heart rate from R-peaks
        
        Args:
            r_peaks (np.ndarray): R-peak indices
            time (np.ndarray): Time array
            
        Returns:
            float: Heart rate in BPM
        """
        if len(r_peaks) < 2:
            return 0.0
        
        # Calculate RR intervals
        rr_intervals = np.diff(r_peaks) * self.dt
        
        # Remove outliers (RR intervals that are too short or too long)
        valid_rr = rr_intervals[(rr_intervals > 0.3) & (rr_intervals < 2.0)]
        
        if len(valid_rr) == 0:
            return 0.0
        
        # Calculate average heart rate
        avg_rr = float(np.mean(valid_rr))
        heart_rate = 60.0 / avg_rr
        
        return heart_rate
    
    def detect_p_waves(self, ecg: np.ndarray, r_peaks: np.ndarray) -> List[int]:
        """
        Detect P-waves before each R-peak
        
        Args:
            ecg (np.ndarray): ECG signal
            r_peaks (np.ndarray): R-peak indices
            
        Returns:
            List[int]: P-wave indices
        """
        p_peaks = []
        
        for r_peak in r_peaks:
            # Look for P-wave in window before R-peak (150-300ms before)
            start_idx = max(0, r_peak - int(0.3 * self.sampling_rate))
            end_idx = max(0, r_peak - int(0.15 * self.sampling_rate))
            
            if start_idx < end_idx:
                window = ecg[start_idx:end_idx]
                if len(window) > 0:
                    # Find local maximum in window
                    p_idx = start_idx + np.argmax(window)
                    p_peaks.append(p_idx)
                else:
                    p_peaks.append(-1)  # No P-wave detected
            else:
                p_peaks.append(-1)
        
        return p_peaks
    
    def detect_q_waves(self, ecg: np.ndarray, r_peaks: np.ndarray) -> List[int]:
        """
        Detect Q-waves before each R-peak
        
        Args:
            ecg (np.ndarray): ECG signal
            r_peaks (np.ndarray): R-peak indices
            
        Returns:
            List[int]: Q-wave indices
        """
        q_peaks = []
        
        for r_peak in r_peaks:
            # Look for Q-wave in window before R-peak (20-80ms before)
            start_idx = max(0, r_peak - int(0.08 * self.sampling_rate))
            end_idx = max(0, r_peak - int(0.02 * self.sampling_rate))
            
            if start_idx < end_idx:
                window = ecg[start_idx:end_idx]
                if len(window) > 0:
                    # Find local minimum in window
                    q_idx = start_idx + np.argmin(window)
                    q_peaks.append(q_idx)
                else:
                    q_peaks.append(-1)
            else:
                q_peaks.append(-1)
        
        return q_peaks
    
    def detect_t_waves(self, ecg: np.ndarray, r_peaks: np.ndarray) -> List[int]:
        """
        Detect T-waves after each R-peak
        
        Args:
            ecg (np.ndarray): ECG signal
            r_peaks (np.ndarray): R-peak indices
            
        Returns:
            List[int]: T-wave indices
        """
        t_peaks = []
        
        for r_peak in r_peaks:
            # Look for T-wave in window after R-peak (200-400ms after)
            start_idx = min(len(ecg) - 1, r_peak + int(0.2 * self.sampling_rate))
            end_idx = min(len(ecg) - 1, r_peak + int(0.4 * self.sampling_rate))
            
            if start_idx < end_idx:
                window = ecg[start_idx:end_idx]
                if len(window) > 0:
                    # Find local maximum in window
                    t_idx = start_idx + np.argmax(window)
                    t_peaks.append(t_idx)
                else:
                    t_peaks.append(-1)
            else:
                t_peaks.append(-1)
        
        return t_peaks
    
    def calculate_pr_interval(self, p_peaks: List[int], r_peaks: np.ndarray) -> float:
        """
        Calculate PR interval
        
        Args:
            p_peaks (List[int]): P-wave indices
            r_peaks (np.ndarray): R-peak indices
            
        Returns:
            float: Average PR interval in milliseconds
        """
        pr_intervals = []
        
        for i, (p_peak, r_peak) in enumerate(zip(p_peaks, r_peaks)):
            if p_peak != -1 and p_peak < r_peak:
                pr_interval = (r_peak - p_peak) * self.dt * 1000  # Convert to ms
                if 120 <= pr_interval <= 200:  # Normal PR interval range
                    pr_intervals.append(pr_interval)
        
        return float(np.mean(pr_intervals)) if pr_intervals else 0.0
    
    def calculate_qt_interval(self, q_peaks: List[int], t_peaks: List[int]) -> float:
        """
        Calculate QT interval
        
        Args:
            q_peaks (List[int]): Q-wave indices
            t_peaks (List[int]): T-wave indices
            
        Returns:
            float: Average QT interval in milliseconds
        """
        qt_intervals = []
        
        for q_peak, t_peak in zip(q_peaks, t_peaks):
            if q_peak != -1 and t_peak != -1 and q_peak < t_peak:
                qt_interval = (t_peak - q_peak) * self.dt * 1000  # Convert to ms
                if 300 <= qt_interval <= 500:  # Normal QT interval range
                    qt_intervals.append(qt_interval)
        
        return float(np.mean(qt_intervals)) if qt_intervals else 0.0
    
    def calculate_qtc_interval(self, qt_interval: float, heart_rate: float) -> float:
        """
        Calculate corrected QT interval using Bazett's formula
        
        Args:
            qt_interval (float): QT interval in ms
            heart_rate (float): Heart rate in BPM
            
        Returns:
            float: Corrected QT interval in ms
        """
        if heart_rate <= 0:
            return 0.0
        
        rr_interval = 60.0 / heart_rate  # RR interval in seconds
        qtc = qt_interval / np.sqrt(rr_interval)
        return qtc
    
    def calculate_hrv_rmssd(self, r_peaks: np.ndarray) -> float:
        """
        Calculate HRV RMSSD (Root Mean Square of Successive Differences)
        
        Args:
            r_peaks (np.ndarray): R-peak indices
            
        Returns:
            float: RMSSD in milliseconds
        """
        if len(r_peaks) < 3:
            return 0.0
        
        # Calculate RR intervals
        rr_intervals = np.diff(r_peaks) * self.dt * 1000  # Convert to ms
        
        # Remove outliers
        valid_rr = rr_intervals[(rr_intervals > 300) & (rr_intervals < 2000)]
        
        if len(valid_rr) < 2:
            return 0.0
        
        # Calculate successive differences
        successive_diff = np.diff(valid_rr)
        
        # Calculate RMSSD
        rmssd = np.sqrt(np.mean(successive_diff ** 2))
        
        return rmssd
    
    def analyze_ecg(self, filepath: str) -> Dict:
        """
        Complete ECG analysis
        
        Args:
            filepath (str): Path to ECG data file
            
        Returns:
            Dict: Analysis results
        """
        print(f"Analyzing: {os.path.basename(filepath)}")
        
        # Load data
        time, ecg = self.load_ecg_data(filepath)
        
        # Preprocess
        filtered_ecg = self.preprocess_ecg(ecg)
        
        # Detect R-peaks
        r_peaks = self.detect_r_peaks(filtered_ecg)
        
        if len(r_peaks) < 2:
            return {
                'filename': os.path.basename(filepath),
                'heart_rate': 0.0,
                'pr_interval': 0.0,
                'qt_interval': 0.0,
                'qtc_interval': 0.0,
                'hrv_rmssd': 0.0,
                'num_beats': 0,
                'error': 'Insufficient R-peaks detected'
            }
        
        # Calculate heart rate
        heart_rate = self.calculate_heart_rate(r_peaks, time)
        
        # Detect other waves
        p_peaks = self.detect_p_waves(filtered_ecg, r_peaks)
        q_peaks = self.detect_q_waves(filtered_ecg, r_peaks)
        t_peaks = self.detect_t_waves(filtered_ecg, r_peaks)
        
        # Calculate intervals
        pr_interval = self.calculate_pr_interval(p_peaks, r_peaks)
        qt_interval = self.calculate_qt_interval(q_peaks, t_peaks)
        qtc_interval = self.calculate_qtc_interval(qt_interval, heart_rate)
        hrv_rmssd = self.calculate_hrv_rmssd(r_peaks)
        
        return {
            'filename': os.path.basename(filepath),
            'heart_rate': round(heart_rate, 1),
            'pr_interval': round(pr_interval, 1),
            'qt_interval': round(qt_interval, 1),
            'qtc_interval': round(qtc_interval, 1),
            'hrv_rmssd': round(hrv_rmssd, 1),
            'num_beats': len(r_peaks),
            'error': None
        }
    
    def analyze_all_datasets(self, datasets_folder: str = "./datasets") -> pd.DataFrame:
        """
        Analyze all ECG datasets in a folder
        
        Args:
            datasets_folder (str): Path to datasets folder
            
        Returns:
            pd.DataFrame: Analysis results for all datasets
        """
        # Find all CSV files
        csv_files = glob.glob(os.path.join(datasets_folder, "*.csv"))
        
        if not csv_files:
            print(f"No CSV files found in {datasets_folder}")
            return pd.DataFrame()
        
        results = []
        
        for filepath in csv_files:
            try:
                result = self.analyze_ecg(filepath)
                results.append(result)
            except Exception as e:
                print(f"Error analyzing {filepath}: {e}")
                results.append({
                    'filename': os.path.basename(filepath),
                    'heart_rate': 0.0,
                    'pr_interval': 0.0,
                    'qt_interval': 0.0,
                    'qtc_interval': 0.0,
                    'hrv_rmssd': 0.0,
                    'num_beats': 0,
                    'error': str(e)
                })
        
        return pd.DataFrame(results)
    
    def plot_analysis(self, filepath: str, save_path: Optional[str] = None):
        """
        Plot ECG analysis with detected waves
        
        Args:
            filepath (str): Path to ECG data file
            save_path (str): Path to save plot (optional)
        """
        # Load and preprocess data
        time, ecg = self.load_ecg_data(filepath)
        filtered_ecg = self.preprocess_ecg(ecg)
        
        # Detect waves
        r_peaks = self.detect_r_peaks(filtered_ecg)
        p_peaks = self.detect_p_waves(filtered_ecg, r_peaks)
        q_peaks = self.detect_q_waves(filtered_ecg, r_peaks)
        t_peaks = self.detect_t_waves(filtered_ecg, r_peaks)
        
        # Create plot
        plt.figure(figsize=(15, 8))
        
        # Plot ECG signal
        plt.plot(time, filtered_ecg, 'b-', linewidth=1, label='ECG Signal')
        
        # Plot detected waves
        if len(r_peaks) > 0:
            plt.plot(time[r_peaks], filtered_ecg[r_peaks], 'ro', markersize=8, label='R-peaks')
        
        # Plot P-waves (valid ones only)
        valid_p_peaks = [p for p in p_peaks if p != -1]
        if valid_p_peaks:
            plt.plot(time[valid_p_peaks], filtered_ecg[valid_p_peaks], 'go', markersize=6, label='P-waves')
        
        # Plot Q-waves (valid ones only)
        valid_q_peaks = [q for q in q_peaks if q != -1]
        if valid_q_peaks:
            plt.plot(time[valid_q_peaks], filtered_ecg[valid_q_peaks], 'mo', markersize=6, label='Q-waves')
        
        # Plot T-waves (valid ones only)
        valid_t_peaks = [t for t in t_peaks if t != -1]
        if valid_t_peaks:
            plt.plot(time[valid_t_peaks], filtered_ecg[valid_t_peaks], 'co', markersize=6, label='T-waves')
        
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude (mV)')
        plt.title(f'ECG Analysis: {os.path.basename(filepath)}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

def main():
    """
    Main function to analyze all datasets
    """
    print("ECG Analysis Pipeline")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = ECGAnalyzer(sampling_rate=500)
    
    # Analyze all datasets
    results_df = analyzer.analyze_all_datasets("./datasets")
    
    if not results_df.empty:
        print("\nAnalysis Results:")
        print("=" * 50)
        print(results_df.to_string(index=False))
        
        # Save results
        results_df.to_csv("ecg_analysis_results.csv", index=False)
        print(f"\nResults saved to: ecg_analysis_results.csv")
        
        # Summary statistics
        print("\nSummary Statistics:")
        print("=" * 50)
        print(f"Average Heart Rate: {results_df['heart_rate'].mean():.1f} BPM")
        print(f"Average PR Interval: {results_df['pr_interval'].mean():.1f} ms")
        print(f"Average QTc Interval: {results_df['qtc_interval'].mean():.1f} ms")
        print(f"Average HRV RMSSD: {results_df['hrv_rmssd'].mean():.1f} ms")
        
        # Plot analysis for first dataset
        csv_files = glob.glob("./datasets/*.csv")
        if csv_files:
            print(f"\nGenerating analysis plot for: {os.path.basename(csv_files[0])}")
            analyzer.plot_analysis(csv_files[0])
    
    else:
        print("No datasets found or analysis failed.")

if __name__ == "__main__":
    main() 