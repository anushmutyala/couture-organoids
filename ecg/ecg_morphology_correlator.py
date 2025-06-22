#!/usr/bin/env python3
"""
ECG Morphology Correlator
Compares ECG recordings based on individual spike morphology
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr, spearmanr
from scipy.integrate import trapz
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple, Optional, NamedTuple
import os

class HeartbeatMorphology(NamedTuple):
    """Structure to hold heartbeat morphology data"""
    r_peak_idx: int
    start_idx: int
    end_idx: int
    signal: np.ndarray
    time_window: np.ndarray
    features: Dict[str, float]

class ECGMorphologyCorrelator:
    """
    ECG morphology correlation analyzer
    Compares individual heartbeat shapes and characteristics
    """
    
    def __init__(self, sampling_rate: int = 500, window_size: float = 0.8):
        """
        Initialize morphology correlator
        
        Args:
            sampling_rate (int): Sampling rate in Hz
            window_size (float): Window size around R-peak in seconds
        """
        self.sampling_rate = sampling_rate
        self.dt = 1.0 / sampling_rate
        self.window_size = window_size
        self.window_samples = int(window_size * sampling_rate)
        
    def load_ecg_data(self, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load ECG data from CSV file"""
        df = pd.read_csv(filepath)
        time = np.array(df['Time'].values)
        ecg = np.array(df['ECG'].values)
        return time, ecg
    
    def preprocess_ecg(self, ecg: np.ndarray) -> np.ndarray:
        """Preprocess ECG signal"""
        # Remove baseline wander
        result = signal.butter(3, 0.5, 'high', fs=self.sampling_rate)
        b, a = result[0], result[1]
        filtered_ecg = signal.filtfilt(b, a, ecg)
        
        # Remove powerline interference
        result = signal.butter(4, [45, 65], 'bandstop', fs=self.sampling_rate)
        b, a = result[0], result[1]
        filtered_ecg = signal.filtfilt(b, a, filtered_ecg)
        
        return filtered_ecg
    
    def detect_r_peaks(self, ecg: np.ndarray, prominence: float = 0.5) -> np.ndarray:
        """Detect R-peaks in ECG signal"""
        peaks, _ = find_peaks(ecg, prominence=prominence, distance=int(0.3 * self.sampling_rate))
        return peaks
    
    def extract_heartbeat_windows(self, ecg: np.ndarray, r_peaks: np.ndarray) -> List[HeartbeatMorphology]:
        """
        Extract individual heartbeat windows around R-peaks
        
        Args:
            ecg (np.ndarray): ECG signal
            r_peaks (np.ndarray): R-peak indices
            
        Returns:
            List[HeartbeatMorphology]: List of heartbeat morphology data
        """
        heartbeats = []
        
        for r_peak in r_peaks:
            # Define window around R-peak
            half_window = self.window_samples // 2
            start_idx = max(0, r_peak - half_window)
            end_idx = min(len(ecg), r_peak + half_window)
            
            # Extract signal window
            signal_window = ecg[start_idx:end_idx]
            time_window = np.arange(len(signal_window)) * self.dt
            
            # Normalize signal to same length
            if len(signal_window) < self.window_samples:
                # Pad with zeros if too short
                padding = self.window_samples - len(signal_window)
                signal_window = np.pad(signal_window, (0, padding), 'constant')
                time_window = np.arange(self.window_samples) * self.dt
            elif len(signal_window) > self.window_samples:
                # Truncate if too long
                signal_window = signal_window[:self.window_samples]
                time_window = np.arange(self.window_samples) * self.dt
            
            # Extract morphological features
            features = self.extract_morphological_features(signal_window, time_window)
            
            heartbeat = HeartbeatMorphology(
                r_peak_idx=r_peak,
                start_idx=start_idx,
                end_idx=end_idx,
                signal=signal_window,
                time_window=time_window,
                features=features
            )
            heartbeats.append(heartbeat)
        
        return heartbeats
    
    def extract_morphological_features(self, signal_data: np.ndarray, time: np.ndarray) -> Dict[str, float]:
        """
        Extract morphological features from heartbeat signal
        
        Args:
            signal_data (np.ndarray): Heartbeat signal window
            time (np.ndarray): Time array for the window
            
        Returns:
            Dict[str, float]: Dictionary of morphological features
        """
        features = {}
        
        # Basic statistical features
        features['mean'] = float(np.mean(signal_data))
        features['std'] = float(np.std(signal_data))
        features['max'] = float(np.max(signal_data))
        features['min'] = float(np.min(signal_data))
        features['range'] = features['max'] - features['min']
        features['rms'] = float(np.sqrt(np.mean(signal_data**2)))
        
        # Peak features
        peaks, _ = find_peaks(signal_data, prominence=0.1)
        valleys, _ = find_peaks(-signal_data, prominence=0.1)
        
        features['num_peaks'] = len(peaks)
        features['num_valleys'] = len(valleys)
        
        if len(peaks) > 0:
            features['peak_heights'] = float(np.mean(signal_data[peaks]))
            features['peak_positions'] = float(np.mean(peaks) / len(signal_data))
        else:
            features['peak_heights'] = 0.0
            features['peak_positions'] = 0.0
        
        if len(valleys) > 0:
            features['valley_depths'] = float(np.mean(signal_data[valleys]))
            features['valley_positions'] = float(np.mean(valleys) / len(signal_data))
        else:
            features['valley_depths'] = 0.0
            features['valley_positions'] = 0.0
        
        # QRS complex features (assuming R-peak is at center)
        center_idx = len(signal_data) // 2
        qrs_start = max(0, center_idx - int(0.04 * self.sampling_rate))
        qrs_end = min(len(signal_data), center_idx + int(0.04 * self.sampling_rate))
        qrs_signal = signal_data[qrs_start:qrs_end]
        
        if len(qrs_signal) > 0:
            features['qrs_width'] = float(len(qrs_signal) * self.dt * 1000)  # ms
            features['qrs_amplitude'] = float(np.max(qrs_signal) - np.min(qrs_signal))
            features['qrs_area'] = float(trapz(np.abs(qrs_signal), time[qrs_start:qrs_end]))
        else:
            features['qrs_width'] = 0.0
            features['qrs_amplitude'] = 0.0
            features['qrs_area'] = 0.0
        
        # Waveform complexity features
        # First derivative (rate of change)
        derivative = np.diff(signal_data)
        features['mean_derivative'] = float(np.mean(np.abs(derivative)))
        features['max_derivative'] = float(np.max(np.abs(derivative)))
        
        # Second derivative (acceleration)
        second_derivative = np.diff(derivative)
        if len(second_derivative) > 0:
            features['mean_acceleration'] = float(np.mean(np.abs(second_derivative)))
            features['max_acceleration'] = float(np.max(np.abs(second_derivative)))
        else:
            features['mean_acceleration'] = 0.0
            features['max_acceleration'] = 0.0
        
        # Spectral features
        if len(signal_data) > 1:
            freqs, psd = signal.welch(signal_data, fs=self.sampling_rate, nperseg=min(256, len(signal_data)//2))
            features['dominant_freq'] = float(freqs[np.argmax(psd)])
            features['spectral_energy'] = float(np.sum(psd))
            features['spectral_centroid'] = float(np.sum(freqs * psd) / np.sum(psd)) if np.sum(psd) > 0 else 0.0
        else:
            features['dominant_freq'] = 0.0
            features['spectral_energy'] = 0.0
            features['spectral_centroid'] = 0.0
        
        # Shape features
        # Symmetry around center
        center = len(signal_data) // 2
        left_half = signal_data[:center]
        right_half = signal_data[center:][::-1]  # Reverse for comparison
        
        if len(left_half) == len(right_half) and len(left_half) > 0:
            symmetry_corr, _ = pearsonr(left_half, right_half)
            features['symmetry'] = float(symmetry_corr) if not np.isnan(symmetry_corr) else 0.0
        else:
            features['symmetry'] = 0.0
        
        # Kurtosis and skewness
        features['kurtosis'] = float(self._calculate_kurtosis(signal_data))
        features['skewness'] = float(self._calculate_skewness(signal_data))
        
        return features
    
    def _calculate_kurtosis(self, signal: np.ndarray) -> float:
        """Calculate kurtosis of signal"""
        if len(signal) < 4:
            return 0.0
        mean = np.mean(signal)
        std = np.std(signal)
        if std == 0:
            return 0.0
        kurtosis = np.mean(((signal - mean) / std) ** 4) - 3
        return kurtosis
    
    def _calculate_skewness(self, signal: np.ndarray) -> float:
        """Calculate skewness of signal"""
        if len(signal) < 3:
            return 0.0
        mean = np.mean(signal)
        std = np.std(signal)
        if std == 0:
            return 0.0
        skewness = np.mean(((signal - mean) / std) ** 3)
        return skewness
    
    def calculate_morphology_similarity(self, heartbeat1: HeartbeatMorphology, 
                                      heartbeat2: HeartbeatMorphology) -> Dict[str, float]:
        """
        Calculate similarity between two heartbeats
        
        Args:
            heartbeat1 (HeartbeatMorphology): First heartbeat
            heartbeat2 (HeartbeatMorphology): Second heartbeat
            
        Returns:
            Dict[str, float]: Similarity scores
        """
        similarities = {}
        
        # Signal correlation
        if len(heartbeat1.signal) == len(heartbeat2.signal):
            # Pearson correlation
            corr, _ = pearsonr(heartbeat1.signal, heartbeat2.signal)
            similarities['signal_correlation'] = float(corr) if not np.isnan(corr) else 0.0
            
            # Spearman correlation
            spearman_corr, _ = spearmanr(heartbeat1.signal, heartbeat2.signal)
            similarities['signal_spearman'] = float(spearman_corr) if not np.isnan(spearman_corr) else 0.0
            
            # Cosine similarity
            cos_sim = cosine_similarity(heartbeat1.signal.reshape(1, -1), 
                                      heartbeat2.signal.reshape(1, -1))[0, 0]
            similarities['cosine_similarity'] = float(cos_sim)
            
            # Euclidean distance (normalized)
            euclidean_dist = np.linalg.norm(heartbeat1.signal - heartbeat2.signal)
            max_possible_dist = np.sqrt(len(heartbeat1.signal)) * (np.max(heartbeat1.signal) - np.min(heartbeat1.signal))
            similarities['euclidean_similarity'] = float(1.0 - (euclidean_dist / max_possible_dist)) if max_possible_dist > 0 else 0.0
        else:
            similarities['signal_correlation'] = 0.0
            similarities['signal_spearman'] = 0.0
            similarities['cosine_similarity'] = 0.0
            similarities['euclidean_similarity'] = 0.0
        
        # Feature-based similarity
        feature_similarities = []
        for key in heartbeat1.features:
            if key in heartbeat2.features:
                val1 = heartbeat1.features[key]
                val2 = heartbeat2.features[key]
                
                # Normalize feature values
                max_val = max(abs(val1), abs(val2))
                if max_val > 0:
                    normalized_diff = abs(val1 - val2) / max_val
                    feature_similarities.append(1.0 - normalized_diff)
                else:
                    feature_similarities.append(1.0)  # Both zero
        
        similarities['feature_similarity'] = float(np.mean(feature_similarities)) if feature_similarities else 0.0
        
        # Overall similarity (weighted average)
        weights = {
            'signal_correlation': 0.4,
            'cosine_similarity': 0.3,
            'feature_similarity': 0.3
        }
        
        overall_score = sum(similarities[key] * weights[key] for key in weights)
        similarities['overall_similarity'] = float(overall_score)
        
        return similarities
    
    def compare_ecg_recordings(self, filepath1: str, filepath2: str) -> Dict:
        """
        Compare two ECG recordings based on morphology
        
        Args:
            filepath1 (str): Path to first ECG file
            filepath2 (str): Path to second ECG file
            
        Returns:
            Dict: Comparison results
        """
        print(f"Comparing: {os.path.basename(filepath1)} vs {os.path.basename(filepath2)}")
        
        # Load and preprocess data
        time1, ecg1 = self.load_ecg_data(filepath1)
        time2, ecg2 = self.load_ecg_data(filepath2)
        
        filtered_ecg1 = self.preprocess_ecg(ecg1)
        filtered_ecg2 = self.preprocess_ecg(ecg2)
        
        # Detect R-peaks
        r_peaks1 = self.detect_r_peaks(filtered_ecg1)
        r_peaks2 = self.detect_r_peaks(filtered_ecg2)
        
        if len(r_peaks1) < 2 or len(r_peaks2) < 2:
            return {
                'error': 'Insufficient R-peaks detected in one or both recordings',
                'similarity_scores': {},
                'heartbeat_comparisons': []
            }
        
        # Extract heartbeat windows
        heartbeats1 = self.extract_heartbeat_windows(filtered_ecg1, r_peaks1)
        heartbeats2 = self.extract_heartbeat_windows(filtered_ecg2, r_peaks2)
        
        print(f"Extracted {len(heartbeats1)} heartbeats from recording 1")
        print(f"Extracted {len(heartbeats2)} heartbeats from recording 2")
        
        # Compare heartbeats
        heartbeat_comparisons = []
        all_similarities = []
        
        # Compare each heartbeat from recording 1 with each from recording 2
        for i, hb1 in enumerate(heartbeats1):
            for j, hb2 in enumerate(heartbeats2):
                similarity = self.calculate_morphology_similarity(hb1, hb2)
                heartbeat_comparisons.append({
                    'recording1_beat': i,
                    'recording2_beat': j,
                    'similarity_scores': similarity
                })
                all_similarities.append(similarity)
        
        # Calculate aggregate similarity scores
        if all_similarities:
            aggregate_scores = {}
            for key in all_similarities[0].keys():
                values = [sim[key] for sim in all_similarities]
                aggregate_scores[f'mean_{key}'] = float(np.mean(values))
                aggregate_scores[f'std_{key}'] = float(np.std(values))
                aggregate_scores[f'max_{key}'] = float(np.max(values))
                aggregate_scores[f'min_{key}'] = float(np.min(values))
        else:
            aggregate_scores = {}
        
        return {
            'recording1': os.path.basename(filepath1),
            'recording2': os.path.basename(filepath2),
            'num_heartbeats1': len(heartbeats1),
            'num_heartbeats2': len(heartbeats2),
            'total_comparisons': len(heartbeat_comparisons),
            'similarity_scores': aggregate_scores,
            'heartbeat_comparisons': heartbeat_comparisons[:10],  # Keep first 10 for display
            'error': None
        }
    
    def plot_morphology_comparison(self, filepath1: str, filepath2: str, 
                                 save_path: Optional[str] = None):
        """
        Plot morphology comparison between two ECG recordings
        
        Args:
            filepath1 (str): Path to first ECG file
            filepath2 (str): Path to second ECG file
            save_path (str): Path to save plot (optional)
        """
        # Load and preprocess data
        time1, ecg1 = self.load_ecg_data(filepath1)
        time2, ecg2 = self.load_ecg_data(filepath2)
        
        filtered_ecg1 = self.preprocess_ecg(ecg1)
        filtered_ecg2 = self.preprocess_ecg(ecg2)
        
        # Detect R-peaks
        r_peaks1 = self.detect_r_peaks(filtered_ecg1)
        r_peaks2 = self.detect_r_peaks(filtered_ecg2)
        
        # Extract heartbeats
        heartbeats1 = self.extract_heartbeat_windows(filtered_ecg1, r_peaks1)
        heartbeats2 = self.extract_heartbeat_windows(filtered_ecg2, r_peaks2)
        
        # Create comprehensive comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('ECG Morphology Comparison Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Full ECG signals overlaid
        axes[0, 0].plot(time1, filtered_ecg1, 'b-', linewidth=1.5, alpha=0.8, 
                       label=f'{os.path.basename(filepath1)}', zorder=2)
        axes[0, 0].plot(time2, filtered_ecg2, 'r-', linewidth=1.5, alpha=0.8, 
                       label=f'{os.path.basename(filepath2)}', zorder=2)
        axes[0, 0].set_title('Full ECG Signals Overlay', fontweight='bold')
        axes[0, 0].set_ylabel('Amplitude (mV)')
        axes[0, 0].set_xlabel('Time (seconds)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Individual heartbeats overlaid with deviation area
        if len(heartbeats1) > 0 and len(heartbeats2) > 0:
            # Use the first heartbeat from each recording for comparison
            hb1 = heartbeats1[0]
            hb2 = heartbeats2[0]
            
            # Ensure both heartbeats have the same time axis
            time_axis = hb1.time_window * 1000  # Convert to ms
            
            # Plot both heartbeats
            axes[0, 1].plot(time_axis, hb1.signal, 'b-', linewidth=2, 
                           label=f'{os.path.basename(filepath1)} Beat 1', zorder=3)
            axes[0, 1].plot(time_axis, hb2.signal, 'r-', linewidth=2, 
                           label=f'{os.path.basename(filepath2)} Beat 1', zorder=3)
            
            # Fill area between curves to show deviation
            axes[0, 1].fill_between(time_axis, hb1.signal, hb2.signal, 
                                   alpha=0.3, color='purple', 
                                   label='Deviation Area', zorder=1)
            
            axes[0, 1].set_title('Individual Heartbeat Comparison\n(Deviation Area Highlighted)', 
                                fontweight='bold')
            axes[0, 1].set_xlabel('Time (ms)')
            axes[0, 1].set_ylabel('Amplitude (mV)')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Calculate and display deviation metrics
            deviation_area = np.trapz(np.abs(hb1.signal - hb2.signal), time_axis)
            correlation = np.corrcoef(hb1.signal, hb2.signal)[0, 1]
            axes[0, 1].text(0.02, 0.98, f'Deviation Area: {deviation_area:.2f} mV·ms\n'
                                        f'Correlation: {correlation:.3f}', 
                           transform=axes[0, 1].transAxes, fontsize=10,
                           verticalalignment='top', bbox=dict(boxstyle='round', 
                           facecolor='white', alpha=0.8))
        
        # Plot 3: Multiple heartbeats comparison (first 3 from each recording)
        if len(heartbeats1) > 0 and len(heartbeats2) > 0:
            colors1 = ['blue', 'navy', 'darkblue']
            colors2 = ['red', 'darkred', 'crimson']
            
            for i in range(min(3, len(heartbeats1))):
                hb = heartbeats1[i]
                time_axis = hb.time_window * 1000
                axes[1, 0].plot(time_axis, hb.signal, color=colors1[i], 
                               linewidth=1.5, alpha=0.7, 
                               label=f'Beat {i+1}' if i == 0 else "")
            
            for i in range(min(3, len(heartbeats2))):
                hb = heartbeats2[i]
                time_axis = hb.time_window * 1000
                axes[1, 0].plot(time_axis, hb.signal, color=colors2[i], 
                               linewidth=1.5, alpha=0.7, 
                               label=f'Beat {i+1}' if i == 0 else "")
            
            axes[1, 0].set_title('Multiple Heartbeats Comparison\n(First 3 from each recording)', 
                                fontweight='bold')
            axes[1, 0].set_xlabel('Time (ms)')
            axes[1, 0].set_ylabel('Amplitude (mV)')
            axes[1, 0].legend(['Recording 1', 'Recording 2'])
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Deviation analysis over time
        if len(heartbeats1) > 0 and len(heartbeats2) > 0:
            # Calculate deviation for each heartbeat pair
            deviations = []
            correlations = []
            
            for i in range(min(len(heartbeats1), len(heartbeats2))):
                hb1 = heartbeats1[i]
                hb2 = heartbeats2[i]
                
                # Ensure same length
                min_len = min(len(hb1.signal), len(hb2.signal))
                sig1 = hb1.signal[:min_len]
                sig2 = hb2.signal[:min_len]
                
                # Calculate deviation area
                time_axis = np.arange(min_len) * self.dt * 1000
                deviation = np.trapz(np.abs(sig1 - sig2), time_axis)
                deviations.append(deviation)
                
                # Calculate correlation
                corr = np.corrcoef(sig1, sig2)[0, 1]
                correlations.append(corr if not np.isnan(corr) else 0.0)
            
            # Plot deviation over time
            beat_numbers = range(1, len(deviations) + 1)
            axes[1, 1].plot(beat_numbers, deviations, 'o-', color='purple', 
                           linewidth=2, markersize=6, label='Deviation Area')
            axes[1, 1].set_title('Deviation Analysis Over Time', fontweight='bold')
            axes[1, 1].set_xlabel('Heartbeat Number')
            axes[1, 1].set_ylabel('Deviation Area (mV·ms)')
            axes[1, 1].grid(True, alpha=0.3)
            
            # Add correlation on secondary y-axis
            ax2 = axes[1, 1].twinx()
            ax2.plot(beat_numbers, correlations, 's-', color='orange', 
                    linewidth=2, markersize=6, label='Correlation')
            ax2.set_ylabel('Correlation Coefficient', color='orange')
            ax2.tick_params(axis='y', labelcolor='orange')
            
            # Add legends
            axes[1, 1].legend(loc='upper left')
            ax2.legend(loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

    def plot_simple_comparison(self, filepath1: str, filepath2: str, 
                             save_path: Optional[str] = None):
        """
        Simple comparison plot showing both recordings overlaid with deviation areas
        
        Args:
            filepath1 (str): Path to first ECG file
            filepath2 (str): Path to second ECG file
            save_path (str): Path to save plot (optional)
        """
        # Load and preprocess data
        time1, ecg1 = self.load_ecg_data(filepath1)
        time2, ecg2 = self.load_ecg_data(filepath2)
        
        filtered_ecg1 = self.preprocess_ecg(ecg1)
        filtered_ecg2 = self.preprocess_ecg(ecg2)
        
        # Detect R-peaks
        r_peaks1 = self.detect_r_peaks(filtered_ecg1)
        r_peaks2 = self.detect_r_peaks(filtered_ecg2)
        
        # Extract heartbeats
        heartbeats1 = self.extract_heartbeat_windows(filtered_ecg1, r_peaks1)
        heartbeats2 = self.extract_heartbeat_windows(filtered_ecg2, r_peaks2)
        
        # Create simple comparison plot
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        fig.suptitle('ECG Morphology Comparison - Deviation Analysis', 
                    fontsize=16, fontweight='bold')
        
        # Plot 1: Full signals overlaid with deviation areas
        axes[0].plot(time1, filtered_ecg1, 'b-', linewidth=2, alpha=0.8, 
                    label=f'{os.path.basename(filepath1)}', zorder=2)
        axes[0].plot(time2, filtered_ecg2, 'r-', linewidth=2, alpha=0.8, 
                    label=f'{os.path.basename(filepath2)}', zorder=2)
        
        # Fill area between curves to show deviation
        # Interpolate to same time points if needed
        if len(time1) != len(time2):
            from scipy.interpolate import interp1d
            f1 = interp1d(time1, filtered_ecg1, bounds_error=False, fill_value='extrapolate')
            f2 = interp1d(time2, filtered_ecg2, bounds_error=False, fill_value='extrapolate')
            time_common = np.linspace(max(time1[0], time2[0]), 
                                    min(time1[-1], time2[-1]), 
                                    min(len(time1), len(time2)))
            ecg1_interp = f1(time_common)
            ecg2_interp = f2(time_common)
            axes[0].fill_between(time_common, ecg1_interp, ecg2_interp, 
                               alpha=0.3, color='purple', 
                               label='Deviation Area', zorder=1)
        else:
            axes[0].fill_between(time1, filtered_ecg1, filtered_ecg2, 
                               alpha=0.3, color='purple', 
                               label='Deviation Area', zorder=1)
        
        axes[0].set_title('Full ECG Signals with Deviation Areas', fontweight='bold')
        axes[0].set_ylabel('Amplitude (mV)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Individual heartbeat comparison with deviation
        if len(heartbeats1) > 0 and len(heartbeats2) > 0:
            # Use the first heartbeat from each recording
            hb1 = heartbeats1[0]
            hb2 = heartbeats2[0]
            
            # Ensure both heartbeats have the same time axis
            time_axis = hb1.time_window * 1000  # Convert to ms
            
            # Plot both heartbeats
            axes[1].plot(time_axis, hb1.signal, 'b-', linewidth=3, 
                        label=f'{os.path.basename(filepath1)} Beat 1', zorder=3)
            axes[1].plot(time_axis, hb2.signal, 'r-', linewidth=3, 
                        label=f'{os.path.basename(filepath2)} Beat 1', zorder=3)
            
            # Fill area between curves to show deviation
            axes[1].fill_between(time_axis, hb1.signal, hb2.signal, 
                               alpha=0.4, color='purple', 
                               label='Deviation Area', zorder=1)
            
            # Calculate and display key metrics
            deviation_area = np.trapz(np.abs(hb1.signal - hb2.signal), time_axis)
            correlation = np.corrcoef(hb1.signal, hb2.signal)[0, 1]
            max_deviation = np.max(np.abs(hb1.signal - hb2.signal))
            
            # Add metrics text box
            metrics_text = (f'Deviation Area: {deviation_area:.2f} mV·ms\n'
                          f'Correlation: {correlation:.3f}\n'
                          f'Max Deviation: {max_deviation:.3f} mV')
            
            axes[1].text(0.02, 0.98, metrics_text, 
                        transform=axes[1].transAxes, fontsize=12,
                        verticalalignment='top', 
                        bbox=dict(boxstyle='round,pad=0.5', 
                                facecolor='white', alpha=0.9, 
                                edgecolor='black'))
            
            axes[1].set_title('Individual Heartbeat Comparison\n(Deviation Area = Purple Shaded Region)', 
                            fontweight='bold')
            axes[1].set_xlabel('Time (ms)')
            axes[1].set_ylabel('Amplitude (mV)')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Print summary statistics
        print(f"\n{'='*60}")
        print("DEVIATION ANALYSIS SUMMARY")
        print(f"{'='*60}")
        print(f"Recording 1: {os.path.basename(filepath1)}")
        print(f"Recording 2: {os.path.basename(filepath2)}")
        print(f"Heartbeats 1: {len(heartbeats1)}")
        print(f"Heartbeats 2: {len(heartbeats2)}")
        
        if len(heartbeats1) > 0 and len(heartbeats2) > 0:
            print(f"\nFirst Heartbeat Comparison:")
            print(f"  Deviation Area: {deviation_area:.2f} mV·ms")
            print(f"  Correlation: {correlation:.3f}")
            print(f"  Max Deviation: {max_deviation:.3f} mV")
            
            # Calculate average deviation across all heartbeats
            all_deviations = []
            all_correlations = []
            
            for i in range(min(len(heartbeats1), len(heartbeats2))):
                hb1 = heartbeats1[i]
                hb2 = heartbeats2[i]
                
                min_len = min(len(hb1.signal), len(hb2.signal))
                sig1 = hb1.signal[:min_len]
                sig2 = hb2.signal[:min_len]
                
                time_axis = np.arange(min_len) * self.dt * 1000
                deviation = np.trapz(np.abs(sig1 - sig2), time_axis)
                all_deviations.append(deviation)
                
                corr = np.corrcoef(sig1, sig2)[0, 1]
                all_correlations.append(corr if not np.isnan(corr) else 0.0)
            
            print(f"\nOverall Statistics (all heartbeats):")
            print(f"  Avg Deviation Area: {np.mean(all_deviations):.2f} mV·ms")
            print(f"  Avg Correlation: {np.mean(all_correlations):.3f}")
            print(f"  Std Deviation Area: {np.std(all_deviations):.2f} mV·ms")
            print(f"  Min Correlation: {np.min(all_correlations):.3f}")
            print(f"  Max Correlation: {np.max(all_correlations):.3f}")

def main():
    """
    Example usage of the ECG morphology correlator
    """
    print("ECG Morphology Correlation Analysis")
    print("=" * 50)
    
    # Initialize correlator
    correlator = ECGMorphologyCorrelator(sampling_rate=500, window_size=0.8)
    
    # Example comparison
    datasets_folder = "../datasets"
    csv_files = [f for f in os.listdir(datasets_folder) if f.endswith('.csv')]
    
    if len(csv_files) >= 2:
        file1 = os.path.join(datasets_folder, csv_files[0])
        file2 = os.path.join(datasets_folder, csv_files[1])
        
        # Compare recordings
        results = correlator.compare_ecg_recordings(file1, file2)
        
        if results.get('error'):
            print(f"Error: {results['error']}")
        else:
            print("\nComparison Results:")
            print("=" * 50)
            print(f"Recording 1: {results['recording1']}")
            print(f"Recording 2: {results['recording2']}")
            print(f"Heartbeats 1: {results['num_heartbeats1']}")
            print(f"Heartbeats 2: {results['num_heartbeats2']}")
            print(f"Total comparisons: {results['total_comparisons']}")
            
            print("\nAggregate Similarity Scores:")
            print("-" * 30)
            for key, value in results['similarity_scores'].items():
                print(f"{key}: {value:.3f}")
            
            # Plot comparison
            correlator.plot_morphology_comparison(file1, file2)
    else:
        print("Need at least 2 CSV files in the datasets folder for comparison.")

if __name__ == "__main__":
    main() 