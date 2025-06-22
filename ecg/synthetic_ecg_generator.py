#!/usr/bin/env python3
"""
Synthetic ECG Generator
Generates realistic ECG signals using mathematical and physiological models
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import norm
import pandas as pd
from typing import Tuple, List, Dict, Optional
import random

class SyntheticECGGenerator:
    """
    A comprehensive synthetic ECG generator with multiple generation methods
    """
    
    def __init__(self, sampling_rate: int = 500, duration: float = 10.0):
        """
        Initialize the ECG generator
        
        Args:
            sampling_rate (int): Sampling rate in Hz
            duration (float): Duration of signal in seconds
        """
        self.sampling_rate = sampling_rate
        self.duration = duration
        self.time = np.arange(0, duration, 1/sampling_rate)
        self.n_samples = len(self.time)
        
        # Default ECG parameters (normal sinus rhythm)
        self.default_params = {
            'heart_rate': 72,  # BPM
            'p_amplitude': 0.25,  # mV
            'qrs_amplitude': 1.0,  # mV
            't_amplitude': 0.35,  # mV
            'p_width': 0.08,  # seconds
            'qrs_width': 0.08,  # seconds
            't_width': 0.16,  # seconds
            'pr_interval': 0.16,  # seconds
            'qt_interval': 0.40,  # seconds
            'baseline': 0.0,  # mV
        }
    
    def generate_simple_mathematical_ecg(self, heart_rate: int = 72, 
                                       noise_level: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate ECG using simple mathematical model with sine waves and Gaussian functions
        
        Args:
            heart_rate (int): Heart rate in BPM
            noise_level (float): Level of noise to add (0-1)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Time array and ECG signal
        """
        # Calculate RR interval
        rr_interval = 60.0 / heart_rate
        
        # Generate R-peak positions
        r_peaks = np.arange(0, self.duration, rr_interval)
        
        # Initialize signal
        ecg_signal = np.zeros_like(self.time)
        
        # Add each heartbeat
        for r_peak in r_peaks:
            if r_peak < self.duration:
                # P wave (Gaussian)
                p_peak = r_peak - 0.16  # PR interval
                p_wave = 0.25 * np.exp(-((self.time - p_peak) ** 2) / (2 * 0.02 ** 2))
                
                # QRS complex (combination of functions)
                qrs_complex = self._generate_qrs_complex(self.time - r_peak)
                
                # T wave (Gaussian)
                t_peak = r_peak + 0.20  # After QRS
                t_wave = 0.35 * np.exp(-((self.time - t_peak) ** 2) / (2 * 0.04 ** 2))
                
                # Combine waves
                heartbeat = p_wave + qrs_complex + t_wave
                ecg_signal += heartbeat
        
        # Add noise
        if noise_level > 0:
            noise = noise_level * np.random.normal(0, 1, len(ecg_signal))
            ecg_signal += noise
        
        return self.time, ecg_signal
    
    def _generate_qrs_complex(self, t: np.ndarray) -> np.ndarray:
        """
        Generate QRS complex using mathematical functions
        
        Args:
            t (np.ndarray): Time array relative to R peak
            
        Returns:
            np.ndarray: QRS complex signal
        """
        qrs = np.zeros_like(t)
        
        # Q wave (negative deflection)
        q_mask = (t >= -0.04) & (t < -0.02)
        qrs[q_mask] = -0.1 * np.sin(np.pi * (t[q_mask] + 0.04) / 0.02)
        
        # R wave (positive peak)
        r_mask = (t >= -0.02) & (t < 0.02)
        qrs[r_mask] = 1.0 * np.exp(-((t[r_mask] - 0) ** 2) / (2 * 0.01 ** 2))
        
        # S wave (negative deflection)
        s_mask = (t >= 0.02) & (t < 0.04)
        qrs[s_mask] = -0.2 * np.sin(np.pi * (t[s_mask] - 0.02) / 0.02)
        
        return qrs
    
    def generate_physiological_ecg(self, params: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate ECG using physiological model based on cardiac action potentials
        
        Args:
            params (Dict): ECG parameters (uses defaults if None)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Time array and ECG signal
        """
        if params is None:
            params = {}
        
        # Ensure all required parameters are present
        full_params = self.default_params.copy()
        full_params.update(params)
        
        # Calculate timing
        rr_interval = 60.0 / full_params['heart_rate']
        r_peaks = np.arange(0, self.duration, rr_interval)
        
        # Initialize signal
        ecg_signal = np.zeros_like(self.time)
        
        for r_peak in r_peaks:
            if r_peak < self.duration:
                # Generate individual waves with physiological timing
                heartbeat = self._generate_physiological_heartbeat(
                    self.time - r_peak, full_params
                )
                ecg_signal += heartbeat
        
        return self.time, ecg_signal
    
    def _generate_physiological_heartbeat(self, t: np.ndarray, params: Dict) -> np.ndarray:
        """
        Generate a single heartbeat using physiological model
        
        Args:
            t (np.ndarray): Time array relative to R peak
            params (Dict): ECG parameters
            
        Returns:
            np.ndarray: Single heartbeat signal
        """
        heartbeat = np.zeros_like(t)
        
        # P wave (atrial depolarization)
        p_peak = -params['pr_interval']
        p_wave = params['p_amplitude'] * np.exp(-((t - p_peak) ** 2) / (2 * (params['p_width']/4) ** 2))
        
        # QRS complex (ventricular depolarization)
        qrs = self._generate_physiological_qrs(t, params)
        
        # T wave (ventricular repolarization)
        t_peak = params['qt_interval'] - params['qrs_width']/2
        t_wave = params['t_amplitude'] * np.exp(-((t - t_peak) ** 2) / (2 * (params['t_width']/4) ** 2))
        
        # Combine waves
        heartbeat = p_wave + qrs + t_wave + params['baseline']
        
        return heartbeat
    
    def _generate_physiological_qrs(self, t: np.ndarray, params: Dict) -> np.ndarray:
        """
        Generate physiological QRS complex
        
        Args:
            t (np.ndarray): Time array relative to R peak
            params (Dict): ECG parameters
            
        Returns:
            np.ndarray: QRS complex
        """
        qrs = np.zeros_like(t)
        width = params['qrs_width']
        
        # Q wave
        q_start = -width/2
        q_end = -width/6
        q_mask = (t >= q_start) & (t < q_end)
        if np.any(q_mask):
            qrs[q_mask] = -0.1 * params['qrs_amplitude'] * np.sin(np.pi * (t[q_mask] - q_start) / (q_end - q_start))
        
        # R wave
        r_start = -width/6
        r_end = width/6
        r_mask = (t >= r_start) & (t < r_end)
        if np.any(r_mask):
            qrs[r_mask] = params['qrs_amplitude'] * np.exp(-((t[r_mask] - 0) ** 2) / (2 * (width/12) ** 2))
        
        # S wave
        s_start = width/6
        s_end = width/2
        s_mask = (t >= s_start) & (t < s_end)
        if np.any(s_mask):
            qrs[s_mask] = -0.2 * params['qrs_amplitude'] * np.sin(np.pi * (t[s_mask] - s_start) / (s_end - s_start))
        
        return qrs
    
    def add_realistic_noise(self, ecg_signal: np.ndarray, 
                          noise_types: List[str] = ['baseline_wander', 'powerline', 'muscle']) -> np.ndarray:
        """
        Add realistic noise to ECG signal
        
        Args:
            ecg_signal (np.ndarray): Clean ECG signal
            noise_types (List[str]): Types of noise to add
            
        Returns:
            np.ndarray: Noisy ECG signal
        """
        noisy_signal = ecg_signal.copy()
        
        for noise_type in noise_types:
            if noise_type == 'baseline_wander':
                # Low frequency baseline wander
                baseline_wander = 0.1 * np.sin(2 * np.pi * 0.5 * self.time) + \
                                0.05 * np.sin(2 * np.pi * 0.1 * self.time)
                noisy_signal += baseline_wander
            
            elif noise_type == 'powerline':
                # 50/60 Hz powerline interference
                powerline = 0.05 * np.sin(2 * np.pi * 50 * self.time) + \
                          0.03 * np.sin(2 * np.pi * 60 * self.time)
                noisy_signal += powerline
            
            elif noise_type == 'muscle':
                # High frequency muscle noise
                muscle_noise = 0.02 * np.random.normal(0, 1, len(ecg_signal))
                # Simple high-pass filtering using difference
                muscle_noise = np.diff(muscle_noise, prepend=muscle_noise[0])
                noisy_signal += muscle_noise
        
        return noisy_signal
    
    def generate_abnormal_rhythms(self, rhythm_type: str = 'tachycardia', 
                                params: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate abnormal ECG rhythms
        
        Args:
            rhythm_type (str): Type of abnormal rhythm
            params (Dict): Additional parameters
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Time array and ECG signal
        """
        if params is None:
            params = {}
        
        if rhythm_type == 'tachycardia':
            # Fast heart rate (>100 BPM)
            heart_rate = params.get('heart_rate', 120)
            return self.generate_physiological_ecg({'heart_rate': heart_rate})
        
        elif rhythm_type == 'bradycardia':
            # Slow heart rate (<60 BPM)
            heart_rate = params.get('heart_rate', 45)
            return self.generate_physiological_ecg({'heart_rate': heart_rate})
        
        elif rhythm_type == 'arrhythmia':
            # Irregular heart rate
            return self._generate_arrhythmia_ecg(params)
        
        elif rhythm_type == 'afib':
            # Atrial fibrillation
            return self._generate_afib_ecg(params)
        
        else:
            raise ValueError(f"Unknown rhythm type: {rhythm_type}")
    
    def _generate_arrhythmia_ecg(self, params: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate arrhythmic ECG with irregular intervals
        """
        base_heart_rate = params.get('heart_rate', 72)
        variability = params.get('variability', 0.3)  # 30% variability
        
        # Generate irregular RR intervals
        rr_intervals = []
        current_time = 0
        
        while current_time < self.duration:
            # Add random variation to RR interval
            rr = 60.0 / base_heart_rate
            rr_variation = rr * (1 + np.random.normal(0, variability))
            rr_intervals.append(rr_variation)
            current_time += rr_variation
        
        # Generate ECG with irregular intervals
        ecg_signal = np.zeros_like(self.time)
        current_time = 0
        
        for rr in rr_intervals:
            if current_time < self.duration:
                heartbeat = self._generate_physiological_heartbeat(
                    self.time - current_time, self.default_params
                )
                ecg_signal += heartbeat
                current_time += rr
        
        return self.time, ecg_signal
    
    def _generate_afib_ecg(self, params: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate atrial fibrillation ECG
        """
        # AFib has irregular ventricular response and absent P waves
        base_heart_rate = params.get('heart_rate', 100)
        variability = params.get('variability', 0.5)  # High variability
        
        # Generate irregular RR intervals (more variable than regular arrhythmia)
        rr_intervals = []
        current_time = 0
        
        while current_time < self.duration:
            rr = 60.0 / base_heart_rate
            rr_variation = rr * (1 + np.random.normal(0, variability))
            rr_intervals.append(rr_variation)
            current_time += rr_variation
        
        # Generate ECG without P waves
        ecg_signal = np.zeros_like(self.time)
        current_time = 0
        
        for rr in rr_intervals:
            if current_time < self.duration:
                # Generate heartbeat without P wave
                t = self.time - current_time
                heartbeat = np.zeros_like(t)
                
                # QRS complex
                qrs = self._generate_physiological_qrs(t, self.default_params)
                
                # T wave
                t_peak = self.default_params['qt_interval'] - self.default_params['qrs_width']/2
                t_wave = self.default_params['t_amplitude'] * np.exp(-((t - t_peak) ** 2) / (2 * (self.default_params['t_width']/4) ** 2))
                
                heartbeat = qrs + t_wave + self.default_params['baseline']
                ecg_signal += heartbeat
                current_time += rr
        
        return self.time, ecg_signal
    
    def plot_ecg(self, time: np.ndarray, ecg_signal: np.ndarray, 
                title: str = "Synthetic ECG Signal", save_path: Optional[str] = None):
        """
        Plot ECG signal
        
        Args:
            time (np.ndarray): Time array
            ecg_signal (np.ndarray): ECG signal
            title (str): Plot title
            save_path (str): Path to save plot (optional)
        """
        plt.figure(figsize=(15, 8))
        plt.plot(time, ecg_signal, 'b-', linewidth=1)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude (mV)')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_ecg_data(self, time: np.ndarray, ecg_signal: np.ndarray, 
                     filename: str = "synthetic_ecg.csv"):
        """
        Save ECG data to CSV file
        
        Args:
            time (np.ndarray): Time array
            ecg_signal (np.ndarray): ECG signal
            filename (str): Output filename
        """
        df = pd.DataFrame({
            'Time': time,
            'ECG': ecg_signal
        })
        df.to_csv(filename, index=False)
        print(f"ECG data saved to {filename}")

def main():
    """
    Example usage of the synthetic ECG generator
    """
    print("Synthetic ECG Generator Demo")
    print("=" * 40)
    
    # Initialize generator
    generator = SyntheticECGGenerator(sampling_rate=500, duration=10.0)
    
    # Generate different types of ECG signals
    print("1. Generating normal sinus rhythm...")
    time, normal_ecg = generator.generate_physiological_ecg()
    normal_ecg = generator.add_realistic_noise(normal_ecg)
    generator.plot_ecg(time, normal_ecg, "Normal Sinus Rhythm")
    generator.save_ecg_data(time, normal_ecg, "normal_ecg.csv")
    
    print("2. Generating tachycardia...")
    time, tachy_ecg = generator.generate_abnormal_rhythms('tachycardia', {'heart_rate': 120})
    tachy_ecg = generator.add_realistic_noise(tachy_ecg)
    generator.plot_ecg(time, tachy_ecg, "Tachycardia")
    generator.save_ecg_data(time, tachy_ecg, "tachycardia_ecg.csv")
    
    print("3. Generating atrial fibrillation...")
    time, afib_ecg = generator.generate_abnormal_rhythms('afib', {'heart_rate': 100})
    afib_ecg = generator.add_realistic_noise(afib_ecg)
    generator.plot_ecg(time, afib_ecg, "Atrial Fibrillation")
    generator.save_ecg_data(time, afib_ecg, "afib_ecg.csv")
    
    print("4. Generating simple mathematical model...")
    time, simple_ecg = generator.generate_simple_mathematical_ecg(heart_rate=80, noise_level=0.1)
    generator.plot_ecg(time, simple_ecg, "Simple Mathematical Model")
    generator.save_ecg_data(time, simple_ecg, "simple_ecg.csv")
    
    print("Demo completed!")

if __name__ == "__main__":
    main() 