#!/usr/bin/env python3
"""
ECG PID Control Loop Visualization
Shows chemical modulation (A: activator, B: inhibitor) to match normal ECG to afib patterns
Uses deviation area as error signal for PID control
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, FancyBboxPatch
import pandas as pd
import os
from scipy import signal
from scipy.integrate import trapz
from typing import Tuple, List, Dict, Optional, Any

class ECGPIDController:
    """
    PID controller for ECG signal matching using chemical modulation
    """
    
    def __init__(self, kp: float = 0.5, ki: float = 0.1, kd: float = 0.05):
        """
        Initialize PID controller
        
        Args:
            kp (float): Proportional gain
            ki (float): Integral gain  
            kd (float): Derivative gain
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        # PID state variables
        self.prev_error = 0.0
        self.integral = 0.0
        self.setpoint = 0.0  # Target deviation area (0 = perfect match)
        
        # Chemical concentration limits
        self.chem_a_min, self.chem_a_max = 0.0, 10.0  # Activator range
        self.chem_b_min, self.chem_b_max = 0.0, 10.0  # Inhibitor range
        
        # Current chemical concentrations
        self.chem_a = 5.0  # Initial activator concentration
        self.chem_b = 5.0  # Initial inhibitor concentration
        
        # History for plotting
        self.time_history: List[float] = []
        self.error_history: List[float] = []
        self.chem_a_history: List[float] = []
        self.chem_b_history: List[float] = []
        self.output_history: List[float] = []
        self.deviation_history: List[float] = []
        
    def calculate_pid_output(self, error: float, dt: float) -> Tuple[float, float]:
        """
        Calculate PID control output for chemical concentrations
        
        Args:
            error (float): Current deviation area error
            dt (float): Time step
            
        Returns:
            Tuple[float, float]: (chemical_a, chemical_b) concentrations
        """
        # PID calculations
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        
        # PID output
        pid_output = (self.kp * error + 
                     self.ki * self.integral + 
                     self.kd * derivative)
        
        # Update previous error
        self.prev_error = error
        
        # Map PID output to chemical concentrations
        # Positive output increases activator, decreases inhibitor
        # Negative output decreases activator, increases inhibitor
        
        # Chemical A (activator) - increases with positive PID output
        chem_a_delta = pid_output * 0.5
        self.chem_a = np.clip(self.chem_a + chem_a_delta, 
                             self.chem_a_min, self.chem_a_max)
        
        # Chemical B (inhibitor) - increases with negative PID output  
        chem_b_delta = -pid_output * 0.5
        self.chem_b = np.clip(self.chem_b + chem_b_delta,
                             self.chem_b_min, self.chem_b_max)
        
        return self.chem_a, self.chem_b
    
    def simulate_ecg_response(self, chem_a: float, chem_b: float, 
                            base_signal: np.ndarray) -> np.ndarray:
        """
        Simulate ECG response to chemical concentrations
        
        Args:
            chem_a (float): Activator concentration
            chem_b (float): Inhibitor concentration
            base_signal (np.ndarray): Base normal ECG signal
            
        Returns:
            np.ndarray: Modified ECG signal
        """
        # Normalize chemical effects
        chem_a_norm = (chem_a - self.chem_a_min) / (self.chem_a_max - self.chem_a_min)
        chem_b_norm = (chem_b - self.chem_b_min) / (self.chem_b_max - self.chem_b_min)
        
        # Chemical effects on ECG morphology
        # Activator: increases amplitude and frequency
        # Inhibitor: decreases amplitude and frequency
        
        # Amplitude modulation
        amplitude_factor = 1.0 + 0.3 * chem_a_norm - 0.3 * chem_b_norm
        
        # Frequency modulation (simulated by time scaling)
        freq_factor = 1.0 + 0.2 * chem_a_norm - 0.2 * chem_b_norm
        
        # Add some irregularity (simulating afib-like effects)
        irregularity = 0.1 * chem_a_norm * np.random.normal(0, 1, len(base_signal))
        
        # Apply modifications
        modified_signal = base_signal * amplitude_factor + irregularity
        
        # Time scaling (simplified)
        if freq_factor != 1.0:
            # Resample signal to simulate frequency change
            new_length = int(len(modified_signal) / freq_factor)
            resampled = signal.resample(modified_signal, new_length)
            # Ensure we get a numpy array
            if isinstance(resampled, tuple):
                modified_signal = resampled[0]
            else:
                modified_signal = resampled
            # Pad or truncate to original length
            if len(modified_signal) < len(base_signal):
                modified_signal = np.pad(modified_signal, 
                                       (0, len(base_signal) - len(modified_signal)))
            else:
                modified_signal = modified_signal[:len(base_signal)]
        
        return modified_signal.astype(np.float64)
    
    def calculate_deviation_area(self, signal1: np.ndarray, 
                               signal2: np.ndarray) -> float:
        """
        Calculate area between two signals
        
        Args:
            signal1 (np.ndarray): First signal
            signal2 (np.ndarray): Second signal
            
        Returns:
            float: Deviation area
        """
        # Ensure same length
        min_len = min(len(signal1), len(signal2))
        sig1 = signal1[:min_len]
        sig2 = signal2[:min_len]
        
        # Calculate area between curves
        time_axis = np.arange(min_len)
        deviation_area = trapz(np.abs(sig1 - sig2), time_axis)
        
        return float(deviation_area)

class ECGPIDVisualizer:
    """
    Visualizer for ECG PID control loop
    """
    
    def __init__(self, normal_file: str, afib_file: str):
        """
        Initialize visualizer
        
        Args:
            normal_file (str): Path to normal ECG file
            afib_file (str): Path to afib ECG file
        """
        self.normal_file = normal_file
        self.afib_file = afib_file
        
        # Load ECG data
        self.normal_time, self.normal_ecg = self.load_ecg_data(normal_file)
        self.afib_time, self.afib_ecg = self.load_ecg_data(afib_file)
        
        # Preprocess signals
        self.normal_ecg = self.preprocess_ecg(self.normal_ecg)
        self.afib_ecg = self.preprocess_ecg(self.afib_ecg)
        
        # Initialize PID controller
        self.pid_controller = ECGPIDController(kp=0.8, ki=0.2, kd=0.1)
        
        # Extract heartbeat windows for comparison
        self.normal_heartbeats = self.extract_heartbeats(self.normal_ecg)
        self.afib_heartbeats = self.extract_heartbeats(self.afib_ecg)
        
        # Current heartbeat index
        self.heartbeat_idx = 0
        
        # Setup animation
        self.fig, self.axes = plt.subplots(3, 2, figsize=(16, 12))
        self.fig.suptitle('ECG PID Control Loop: Chemical Modulation to Match Afib Pattern', 
                         fontsize=16, fontweight='bold')
        
        # Animation variables
        self.animation = None
        self.time_step = 0
        
    def load_ecg_data(self, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load ECG data from CSV file"""
        df = pd.read_csv(filepath)
        time = df['Time'].values.astype(np.float64)
        ecg = df['ECG'].values.astype(np.float64)
        return time, ecg
    
    def preprocess_ecg(self, ecg: np.ndarray) -> np.ndarray:
        """Preprocess ECG signal"""
        # Simple filtering
        butter_result = signal.butter(3, 0.1, 'low')
        if isinstance(butter_result, tuple) and len(butter_result) >= 2:
            b, a = butter_result[0], butter_result[1]
        else:
            # Fallback if butter doesn't return expected tuple
            b, a = signal.butter(3, 0.1, 'low', output='ba')
        filtered = signal.filtfilt(b, a, ecg)
        return filtered.astype(np.float64)
    
    def extract_heartbeats(self, ecg: np.ndarray, window_size: int = 400) -> List[np.ndarray]:
        """Extract individual heartbeats"""
        # Simple peak detection
        peaks, _ = signal.find_peaks(ecg, height=np.max(ecg)*0.7, distance=200)
        
        heartbeats = []
        for peak in peaks:
            start = max(0, peak - window_size//2)
            end = min(len(ecg), peak + window_size//2)
            heartbeat = ecg[start:end]
            if len(heartbeat) == window_size:
                heartbeats.append(heartbeat.astype(np.float64))
        
        return heartbeats
    
    def update_plot(self, frame: int) -> List[Any]:
        """Update plot for animation"""
        if self.heartbeat_idx >= len(self.normal_heartbeats) or self.heartbeat_idx >= len(self.afib_heartbeats):
            return []
        
        # Get current heartbeats
        normal_hb = self.normal_heartbeats[self.heartbeat_idx]
        afib_hb = self.afib_heartbeats[self.heartbeat_idx]
        
        # Simulate modified normal heartbeat based on current chemical concentrations
        modified_hb = self.pid_controller.simulate_ecg_response(
            self.pid_controller.chem_a, 
            self.pid_controller.chem_b, 
            normal_hb
        )
        
        # Calculate deviation area (error)
        deviation_area = self.pid_controller.calculate_deviation_area(modified_hb, afib_hb)
        error = deviation_area
        
        # Update PID controller
        dt = 0.1  # Time step
        chem_a, chem_b = self.pid_controller.calculate_pid_output(error, dt)
        
        # Store history
        self.pid_controller.time_history.append(self.time_step * dt)
        self.pid_controller.error_history.append(error)
        self.pid_controller.chem_a_history.append(chem_a)
        self.pid_controller.chem_b_history.append(chem_b)
        self.pid_controller.deviation_history.append(deviation_area)
        
        # Clear plots
        for ax in self.axes.flat:
            ax.clear()
        
        # Plot 1: ECG Comparison
        time_axis = np.arange(len(normal_hb)) * 2  # 2ms per sample
        self.axes[0, 0].plot(time_axis, normal_hb, 'b-', linewidth=2, label='Normal ECG', alpha=0.7)
        self.axes[0, 0].plot(time_axis, afib_hb, 'r-', linewidth=2, label='Target (Afib)', alpha=0.7)
        self.axes[0, 0].plot(time_axis, modified_hb, 'g-', linewidth=2, label='Modified (Controlled)', alpha=0.9)
        self.axes[0, 0].fill_between(time_axis, modified_hb, afib_hb, alpha=0.3, color='purple', label='Deviation Area')
        self.axes[0, 0].set_title('ECG Heartbeat Comparison', fontweight='bold')
        self.axes[0, 0].set_xlabel('Time (ms)')
        self.axes[0, 0].set_ylabel('Amplitude (mV)')
        self.axes[0, 0].legend()
        self.axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: PID Control Block Diagram
        self.plot_pid_diagram()
        
        # Plot 3: Chemical Concentrations
        if len(self.pid_controller.time_history) > 1:
            self.axes[1, 0].plot(self.pid_controller.time_history, self.pid_controller.chem_a_history, 
                                'g-', linewidth=2, label='Chemical A (Activator)')
            self.axes[1, 0].plot(self.pid_controller.time_history, self.pid_controller.chem_b_history, 
                                'r-', linewidth=2, label='Chemical B (Inhibitor)')
            self.axes[1, 0].set_title('Chemical Concentrations Over Time', fontweight='bold')
            self.axes[1, 0].set_xlabel('Time (s)')
            self.axes[1, 0].set_ylabel('Concentration')
            self.axes[1, 0].legend()
            self.axes[1, 0].grid(True, alpha=0.3)
            self.axes[1, 0].set_ylim(0, 10)
        
        # Plot 4: Error Signal (Deviation Area)
        if len(self.pid_controller.time_history) > 1:
            self.axes[1, 1].plot(self.pid_controller.time_history, self.pid_controller.error_history, 
                                'purple', linewidth=2, label='Deviation Area (Error)')
            self.axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5, label='Target (Zero Error)')
            self.axes[1, 1].set_title('Error Signal Over Time', fontweight='bold')
            self.axes[1, 1].set_xlabel('Time (s)')
            self.axes[1, 1].set_ylabel('Deviation Area')
            self.axes[1, 1].legend()
            self.axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 5: Current Chemical Levels (Gauge)
        self.plot_chemical_gauges()
        
        # Plot 6: Control Performance Metrics
        self.plot_performance_metrics()
        
        # Update heartbeat index
        self.heartbeat_idx = (self.heartbeat_idx + 1) % min(len(self.normal_heartbeats), len(self.afib_heartbeats))
        self.time_step += 1
        
        plt.tight_layout()
        
        return []
    
    def plot_pid_diagram(self):
        """Plot PID control block diagram"""
        ax = self.axes[0, 1]
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Draw PID control loop
        # Setpoint
        ax.text(1, 8, 'Setpoint\n(Zero Error)', ha='center', va='center', 
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue'))
        
        # Error calculation
        ax.text(3, 8, 'Error\n(Deviation Area)', ha='center', va='center',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow'))
        
        # PID Controller
        ax.text(5, 8, 'PID Controller\n(Kp, Ki, Kd)', ha='center', va='center',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen'))
        
        # Chemical System
        ax.text(7, 8, 'Chemical System\n(A: Activator, B: Inhibitor)', ha='center', va='center',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='orange'))
        
        # ECG Process
        ax.text(7, 6, 'ECG Process\n(Modified Signal)', ha='center', va='center',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral'))
        
        # Feedback
        ax.text(7, 4, 'Feedback\n(Deviation Area)', ha='center', va='center',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray'))
        
        # Arrows
        arrows = [
            ((1.5, 8), (2.5, 8)),  # Setpoint to Error
            ((3.5, 8), (4.5, 8)),  # Error to PID
            ((5.5, 8), (6.5, 8)),  # PID to Chemical
            ((7, 7.5), (7, 6.5)),  # Chemical to ECG
            ((6.5, 6), (4.5, 6)),  # ECG to Feedback
            ((4.5, 5.5), (3.5, 7.5)),  # Feedback to Error
        ]
        
        for start, end in arrows:
            ax.annotate('', xy=end, xytext=start,
                       arrowprops=dict(arrowstyle='->', lw=2, color='black'))
        
        ax.set_title('PID Control Loop Diagram', fontweight='bold')
    
    def plot_chemical_gauges(self):
        """Plot current chemical levels as gauges"""
        ax = self.axes[2, 0]
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Chemical A gauge
        chem_a_level = self.pid_controller.chem_a / self.pid_controller.chem_a_max
        ax.text(2.5, 8, 'Chemical A\n(Activator)', ha='center', va='center', fontweight='bold')
        
        # Gauge background
        gauge_rect = Rectangle((1, 2), 3, 4, facecolor='lightgray', edgecolor='black')
        ax.add_patch(gauge_rect)
        
        # Gauge fill
        fill_height = 4 * chem_a_level
        fill_rect = Rectangle((1, 2), 3, fill_height, facecolor='green', alpha=0.7)
        ax.add_patch(fill_rect)
        
        # Gauge value
        ax.text(2.5, 1, f'{self.pid_controller.chem_a:.1f}', ha='center', va='center', 
               fontsize=14, fontweight='bold')
        
        # Chemical B gauge
        chem_b_level = self.pid_controller.chem_b / self.pid_controller.chem_b_max
        ax.text(7.5, 8, 'Chemical B\n(Inhibitor)', ha='center', va='center', fontweight='bold')
        
        # Gauge background
        gauge_rect2 = Rectangle((6, 2), 3, 4, facecolor='lightgray', edgecolor='black')
        ax.add_patch(gauge_rect2)
        
        # Gauge fill
        fill_height2 = 4 * chem_b_level
        fill_rect2 = Rectangle((6, 2), 3, fill_height2, facecolor='red', alpha=0.7)
        ax.add_patch(fill_rect2)
        
        # Gauge value
        ax.text(7.5, 1, f'{self.pid_controller.chem_b:.1f}', ha='center', va='center', 
               fontsize=14, fontweight='bold')
        
        ax.set_title('Current Chemical Levels', fontweight='bold')
    
    def plot_performance_metrics(self):
        """Plot performance metrics"""
        ax = self.axes[2, 1]
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Current metrics
        current_error = self.pid_controller.error_history[-1] if self.pid_controller.error_history else 0
        avg_error = np.mean(self.pid_controller.error_history) if self.pid_controller.error_history else 0
        
        metrics_text = f"""
Performance Metrics:

Current Error: {current_error:.2f}
Average Error: {avg_error:.2f}
PID Gains:
  Kp: {self.pid_controller.kp}
  Ki: {self.pid_controller.ki}
  Kd: {self.pid_controller.kd}

Chemical A: {self.pid_controller.chem_a:.1f}
Chemical B: {self.pid_controller.chem_b:.1f}

Control Status: {'Active' if current_error > 0.1 else 'Stable'}
        """
        
        ax.text(5, 5, metrics_text, ha='center', va='center', fontsize=10,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
        ax.set_title('Control Performance', fontweight='bold')
    
    def start_animation(self, interval: int = 500):
        """Start the animation"""
        self.animation = animation.FuncAnimation(
            self.fig, self.update_plot, interval=interval, blit=False
        )
        plt.show()
    
    def save_animation(self, filename: str, interval: int = 500):
        """Save animation to file"""
        self.animation = animation.FuncAnimation(
            self.fig, self.update_plot, interval=interval, blit=False
        )
        self.animation.save(filename, writer='pillow', fps=2)

def main():
    """Main function to run the PID control visualization"""
    print("ECG PID Control Loop Visualization")
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
    
    # Create and run visualizer
    visualizer = ECGPIDVisualizer(normal_file, afib_file)
    
    print("\nStarting PID control loop visualization...")
    print("This shows:")
    print("- Real-time ECG comparison (Normal vs Afib vs Controlled)")
    print("- PID control block diagram")
    print("- Chemical concentration modulation over time")
    print("- Error signal (deviation area) minimization")
    print("- Current chemical levels as gauges")
    print("- Performance metrics")
    
    visualizer.start_animation()

if __name__ == "__main__":
    main() 