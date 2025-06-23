"""
ECG Comparison App - Normal vs Afib with PID Control
Clean, focused Streamlit app for ECG signal comparison and convergence
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import sys
import time
from collections import deque
from scipy.integrate import trapz

# Add ECG folder to path
ecg_path = Path(__file__).parent / "ecg"
sys.path.append(str(ecg_path))

# Import ECG helper functions
try:
    from ecg import ECGMorphologyCorrelator, ECGAnalyzer, SyntheticECGGenerator, ECGPIDController
    ECG_MODULES_AVAILABLE = True
except ImportError:
    ECG_MODULES_AVAILABLE = False
    st.error("ECG modules not available. Please ensure the ecg folder is properly set up.")

# Page configuration
st.set_page_config(
    page_title="ECG Comparison: Normal vs Afib",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .control-panel {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

class ECGSignalGenerator:
    """Generate realistic ECG signals for comparison"""
    
    def __init__(self, sampling_rate=500):
        self.sampling_rate = sampling_rate
        self.dt = 1.0 / sampling_rate
        
    def generate_normal_ecg(self, duration=10):
        """Generate normal ECG signal"""
        t = np.arange(0, duration, self.dt)
        
        # Normal ECG parameters
        heart_rate = 72  # BPM
        rr_interval = 60.0 / heart_rate
        
        # Generate multiple heartbeats
        signal = np.zeros_like(t)
        for i in range(int(duration / rr_interval)):
            beat_start = i * rr_interval
            beat_end = min((i + 1) * rr_interval, duration)
            beat_t = np.arange(beat_start, beat_end, self.dt)
            
            if len(beat_t) > 0:
                # Single heartbeat template
                beat_signal = self._generate_heartbeat_template(beat_t - beat_start)
                start_idx = int(beat_start / self.dt)
                end_idx = start_idx + len(beat_signal)
                if end_idx <= len(signal):
                    signal[start_idx:end_idx] = beat_signal
        
        return t, signal
    
    def generate_afib_ecg(self, duration=10):
        """Generate atrial fibrillation ECG signal"""
        t = np.arange(0, duration, self.dt)
        
        # AFib parameters - irregular heart rate
        base_heart_rate = 120  # Higher heart rate
        signal = np.zeros_like(t)
        
        # Irregular RR intervals
        current_time = 0
        while current_time < duration:
            # Random RR interval (irregular)
            rr_interval = np.random.uniform(0.4, 0.8)  # 0.4-0.8 seconds
            
            beat_end = min(current_time + rr_interval, duration)
            beat_t = np.arange(current_time, beat_end, self.dt)
            
            if len(beat_t) > 0:
                # AFib heartbeat template (different morphology)
                beat_signal = self._generate_afib_heartbeat_template(beat_t - current_time)
                start_idx = int(current_time / self.dt)
                end_idx = start_idx + len(beat_signal)
                if end_idx <= len(signal):
                    signal[start_idx:end_idx] = beat_signal
            
            current_time += rr_interval
        
        return t, signal
    
    def _generate_heartbeat_template(self, t):
        """Generate single normal heartbeat template"""
        # P wave
        p_wave = 0.1 * np.exp(-((t - 0.1) / 0.02)**2)
        
        # QRS complex
        qrs = 0.8 * np.exp(-((t - 0.2) / 0.01)**2) - 0.2 * np.exp(-((t - 0.18) / 0.005)**2)
        
        # T wave
        t_wave = 0.3 * np.exp(-((t - 0.35) / 0.04)**2)
        
        # Combine waves
        heartbeat = p_wave + qrs + t_wave
        
        # Add baseline
        heartbeat += 0.1 * np.sin(2 * np.pi * 0.5 * t)
        
        return heartbeat
    
    def _generate_afib_heartbeat_template(self, t):
        """Generate single AFib heartbeat template"""
        # Irregular P wave (absent or abnormal)
        p_wave = 0.05 * np.exp(-((t - 0.08) / 0.03)**2) * np.random.uniform(0.5, 1.5)
        
        # QRS complex (similar but with variations)
        qrs = 0.7 * np.exp(-((t - 0.2) / 0.012)**2) - 0.15 * np.exp(-((t - 0.18) / 0.006)**2)
        qrs *= np.random.uniform(0.8, 1.2)
        
        # T wave (variable)
        t_wave = 0.25 * np.exp(-((t - 0.32) / 0.05)**2) * np.random.uniform(0.7, 1.3)
        
        # Combine waves
        heartbeat = p_wave + qrs + t_wave
        
        # Add more baseline wander
        heartbeat += 0.15 * np.sin(2 * np.pi * 0.3 * t) + 0.1 * np.random.randn(len(t))
        
        return heartbeat

class PIDController:
    """Simple PID controller for signal convergence"""
    
    def __init__(self, kp=0.5, ki=0.1, kd=0.05):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0.0
        self.integral = 0.0
        self.time_history = []
        self.error_history = []
        self.output_history = []
    
    def update(self, error, dt):
        """Update PID controller"""
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        
        self.prev_error = error
        return output
    
    def reset(self):
        """Reset controller state"""
        self.prev_error = 0.0
        self.integral = 0.0
        self.time_history = []
        self.error_history = []
        self.output_history = []

def calculate_deviation_area(signal1, signal2):
    """Calculate area between two signals"""
    min_len = min(len(signal1), len(signal2))
    sig1 = signal1[:min_len]
    sig2 = signal2[:min_len]
    return trapz(np.abs(sig1 - sig2), np.arange(min_len))

def main():
    # Header
    st.markdown('<h1 class="main-header">ECG Comparison: Normal vs Afib</h1>', unsafe_allow_html=True)
    st.markdown("Compare normal and atrial fibrillation ECG signals with PID-controlled convergence")
    
    # Initialize signal generator
    generator = ECGSignalGenerator(sampling_rate=500)
    
    # Sidebar controls
    with st.sidebar:
        st.markdown("### 🎛️ Control Panel")
        
        # Signal parameters
        st.markdown("**Signal Parameters**")
        duration = st.slider("Duration (seconds)", 5, 20, 10)
        sampling_rate = st.slider("Sampling Rate (Hz)", 250, 1000, 500, 50)
        
        # PID parameters
        st.markdown("**PID Control Parameters**")
        kp = st.slider("Proportional Gain (Kp)", 0.1, 2.0, 0.5, 0.1)
        ki = st.slider("Integral Gain (Ki)", 0.01, 0.5, 0.1, 0.01)
        kd = st.slider("Derivative Gain (Kd)", 0.01, 0.3, 0.05, 0.01)
        
        # Playback controls
        st.markdown("**Playback Controls**")
        play_button = st.button("▶️ Start Playback", key="play")
        stop_button = st.button("⏹ Stop", key="stop")
        reset_button = st.button("🔄 Reset", key="reset")
        
        # Analysis options
        st.markdown("**Analysis Options**")
        show_deviation = st.checkbox("Show Deviation Areas", value=True)
        show_pid = st.checkbox("Show PID Control", value=True)
        show_metrics = st.checkbox("Show Metrics", value=True)
    
    # Initialize session state
    if 'playing' not in st.session_state:
        st.session_state.playing = False
    if 'current_time' not in st.session_state:
        st.session_state.current_time = 0.0
    if 'pid_controller' not in st.session_state:
        st.session_state.pid_controller = PIDController(kp, ki, kd)
    
    # Update PID parameters
    st.session_state.pid_controller.kp = kp
    st.session_state.pid_controller.ki = ki
    st.session_state.pid_controller.kd = kd
    
    # Control logic
    if play_button:
        st.session_state.playing = True
    if stop_button:
        st.session_state.playing = False
    if reset_button:
        st.session_state.playing = False
        st.session_state.current_time = 0.0
        st.session_state.pid_controller.reset()
        st.rerun()
    
    # Generate signals
    t_normal, normal_signal = generator.generate_normal_ecg(duration)
    t_afib, afib_signal = generator.generate_afib_ecg(duration)
    
    # Create converged signal (normal signal modified by PID)
    converged_signal = normal_signal.copy()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📊 Signal Comparison")
        
        # Create the main plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot signals
        ax.plot(t_normal, normal_signal, 'b-', linewidth=2, label='Normal ECG', alpha=0.8)
        ax.plot(t_afib, afib_signal, 'r-', linewidth=2, label='Afib ECG', alpha=0.8)
        ax.plot(t_normal, converged_signal, 'g-', linewidth=2, label='Converged (PID)', alpha=0.9)
        
        # Show current time position
        if st.session_state.playing:
            current_idx = int(st.session_state.current_time / generator.dt)
            if current_idx < len(t_normal):
                ax.axvline(x=st.session_state.current_time, color='orange', linestyle='--', 
                          linewidth=2, label='Current Time')
        
        # Show deviation areas
        if show_deviation:
            ax.fill_between(t_normal, normal_signal, converged_signal, 
                          alpha=0.3, color='purple', label='Deviation Area')
        
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Amplitude (mV)')
        ax.set_title('ECG Signal Comparison with PID Control')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        plt.close(fig)
    
    with col2:
        st.markdown("### 📈 Metrics")
        
        if show_metrics:
            # Calculate metrics
            deviation_normal_afib = calculate_deviation_area(normal_signal, afib_signal)
            deviation_normal_converged = calculate_deviation_area(normal_signal, converged_signal)
            
            # Correlation
            correlation = np.corrcoef(normal_signal, converged_signal)[0, 1]
            
            # Display metrics
            st.markdown(f"""
            <div class="metric-card">
                <h4>Signal Metrics</h4>
                <p><strong>Normal-Afib Deviation:</strong> {deviation_normal_afib:.2f}</p>
                <p><strong>Normal-Converged Deviation:</strong> {deviation_normal_converged:.2f}</p>
                <p><strong>Correlation:</strong> {correlation:.3f}</p>
                <p><strong>Convergence:</strong> {((1 - deviation_normal_converged/max(deviation_normal_afib, 0.001)) * 100):.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        if show_pid:
            st.markdown("### 🎛️ PID Control")
            
            # PID status
            current_error = st.session_state.pid_controller.error_history[-1] if st.session_state.pid_controller.error_history else 0
            status = "Active" if current_error > 0.1 else "Stable"
            
            st.markdown(f"""
            <div class="metric-card">
                <h4>Control Status</h4>
                <p><strong>Status:</strong> {status}</p>
                <p><strong>Current Error:</strong> {current_error:.3f}</p>
                <p><strong>Kp:</strong> {kp}</p>
                <p><strong>Ki:</strong> {ki}</p>
                <p><strong>Kd:</strong> {kd}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # PID Control Loop (if playing)
    if st.session_state.playing:
        # Update current time
        st.session_state.current_time += 0.1  # 100ms steps
        
        if st.session_state.current_time >= duration:
            st.session_state.current_time = 0.0
        
        # Calculate error at current time
        current_idx = int(st.session_state.current_time / generator.dt)
        if current_idx < len(normal_signal):
            error = abs(normal_signal[current_idx] - afib_signal[current_idx])
            
            # Update PID controller
            pid_output = st.session_state.pid_controller.update(error, 0.1)
            
            # Apply PID output to converged signal
            if current_idx < len(converged_signal):
                # Modify the converged signal based on PID output
                modification = pid_output * 0.1  # Scale factor
                converged_signal[current_idx] = normal_signal[current_idx] + modification
            
            # Store history
            st.session_state.pid_controller.time_history.append(st.session_state.current_time)
            st.session_state.pid_controller.error_history.append(error)
            st.session_state.pid_controller.output_history.append(pid_output)
        
        # Auto-refresh for animation effect
        time.sleep(0.1)
        st.rerun()
    
    # Additional plots
    if show_pid and st.session_state.pid_controller.time_history:
        st.markdown("### 📊 PID Control Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Error over time
            fig_error, ax_error = plt.subplots(figsize=(8, 4))
            ax_error.plot(st.session_state.pid_controller.time_history, 
                         st.session_state.pid_controller.error_history, 'r-', linewidth=2)
            ax_error.set_xlabel('Time (s)')
            ax_error.set_ylabel('Error')
            ax_error.set_title('Error Signal Over Time')
            ax_error.grid(True, alpha=0.3)
            st.pyplot(fig_error)
            plt.close(fig_error)
        
        with col2:
            # PID output over time
            fig_output, ax_output = plt.subplots(figsize=(8, 4))
            ax_output.plot(st.session_state.pid_controller.time_history, 
                          st.session_state.pid_controller.output_history, 'g-', linewidth=2)
            ax_output.set_xlabel('Time (s)')
            ax_output.set_ylabel('PID Output')
            ax_output.set_title('PID Control Output')
            ax_output.grid(True, alpha=0.3)
            st.pyplot(fig_output)
            plt.close(fig_output)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>ECG Comparison App - Demonstrating PID control for signal convergence</p>
        <p>Normal ECG → PID Controller → Converged Signal (closer to Afib)</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 