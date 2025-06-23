"""
Streamlit ECG Dashboard - Enhanced with ECG Analysis Tools

Features:
* Real-time ECG signal comparison
* Morphology correlation analysis
* PID control loop for signal convergence
* Interval analysis and annotations
* Deviation area visualization

How to run:
    streamlit run streamlit-app.py
"""
import time
import os
import sys
from typing import Tuple, Deque, Optional, List
from pathlib import Path

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from collections import deque
import pandas as pd
from scipy import signal
from scipy.integrate import trapz

# Add ECG folder to path for imports
ecg_path = Path(__file__).parent / "ecg"
sys.path.append(str(ecg_path))

# Import ECG helper functions
try:
    from ecg import ECGMorphologyCorrelator, ECGAnalyzer, SyntheticECGGenerator, ECGPIDController
    print("✓ Successfully imported ECG modules")
except ImportError as e:
    print(f"✗ Failed to import ECG modules: {e}")
    print("Falling back to direct imports...")
    try:
        from ecg.ecg_morphology_correlator import ECGMorphologyCorrelator
        from ecg.ecg_analyzer import ECGAnalyzer
        from ecg.synthetic_ecg_generator import SyntheticECGGenerator
        from ecg.ecg_pid_control import ECGPIDController
        print("✓ Successfully imported ECG modules via direct imports")
    except ImportError as e2:
        print(f"✗ Direct imports also failed: {e2}")
        print("Please ensure the ecg folder contains all required modules")
        # Create dummy classes for fallback
        class ECGMorphologyCorrelator:
            def __init__(self, **kwargs): pass
        class ECGAnalyzer:
            def __init__(self, **kwargs): pass
        class SyntheticECGGenerator:
            def __init__(self, **kwargs): pass
        class ECGPIDController:
            def __init__(self, **kwargs): pass

# ————————————————————————————————————————————————
# CONFIGURATION
# ————————————————————————————————————————————————
BUFFER_SECONDS: int = 10           # visible window length in seconds
FS: int = 500                     # sampling rate [Hz]
MAX_POINTS: int = BUFFER_SECONDS * FS

# ————————————————————————————————————————————————
# DATA BUFFERS
# ————————————————————————————————————————————————
t_buf: Deque[float] = deque(maxlen=MAX_POINTS)
human_buf: Deque[float] = deque(maxlen=MAX_POINTS)
organoid_buf: Deque[float] = deque(maxlen=MAX_POINTS)
converged_buf: Deque[float] = deque(maxlen=MAX_POINTS)

# ————————————————————————————————————————————————
# ECG ANALYSIS OBJECTS
# ————————————————————————————————————————————————
@st.cache_resource
def initialize_ecg_objects():
    """Initialize ECG analysis objects"""
    correlator = ECGMorphologyCorrelator(sampling_rate=FS, window_size=0.8)
    analyzer = ECGAnalyzer(sampling_rate=FS)
    generator = SyntheticECGGenerator(sampling_rate=FS)
    pid_controller = ECGPIDController(kp=0.8, ki=0.2, kd=0.1)
    return correlator, analyzer, generator, pid_controller

# ————————————————————————————————————————————————
# ENHANCED DATA GENERATION
# ————————————————————————————————————————————————
def generate_realistic_ecg(t: float, base_freq: float = 1.2, 
                          noise_level: float = 0.05, 
                          irregularity: float = 0.0) -> float:
    """Generate realistic ECG-like signal"""
    # Base ECG components
    fundamental = np.sin(2 * np.pi * base_freq * t)
    
    # Add harmonics for more realistic shape
    harmonic1 = 0.3 * np.sin(4 * np.pi * base_freq * t)
    harmonic2 = 0.1 * np.sin(6 * np.pi * base_freq * t)
    
    # Add noise
    noise = noise_level * np.random.randn()
    
    # Add irregularity (for afib simulation)
    if irregularity > 0:
        irregularity_factor = 1 + irregularity * np.sin(3 * np.pi * base_freq * t)
        fundamental *= irregularity_factor
    
    return fundamental + harmonic1 + harmonic2 + noise

def get_next_sample(t: float, mode: str = "normal") -> Tuple[float, float, float]:
    """Return next (human, organoid, converged) ECG samples."""
    
    # Human ECG (normal)
    human = generate_realistic_ecg(t, base_freq=1.2, noise_level=0.03)
    
    # Organoid ECG (varies by mode)
    if mode == "normal":
        organoid = generate_realistic_ecg(t, base_freq=1.2, noise_level=0.05)
    elif mode == "afib":
        organoid = generate_realistic_ecg(t, base_freq=1.8, noise_level=0.08, irregularity=0.3)
    elif mode == "tachycardia":
        organoid = generate_realistic_ecg(t, base_freq=2.5, noise_level=0.06)
    else:
        organoid = generate_realistic_ecg(t, base_freq=1.2, noise_level=0.05)
    
    # Converged signal (controlled by PID)
    converged = human  # Will be updated by PID controller
    
    return human, organoid, converged

# ————————————————————————————————————————————————
# ANALYSIS FUNCTIONS
# ————————————————————————————————————————————————
def calculate_deviation_area(signal1: np.ndarray, signal2: np.ndarray) -> float:
    """Calculate area between two signals"""
    min_len = min(len(signal1), len(signal2))
    sig1 = signal1[:min_len]
    sig2 = signal2[:min_len]
    time_axis = np.arange(min_len)
    return trapz(np.abs(sig1 - sig2), time_axis)

def extract_heartbeat_features(signal_data: np.ndarray, time_axis: np.ndarray) -> dict:
    """Extract key heartbeat features"""
    # Simple peak detection
    peaks, _ = signal.find_peaks(signal_data, height=np.max(signal_data)*0.7, distance=200)
    
    if len(peaks) > 0:
        # Use first peak for analysis
        peak_idx = peaks[0]
        window_size = 400
        start = max(0, peak_idx - window_size//2)
        end = min(len(signal_data), peak_idx + window_size//2)
        heartbeat = signal_data[start:end]
        
        # Basic features
        amplitude = np.max(heartbeat) - np.min(heartbeat)
        duration = len(heartbeat) * (time_axis[1] - time_axis[0]) if len(time_axis) > 1 else 0.8
        
        return {
            'amplitude': amplitude,
            'duration': duration,
            'peak_count': len(peaks),
            'heartbeat': heartbeat
        }
    
    return {'amplitude': 0, 'duration': 0, 'peak_count': 0, 'heartbeat': np.array([])}

# ————————————————————————————————————————————————
# STREAMLIT PAGE SETUP
# ————————————————————————————————————————————————
st.set_page_config(page_title="Enhanced ECG Dashboard", layout="wide")
st.title("Enhanced ECG Dashboard - Signal Comparison & Convergence")

# Initialize ECG objects
correlator, analyzer, generator, pid_controller = initialize_ecg_objects()

# Sidebar controls
with st.sidebar:
    st.markdown("### Data Controls")
    run_button = st.button("▶️ Start", key="start")
    stop_button = st.button("⏹ Stop", key="stop")
    
    st.markdown("### Signal Mode")
    mode = st.selectbox(
        "Organoid Signal Type",
        ["normal", "afib", "tachycardia"],
        help="Select the type of signal to compare against"
    )
    
    st.markdown("### PID Control")
    kp = st.slider("Proportional Gain (Kp)", 0.1, 2.0, 0.8, 0.1)
    ki = st.slider("Integral Gain (Ki)", 0.01, 0.5, 0.2, 0.01)
    kd = st.slider("Derivative Gain (Kd)", 0.01, 0.3, 0.1, 0.01)
    
    # Update PID gains
    pid_controller.kp = kp
    pid_controller.ki = ki
    pid_controller.kd = kd
    
    st.markdown("### Analysis Settings")
    show_intervals = st.checkbox("Show ECG Intervals", value=True)
    show_correlation = st.checkbox("Show Correlation Analysis", value=True)
    
    st.markdown("---")
    st.caption("Enhanced ECG analysis with PID control for signal convergence")

# Flag stored in session_state
if "running" not in st.session_state:
    st.session_state.running = False
if run_button:
    st.session_state.running = True
if stop_button:
    st.session_state.running = False

# ————————————————————————————————————————————————
# MAIN LOOP
# ————————————————————————————————————————————————
prev_time = time.perf_counter()

# Create placeholders for plots
col1, col2 = st.columns(2)
with col1:
    main_plot_placeholder = st.empty()
    correlation_plot_placeholder = st.empty()

with col2:
    control_plot_placeholder = st.empty()
    metrics_placeholder = st.empty()

while st.session_state.running:
    # 1️⃣ Acquire new sample
    current_time = time.perf_counter()
    dt = current_time - prev_time
    prev_time = current_time
    next_t = t_buf[-1] + dt if t_buf else 0.0
    human_val, organoid_val, converged_val = get_next_sample(next_t, mode)

    # 2️⃣ Update buffers
    t_buf.append(next_t)
    human_buf.append(human_val)
    organoid_buf.append(organoid_val)
    
    # 3️⃣ PID Control for convergence
    if len(human_buf) > 100 and len(organoid_buf) > 100:
        # Use recent samples for PID control
        recent_human = np.array(list(human_buf)[-100:])
        recent_organoid = np.array(list(organoid_buf)[-100:])
        
        # Calculate error (deviation area)
        error = calculate_deviation_area(recent_human, recent_organoid)
        
        # Update PID controller
        pid_output = pid_controller.calculate_pid_output(error, dt)
        
        # Apply PID output to modify converged signal
        chem_a, chem_b = pid_output
        converged_val = human_val * (1 + 0.1 * chem_a - 0.1 * chem_b)
    
    converged_buf.append(converged_val)

    # 4️⃣ Analysis
    if len(t_buf) > 200:
        # Convert to numpy arrays
        t_array = np.array(t_buf)
        human_array = np.array(human_buf)
        organoid_array = np.array(organoid_buf)
        converged_array = np.array(converged_buf)
        
        # Extract features
        human_features = extract_heartbeat_features(human_array, t_array)
        organoid_features = extract_heartbeat_features(organoid_array, t_array)
        
        # Calculate metrics
        deviation_human_organoid = calculate_deviation_area(human_array, organoid_array)
        deviation_human_converged = calculate_deviation_area(human_array, converged_array)
        correlation = np.corrcoef(human_array, converged_array)[0, 1] if len(human_array) > 1 else 0
        
        # 5️⃣ Create comprehensive visualization
        
        # Main comparison plot
        fig_main, axs_main = plt.subplots(2, 1, figsize=(10, 8))
        
        # Top: Signal comparison
        axs_main[0].plot(t_array, human_array, 'b-', linewidth=2, label='Human ECG', alpha=0.8)
        axs_main[0].plot(t_array, organoid_array, 'r-', linewidth=2, label=f'Organoid ECG ({mode})', alpha=0.8)
        axs_main[0].plot(t_array, converged_array, 'g-', linewidth=2, label='Converged (Controlled)', alpha=0.9)
        axs_main[0].fill_between(t_array, human_array, converged_array, alpha=0.3, color='purple', label='Deviation Area')
        axs_main[0].set_title('Real-Time ECG Signal Comparison', fontweight='bold')
        axs_main[0].set_ylabel('Amplitude (mV)')
        axs_main[0].legend()
        axs_main[0].grid(True, alpha=0.3)
        
        # Bottom: Deviation over time
        deviation_history = []
        for i in range(50, len(t_array)):
            dev = calculate_deviation_area(human_array[:i], converged_array[:i])
            deviation_history.append(dev)
        
        if deviation_history:
            axs_main[1].plot(t_array[50:], deviation_history, 'purple', linewidth=2, label='Deviation Area')
            axs_main[1].axhline(y=0, color='k', linestyle='--', alpha=0.5, label='Target (Zero Error)')
            axs_main[1].set_title('Convergence Progress (Deviation Area)', fontweight='bold')
            axs_main[1].set_xlabel('Time (s)')
            axs_main[1].set_ylabel('Deviation Area')
            axs_main[1].legend()
            axs_main[1].grid(True, alpha=0.3)
        
        fig_main.tight_layout()
        main_plot_placeholder.pyplot(fig_main, clear_figure=True)
        plt.close(fig_main)
        
        # Correlation analysis plot
        if show_correlation:
            fig_corr, axs_corr = plt.subplots(1, 2, figsize=(10, 4))
            
            # Heartbeat comparison
            if len(human_features['heartbeat']) > 0 and len(organoid_features['heartbeat']) > 0:
                hb_time = np.arange(len(human_features['heartbeat'])) * 2  # 2ms per sample
                axs_corr[0].plot(hb_time, human_features['heartbeat'], 'b-', linewidth=2, label='Human')
                axs_corr[0].plot(hb_time, organoid_features['heartbeat'], 'r-', linewidth=2, label=f'Organoid ({mode})')
                axs_corr[0].set_title('Individual Heartbeat Comparison')
                axs_corr[0].set_xlabel('Time (ms)')
                axs_corr[0].set_ylabel('Amplitude (mV)')
                axs_corr[0].legend()
                axs_corr[0].grid(True, alpha=0.3)
            
            # Correlation over time
            window_size = 100
            correlations = []
            for i in range(window_size, len(t_array)):
                corr = np.corrcoef(human_array[i-window_size:i], converged_array[i-window_size:i])[0, 1]
                correlations.append(corr if not np.isnan(corr) else 0)
            
            if correlations:
                axs_corr[1].plot(t_array[window_size:], correlations, 'g-', linewidth=2)
                axs_corr[1].set_title('Correlation Over Time')
                axs_corr[1].set_xlabel('Time (s)')
                axs_corr[1].set_ylabel('Correlation Coefficient')
                axs_corr[1].grid(True, alpha=0.3)
                axs_corr[1].set_ylim(-1, 1)
            
            fig_corr.tight_layout()
            correlation_plot_placeholder.pyplot(fig_corr, clear_figure=True)
            plt.close(fig_corr)
        
        # Control system plot
        fig_control, axs_control = plt.subplots(2, 2, figsize=(10, 8))
        
        # Chemical concentrations
        if hasattr(pid_controller, 'time_history') and len(pid_controller.time_history) > 1:
            axs_control[0, 0].plot(pid_controller.time_history, pid_controller.chem_a_history, 
                                 'g-', linewidth=2, label='Chemical A (Activator)')
            axs_control[0, 0].plot(pid_controller.time_history, pid_controller.chem_b_history, 
                                 'r-', linewidth=2, label='Chemical B (Inhibitor)')
            axs_control[0, 0].set_title('Chemical Concentrations')
            axs_control[0, 0].set_ylabel('Concentration')
            axs_control[0, 0].legend()
            axs_control[0, 0].grid(True, alpha=0.3)
        
        # Error signal
        if hasattr(pid_controller, 'error_history') and len(pid_controller.error_history) > 1:
            axs_control[0, 1].plot(pid_controller.time_history, pid_controller.error_history, 
                                 'purple', linewidth=2, label='Error (Deviation Area)')
            axs_control[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
            axs_control[0, 1].set_title('Error Signal')
            axs_control[0, 1].set_ylabel('Deviation Area')
            axs_control[0, 1].legend()
            axs_control[0, 1].grid(True, alpha=0.3)
        
        # PID gains visualization
        axs_control[1, 0].bar(['Kp', 'Ki', 'Kd'], [kp, ki, kd], color=['blue', 'green', 'red'])
        axs_control[1, 0].set_title('PID Gains')
        axs_control[1, 0].set_ylabel('Gain Value')
        axs_control[1, 0].grid(True, alpha=0.3)
        
        # Control status
        current_error = pid_controller.error_history[-1] if hasattr(pid_controller, 'error_history') and pid_controller.error_history else 0
        status = "Active" if current_error > 0.1 else "Stable"
        axs_control[1, 1].text(0.5, 0.5, f"Control Status:\n{status}\n\nCurrent Error:\n{current_error:.3f}", 
                              ha='center', va='center', fontsize=12,
                              bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        axs_control[1, 1].set_title('Control Status')
        axs_control[1, 1].axis('off')
        
        fig_control.tight_layout()
        control_plot_placeholder.pyplot(fig_control, clear_figure=True)
        plt.close(fig_control)
        
        # Metrics display
        metrics_html = f"""
        <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px;">
            <h3>Real-Time Metrics</h3>
            <table style="width: 100%; border-collapse: collapse;">
                <tr><td><strong>Human-Organoid Deviation:</strong></td><td>{deviation_human_organoid:.3f}</td></tr>
                <tr><td><strong>Human-Converged Deviation:</strong></td><td>{deviation_human_converged:.3f}</td></tr>
                <tr><td><strong>Correlation:</strong></td><td>{correlation:.3f}</td></tr>
                <tr><td><strong>Human Amplitude:</strong></td><td>{human_features['amplitude']:.3f}</td></tr>
                <tr><td><strong>Organoid Amplitude:</strong></td><td>{organoid_features['amplitude']:.3f}</td></tr>
                <tr><td><strong>Convergence Progress:</strong></td><td>{((1 - deviation_human_converged/max(deviation_human_organoid, 0.001)) * 100):.1f}%</td></tr>
            </table>
        </div>
        """
        metrics_placeholder.markdown(metrics_html, unsafe_allow_html=True)

    # 6️⃣ Sleep to maintain sampling rate
    time.sleep(1.0 / FS)

# When stopped, show message
if not st.session_state.running:
    st.info("Click **Start** to begin streaming ECG analysis.")
    
    # Show static analysis if data is available
    if len(t_buf) > 0:
        st.markdown("### Static Analysis of Last Session")
        
        # Create summary plot
        fig_summary, ax_summary = plt.subplots(figsize=(12, 6))
        t_array = np.array(t_buf)
        human_array = np.array(human_buf)
        organoid_array = np.array(organoid_buf)
        converged_array = np.array(converged_buf)
        
        ax_summary.plot(t_array, human_array, 'b-', linewidth=2, label='Human ECG', alpha=0.8)
        ax_summary.plot(t_array, organoid_array, 'r-', linewidth=2, label=f'Organoid ECG ({mode})', alpha=0.8)
        ax_summary.plot(t_array, converged_array, 'g-', linewidth=2, label='Converged (Controlled)', alpha=0.9)
        ax_summary.set_title('Session Summary - ECG Signal Comparison')
        ax_summary.set_xlabel('Time (s)')
        ax_summary.set_ylabel('Amplitude (mV)')
        ax_summary.legend()
        ax_summary.grid(True, alpha=0.3)
        
        st.pyplot(fig_summary)
        plt.close(fig_summary)
