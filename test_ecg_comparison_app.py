#!/usr/bin/env python3
"""
Test script for ECG Comparison App
Verifies the app functionality and signal generation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapz

def test_signal_generation():
    """Test ECG signal generation"""
    print("Testing ECG signal generation...")
    
    # Import the signal generator
    from ecg_comparison_app import ECGSignalGenerator
    
    # Create generator
    generator = ECGSignalGenerator(sampling_rate=500)
    
    # Generate signals
    t_normal, normal_signal = generator.generate_normal_ecg(duration=5)
    t_afib, afib_signal = generator.generate_afib_ecg(duration=5)
    
    print(f"✓ Normal signal generated: {len(normal_signal)} samples")
    print(f"✓ Afib signal generated: {len(afib_signal)} samples")
    print(f"✓ Time range: {t_normal[0]:.1f} to {t_normal[-1]:.1f} seconds")
    
    return t_normal, normal_signal, afib_signal

def test_pid_controller():
    """Test PID controller"""
    print("\nTesting PID controller...")
    
    from ecg_comparison_app import PIDController
    
    # Create controller
    pid = PIDController(kp=0.5, ki=0.1, kd=0.05)
    
    # Test update
    error = 1.0
    output = pid.update(error, 0.1)
    
    print(f"✓ PID controller created")
    print(f"✓ Error: {error}, Output: {output:.3f}")
    
    return pid

def test_deviation_calculation():
    """Test deviation area calculation"""
    print("\nTesting deviation calculation...")
    
    from ecg_comparison_app import calculate_deviation_area
    
    # Create test signals
    t = np.linspace(0, 1, 100)
    signal1 = np.sin(2 * np.pi * 2 * t)
    signal2 = np.sin(2 * np.pi * 2 * t + 0.5)  # Phase shifted
    
    # Calculate deviation
    deviation = calculate_deviation_area(signal1, signal2)
    
    print(f"✓ Deviation calculation: {deviation:.3f}")
    
    return deviation

def test_visualization():
    """Test signal visualization"""
    print("\nTesting visualization...")
    
    # Generate test signals
    t_normal, normal_signal, afib_signal = test_signal_generation()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(t_normal, normal_signal, 'b-', linewidth=2, label='Normal ECG', alpha=0.8)
    ax.plot(t_normal, afib_signal, 'r-', linewidth=2, label='Afib ECG', alpha=0.8)
    
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Amplitude (mV)')
    ax.set_title('ECG Signal Comparison Test')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save test plot
    plt.savefig('test_ecg_signals.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Visualization test completed")
    print("✓ Test plot saved as 'test_ecg_signals.png'")

def main():
    """Main test function"""
    print("ECG Comparison App - Functionality Test")
    print("=" * 50)
    
    try:
        # Test all components
        test_signal_generation()
        test_pid_controller()
        test_deviation_calculation()
        test_visualization()
        
        print("\n🎉 All tests passed!")
        print("\nTo run the Streamlit app:")
        print("streamlit run ecg_comparison_app.py")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print("Please check the error and fix the issue.")

if __name__ == "__main__":
    main() 