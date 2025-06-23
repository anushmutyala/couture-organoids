#!/usr/bin/env python3
"""
Real-time ECG Simulator
Simulates healthy ECG signals and plays them back in real time
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
import time
import threading
import queue
from synthetic_ecg_generator import SyntheticECGGenerator

class RealTimeECGSimulator:
    """
    Real-time ECG simulator with interactive controls
    """
    
    def __init__(self, sampling_rate: int = 500, buffer_duration: float = 10.0):
        """
        Initialize the real-time ECG simulator
        
        Args:
            sampling_rate (int): Sampling rate in Hz
            buffer_duration (float): Duration of the display buffer in seconds
        """
        self.sampling_rate = sampling_rate
        self.buffer_duration = buffer_duration
        self.buffer_size = int(sampling_rate * buffer_duration)
        
        # ECG generator
        self.ecg_generator = SyntheticECGGenerator(sampling_rate=sampling_rate, duration=60.0)
        
        # Real-time parameters
        self.heart_rate = 72  # BPM
        self.noise_level = 0.05
        self.is_running = False
        self.start_time = None
        
        # Data buffers
        self.time_buffer = np.linspace(0, buffer_duration, self.buffer_size)
        self.ecg_buffer = np.zeros(self.buffer_size)
        
        # Threading for real-time generation
        self.data_queue = queue.Queue(maxsize=1000)
        self.generation_thread = None
        
        # Setup the plot
        self.setup_plot()
        
    def setup_plot(self):
        """Setup the matplotlib plot with controls"""
        # Create figure and subplots
        self.fig, (self.ax_ecg, self.ax_controls) = plt.subplots(2, 1, 
                                                                  figsize=(12, 8),
                                                                  gridspec_kw={'height_ratios': [4, 1]})
        
        # ECG plot
        self.ax_ecg.set_title('Real-time ECG Simulation', fontsize=14, fontweight='bold')
        self.ax_ecg.set_xlabel('Time (seconds)')
        self.ax_ecg.set_ylabel('Amplitude (mV)')
        self.ax_ecg.grid(True, alpha=0.3)
        self.ax_ecg.set_ylim(-2, 2)
        
        # Initialize ECG line
        self.ecg_line, = self.ax_ecg.plot(self.time_buffer, self.ecg_buffer, 
                                         'b-', linewidth=1.5, label='ECG Signal')
        
        # Add heart rate indicator
        self.hr_text = self.ax_ecg.text(0.02, 0.95, f'Heart Rate: {self.heart_rate} BPM', 
                                       transform=self.ax_ecg.transAxes, 
                                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        
        # Setup controls
        self.setup_controls()
        
        # Adjust layout
        plt.tight_layout()
        
    def setup_controls(self):
        """Setup interactive controls"""
        # Control panel
        self.ax_controls.set_visible(False)  # Hide the control subplot
        
        # Heart rate slider
        ax_hr_slider = plt.axes((0.2, 0.05, 0.3, 0.03))
        self.hr_slider = Slider(ax_hr_slider, 'Heart Rate (BPM)', 40, 120, 
                               valinit=self.heart_rate, valstep=1)
        self.hr_slider.on_changed(self.update_heart_rate)
        
        # Noise level slider
        ax_noise_slider = plt.axes((0.6, 0.05, 0.3, 0.03))
        self.noise_slider = Slider(ax_noise_slider, 'Noise Level', 0, 0.2, 
                                  valinit=self.noise_level, valstep=0.01)
        self.noise_slider.on_changed(self.update_noise_level)
        
        # Start/Stop button
        ax_button = plt.axes((0.45, 0.12, 0.1, 0.04))
        self.start_stop_button = Button(ax_button, 'Start')
        self.start_stop_button.on_clicked(self.toggle_simulation)
        
        # Reset button
        ax_reset_button = plt.axes((0.6, 0.12, 0.1, 0.04))
        self.reset_button = Button(ax_reset_button, 'Reset')
        self.reset_button.on_clicked(self.reset_simulation)
        
    def update_heart_rate(self, val):
        """Update heart rate from slider"""
        self.heart_rate = int(val)
        self.hr_text.set_text(f'Heart Rate: {self.heart_rate} BPM')
        
    def update_noise_level(self, val):
        """Update noise level from slider"""
        self.noise_level = val
        
    def toggle_simulation(self, event):
        """Toggle simulation start/stop"""
        if self.is_running:
            self.stop_simulation()
        else:
            self.start_simulation()
            
    def start_simulation(self):
        """Start the real-time simulation"""
        self.is_running = True
        self.start_time = time.time()
        self.start_stop_button.label.set_text('Stop')
        self.start_generation_thread()
        
    def stop_simulation(self):
        """Stop the real-time simulation"""
        self.is_running = False
        self.start_stop_button.label.set_text('Start')
        if self.generation_thread and self.generation_thread.is_alive():
            self.generation_thread.join(timeout=1.0)
            
    def reset_simulation(self, event):
        """Reset the simulation"""
        self.stop_simulation()
        self.ecg_buffer = np.zeros(self.buffer_size)
        self.ecg_line.set_ydata(self.ecg_buffer)
        self.start_time = None
        
    def start_generation_thread(self):
        """Start the ECG generation thread"""
        self.generation_thread = threading.Thread(target=self.generate_ecg_data, daemon=True)
        self.generation_thread.start()
        
    def generate_ecg_data(self):
        """Generate ECG data in a separate thread"""
        while self.is_running:
            try:
                # Generate a short segment of ECG data
                segment_duration = 0.1  # 100ms segments
                segment_samples = int(self.sampling_rate * segment_duration)
                
                # Generate ECG segment
                time_segment = np.linspace(0, segment_duration, segment_samples)
                _, ecg_segment = self.ecg_generator.generate_simple_mathematical_ecg(
                    heart_rate=self.heart_rate, 
                    noise_level=self.noise_level
                )
                
                # Take only the first segment_samples
                ecg_segment = ecg_segment[:segment_samples]
                
                # Add timestamp
                current_time = time.time() - self.start_time if self.start_time else 0
                
                # Put data in queue
                self.data_queue.put((current_time, ecg_segment), timeout=0.1)
                
                # Sleep to maintain real-time pacing
                time.sleep(segment_duration)
                
            except queue.Full:
                continue
            except Exception as e:
                print(f"Error in generation thread: {e}")
                break
                
    def update_plot(self, frame):
        """Update the plot with new data"""
        if not self.is_running:
            return self.ecg_line,
            
        # Get new data from queue
        new_data = []
        while not self.data_queue.empty() and len(new_data) < 10:  # Limit updates per frame
            try:
                timestamp, ecg_segment = self.data_queue.get_nowait()
                new_data.append((timestamp, ecg_segment))
            except queue.Empty:
                break
                
        if new_data:
            # Update the buffer with new data
            for timestamp, ecg_segment in new_data:
                # Calculate position in buffer
                buffer_pos = int((timestamp % self.buffer_duration) * self.sampling_rate)
                
                # Update buffer
                for i, sample in enumerate(ecg_segment):
                    pos = (buffer_pos + i) % self.buffer_size
                    self.ecg_buffer[pos] = sample
                    
        # Update the plot
        self.ecg_line.set_ydata(self.ecg_buffer)
        
        return self.ecg_line,
        
    def run(self):
        """Run the real-time ECG simulation"""
        # Setup animation
        self.ani = animation.FuncAnimation(
            self.fig, self.update_plot, interval=50,  # 20 FPS
            blit=True, cache_frame_data=False
        )
        
        # Show the plot
        plt.show()
        
    def close(self):
        """Clean up resources"""
        self.stop_simulation()
        plt.close(self.fig)


def main():
    """Main function to run the real-time ECG simulator"""
    print("Starting Real-time ECG Simulator...")
    print("Controls:")
    print("- Heart Rate Slider: Adjust heart rate (40-120 BPM)")
    print("- Noise Level Slider: Adjust signal noise (0-0.2)")
    print("- Start/Stop Button: Control simulation")
    print("- Reset Button: Clear the display")
    print("\nPress Ctrl+C to exit")
    
    try:
        # Create and run simulator
        simulator = RealTimeECGSimulator(sampling_rate=500, buffer_duration=10.0)
        simulator.run()
    except KeyboardInterrupt:
        print("\nSimulation stopped by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'simulator' in locals():
            simulator.close()


if __name__ == "__main__":
    main() 