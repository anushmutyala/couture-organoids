#!/usr/bin/env python3
"""
ECG Data Reader for Arduino
Reads ECG data from Arduino Uno via Serial communication
Supports real-time plotting and data logging
"""

import serial
import time
import csv
import threading
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import numpy as np

class ECGReader:
    def __init__(self, port='COM3', baudrate=9600, max_points=500):
        """
        Initialize ECG reader
        
        Args:
            port (str): Serial port (e.g., 'COM3' on Windows, '/dev/ttyUSB0' on Linux)
            baudrate (int): Serial baud rate (must match Arduino)
            max_points (int): Maximum number of points to display in real-time plot
        """
        self.port = port
        self.baudrate = baudrate
        self.max_points = max_points
        
        # Data storage
        self.timestamps = deque(maxlen=max_points)
        self.ecg_values = deque(maxlen=max_points)
        self.raw_data = []
        
        # Serial connection
        self.serial_conn = None
        self.is_connected = False
        self.is_running = False
        
        # Plotting
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.line, = self.ax.plot([], [], 'b-', linewidth=1)
        
        # Statistics
        self.stats = {
            'min_value': float('inf'),
            'max_value': float('-inf'),
            'avg_value': 0,
            'sample_count': 0
        }
    
    def connect(self):
        """Establish serial connection to Arduino"""
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=1
            )
            time.sleep(2)  # Wait for Arduino to reset
            self.is_connected = True
            print(f"Connected to Arduino on {self.port}")
            return True
        except serial.SerialException as e:
            print(f"Failed to connect to {self.port}: {e}")
            return False
    
    def disconnect(self):
        """Close serial connection"""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            self.is_connected = False
            print("Disconnected from Arduino")
    
    def read_data(self):
        """Read data from Arduino in a separate thread"""
        while self.is_running and self.is_connected:
            try:
                if self.serial_conn and self.serial_conn.in_waiting > 0:
                    line = self.serial_conn.readline().decode('utf-8').strip()
                    if line and ',' in line:
                        parts = line.split(',')
                        if len(parts) == 2:
                            timestamp = int(parts[0])
                            ecg_value = int(parts[1])
                            
                            # Store data
                            self.timestamps.append(timestamp)
                            self.ecg_values.append(ecg_value)
                            self.raw_data.append([timestamp, ecg_value])
                            
                            # Update statistics
                            self.update_stats(ecg_value)
                            
            except (serial.SerialException, ValueError) as e:
                print(f"Error reading data: {e}")
                break
    
    def update_stats(self, value):
        """Update running statistics"""
        self.stats['min_value'] = min(self.stats['min_value'], value)
        self.stats['max_value'] = max(self.stats['max_value'], value)
        self.stats['sample_count'] += 1
        
        # Calculate running average
        if self.stats['sample_count'] == 1:
            self.stats['avg_value'] = value
        else:
            self.stats['avg_value'] = (self.stats['avg_value'] * (self.stats['sample_count'] - 1) + value) / self.stats['sample_count']
    
    def start_reading(self):
        """Start reading data from Arduino"""
        if not self.is_connected:
            if not self.connect():
                return False
        
        self.is_running = True
        self.read_thread = threading.Thread(target=self.read_data)
        self.read_thread.daemon = True
        self.read_thread.start()
        print("Started reading ECG data...")
        return True
    
    def stop_reading(self):
        """Stop reading data"""
        self.is_running = False
        if hasattr(self, 'read_thread'):
            self.read_thread.join(timeout=1)
        print("Stopped reading ECG data")
    
    def save_data(self, filename=None):
        """Save collected data to CSV file"""
        if not self.raw_data:
            print("No data to save")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ecg_data_{timestamp}.csv"
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Timestamp', 'ECG_Value'])
            writer.writerows(self.raw_data)
        
        print(f"Data saved to {filename}")
    
    def print_stats(self):
        """Print current statistics"""
        print("\n=== ECG Statistics ===")
        print(f"Sample Count: {self.stats['sample_count']}")
        print(f"Min Value: {self.stats['min_value']}")
        print(f"Max Value: {self.stats['max_value']}")
        print(f"Average Value: {self.stats['avg_value']:.2f}")
        print(f"Signal Range: {self.stats['max_value'] - self.stats['min_value']}")
        print("=====================")
    
    def animate_plot(self, frame):
        """Animation function for real-time plotting"""
        if len(self.timestamps) > 1:
            # Convert to relative time (seconds from start)
            start_time = self.timestamps[0]
            relative_times = [(t - start_time) / 1000.0 for t in self.timestamps]
            
            self.line.set_data(relative_times, self.ecg_values)
            self.ax.relim()
            self.ax.autoscale_view()
            
            # Update title with current stats
            self.ax.set_title(f'ECG Signal - Samples: {len(self.ecg_values)} | '
                            f'Range: {self.stats["max_value"] - self.stats["min_value"]} | '
                            f'Avg: {self.stats["avg_value"]:.1f}')
        
        return self.line,
    
    def start_plotting(self):
        """Start real-time plotting"""
        self.ax.set_xlabel('Time (seconds)')
        self.ax.set_ylabel('ECG Value')
        self.ax.set_title('Real-time ECG Signal')
        self.ax.grid(True, alpha=0.3)
        
        # Start animation
        self.ani = animation.FuncAnimation(
            self.fig, self.animate_plot, interval=100, blit=True
        )
        plt.show()

def main():
    """Main function with interactive menu"""
    print("ECG Data Reader")
    print("===============")
    
    # Get serial port from user
    port = input("Enter serial port (e.g., COM3, /dev/ttyUSB0): ").strip()
    if not port:
        port = 'COM3'  # Default for Windows
    
    # Create ECG reader
    reader = ECGReader(port=port)
    
    try:
        # Start reading data
        if reader.start_reading():
            print("\nCommands:")
            print("  'p' - Start plotting")
            print("  's' - Save data")
            print("  't' - Print statistics")
            print("  'q' - Quit")
            print("\nPress Enter to start plotting...")
            
            while True:
                cmd = input().strip().lower()
                
                if cmd == 'p':
                    print("Starting real-time plot...")
                    reader.start_plotting()
                elif cmd == 's':
                    filename = input("Enter filename (or press Enter for auto): ").strip()
                    if not filename:
                        filename = None
                    reader.save_data(filename)
                elif cmd == 't':
                    reader.print_stats()
                elif cmd == 'q':
                    break
                else:
                    print("Unknown command. Use 'p', 's', 't', or 'q'")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        reader.stop_reading()
        reader.disconnect()
        print("Goodbye!")

if __name__ == "__main__":
    main() 