#!/usr/bin/env python3
"""
List available serial ports
Helps identify the correct COM port for Arduino
"""

import serial.tools.list_ports

def list_serial_ports():
    """List all available serial ports"""
    ports = serial.tools.list_ports.comports()
    
    if not ports:
        print("No serial ports found!")
        return
    
    print("Available serial ports:")
    print("=======================")
    
    for port in ports:
        print(f"Port: {port.device}")
        print(f"  Description: {port.description}")
        print(f"  Hardware ID: {port.hwid}")
        print(f"  Manufacturer: {port.manufacturer}")
        print(f"  Product: {port.product}")
        print("---")
    
    # Look for Arduino specifically
    arduino_ports = []
    for port in ports:
        if ('arduino' in port.description.lower() if port.description else False) or \
           ('arduino' in port.manufacturer.lower() if port.manufacturer else False):
            arduino_ports.append(port.device)
    
    if arduino_ports:
        print(f"Arduino devices found: {arduino_ports}")
    else:
        print("No Arduino devices detected automatically")

if __name__ == "__main__":
    list_serial_ports() 