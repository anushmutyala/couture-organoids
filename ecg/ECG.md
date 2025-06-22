# ECG Reader Arduino Code

This folder contains Arduino code for reading analog ECG signals from pin A0.

## Files

- `ecg_reader.ino` - Advanced ECG reader with signal processing, filtering, and heartbeat detection
- `ecg_simple.ino` - Simple ECG reader for basic analog data collection
- `simulation.ipynb` - Jupyter notebook for ECG data analysis and simulation

## Hardware Requirements

- Arduino board (Uno, Nano, Mega, etc.)
- ECG sensor module (e.g., AD8232 Heart Rate Monitor, Grove ECG Sensor)
- Connecting wires
- Optional: Breadboard for prototyping

## Wiring

### Basic Connection
```
ECG Sensor    Arduino
VCC    →      5V or 3.3V (check sensor specs)
GND    →      GND
OUT    →      A0 (Analog Pin 0)
```

### AD8232 Heart Rate Monitor (Common)
```
AD8232        Arduino
VCC    →      3.3V
GND    →      GND
OUTPUT →      A0
LO+   →       Digital Pin 2 (optional)
LO-   →       Digital Pin 3 (optional)
```

## Setup Instructions

1. **Connect Hardware**: Wire your ECG sensor to the Arduino according to the diagram above
2. **Upload Code**: 
   - Open Arduino IDE
   - Load either `ecg_reader.ino` or `ecg_simple.ino`
   - Select your Arduino board and port
   - Upload the code
3. **Open Serial Monitor**: 
   - Set baud rate to 115200 for `ecg_reader.ino`
   - Set baud rate to 9600 for `ecg_simple.ino`

## Code Features

### ecg_reader.ino (Advanced)
- **Real-time signal processing** with moving average filter
- **Automatic baseline calibration** on startup
- **Heartbeat detection** with peak detection algorithm
- **Heart rate calculation** in beats per minute
- **LED indicator** that flashes with each heartbeat
- **CSV output format** for easy data logging
- **Signal statistics** tracking (min/max values)

### ecg_simple.ino (Basic)
- **Simple analog reading** from pin A0
- **Timestamped data output** for basic logging
- **LED status indicator** based on signal threshold
- **Minimal processing** for raw data collection

## Output Format

### ecg_reader.ino
```
Raw,Filtered,Peak,HeartRate
512,510,0,0
515,512,0,0
520,515,1,72
```

### ecg_simple.ino
```
Time,Value
1234,512
1284,515
1334,520
```

## Data Analysis

Use the `simulation.ipynb` notebook to:
- Import and visualize ECG data
- Apply additional signal processing
- Analyze heart rate variability
- Create real-time plots

## Troubleshooting

1. **No signal**: Check wiring and sensor power
2. **Noisy data**: Ensure proper grounding and sensor placement
3. **Incorrect heart rate**: Adjust `HEARTBEAT_THRESHOLD` in advanced code
4. **Serial not working**: Verify baud rate matches code settings

## Customization

- **Sample rate**: Modify `SAMPLE_RATE` constant
- **Filter strength**: Adjust `BUFFER_SIZE` for moving average
- **Peak detection**: Change `HEARTBEAT_THRESHOLD` value
- **Output format**: Modify `outputData()` function

## Safety Notes

- This code is for educational and research purposes
- ECG sensors should be used according to manufacturer guidelines
- Ensure proper electrical safety when working with medical sensors
- This is not a medical device and should not be used for clinical diagnosis 