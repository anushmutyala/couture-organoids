# 🎯 Live ECG Demo - Arduino + Synthetic + PID Control

A comprehensive live demonstration that combines real-time Arduino ECG streaming, synthetic organoid ECG generation, and PID-controlled morphology matching.

## 🎯 Demo Overview

This demo showcases a complete workflow for organoid ECG analysis:

1. **📊 Baseline Collection**: Receive 5 seconds of real ECG data from Arduino via Serial COM3
2. **🧬 Synthetic Generation**: Generate 5 seconds of synthetic "organoid" ECG with different morphology
3. **🎯 PID Convergence**: Live update the synthetic ECG to match baseline morphology using simulated chemical inputs
4. **📈 Real-time Visualization**: Streamlit dashboard with live plots and performance metrics

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Arduino IDE
- Arduino Uno (or compatible board)
- USB cable

### Installation

1. **Install Python dependencies**:
   ```bash
   pip install -r live_demo_requirements.txt
   ```

2. **Upload Arduino sketch**:
   - Open `ecg/arduino_ecg_simulator.ino` in Arduino IDE
   - Select your board and port (COM3 on Windows)
   - Upload the sketch

3. **Run the demo**:
   ```bash
   streamlit run live_ecg_demo.py
   ```

## 🔧 Demo Workflow

### Phase 1: Baseline Collection
- Connect Arduino to COM3
- Click "Collect Baseline (5s)" to gather real ECG data
- System analyzes baseline characteristics (heart rate, amplitude, etc.)

### Phase 2: Synthetic Generation
- Click "Generate Synthetic ECG" to create organoid ECG
- Synthetic signal has similar timing but different morphology
- Uses mathematical models to simulate realistic ECG patterns

### Phase 3: PID Convergence
- Click "Start Convergence" to begin PID control
- System continuously adjusts synthetic ECG to match baseline
- Uses simulated calcium and potassium ion concentrations as control inputs

## 🧪 Chemical Control System

The demo simulates two key ion concentrations that affect ECG morphology:

### Calcium (Ca²⁺) Concentration
- **Effect**: Increases amplitude and spike sharpness
- **Range**: 0-10 mM
- **PID Control**: Responds to deviation area error

### Potassium (K⁺) Concentration  
- **Effect**: Affects repolarization and baseline
- **Range**: 0-10 mM
- **PID Control**: Responds to deviation area error

## 📊 Dashboard Features

### Real-time Plots
- **Baseline ECG**: Raw Arduino data
- **Synthetic ECG**: Generated organoid signal
- **Converged ECG**: PID-controlled output

### Performance Metrics
- **Deviation Area**: Real-time error measurement
- **Convergence Trend**: Improving/worsening indicator
- **Chemical Gauges**: Visual concentration displays

### PID Parameters
- **Proportional (Kp)**: 0.1-2.0 (default: 0.8)
- **Integral (Ki)**: 0.01-0.5 (default: 0.2)  
- **Derivative (Kd)**: 0.01-0.3 (default: 0.1)

## 🔬 Technical Details

### Sampling Configuration
- **Rate**: 500 Hz
- **Baseline Duration**: 5 seconds
- **Buffer Duration**: 10 seconds
- **Max Points**: 5000 samples

### Signal Processing
- **Peak Detection**: Automatic heart rate estimation
- **Morphology Analysis**: Deviation area calculation
- **Chemical Effects**: Amplitude and sharpness modulation

### PID Control Loop
```
Error = Deviation_Area(Baseline, Converged)
Calcium = PID_Output(Error, Kp, Ki, Kd)
Potassium = PID_Output(Error, Kp, Ki, Kd)
Modified_Signal = Apply_Chemical_Effects(Synthetic, Calcium, Potassium)
```

## 🛠️ Troubleshooting

### Arduino Connection Issues
- Verify COM port in Device Manager
- Check baud rate (9600)
- Ensure Arduino is powered and connected

### Import Errors
- Check that all ECG modules are in the `ecg/` folder
- Verify Python path includes ECG directory
- Install missing dependencies

### Performance Issues
- Reduce sampling rate if needed
- Close other applications
- Check system resources

## 📁 File Structure

```
couture-organoids/
├── live_ecg_demo.py              # Main demo application
├── live_demo_requirements.txt    # Python dependencies
├── LIVE_DEMO_README.md          # This file
├── ecg/
│   ├── arduino_ecg_simulator.ino # Arduino sketch
│   ├── synthetic_ecg_generator.py
│   ├── ecg_pid_control.py
│   └── ... (other ECG modules)
└── ...
```

## 🎓 Educational Value

This demo demonstrates:

1. **Real-time Data Acquisition**: Arduino serial communication
2. **Signal Generation**: Mathematical ECG modeling
3. **Control Systems**: PID feedback loops
4. **Chemical Simulation**: Ion concentration effects
5. **Data Visualization**: Live plotting and metrics
6. **System Integration**: Multiple components working together

## 🔮 Future Enhancements

- Real ECG sensor integration
- More sophisticated chemical models
- Machine learning-based morphology matching
- Multi-channel ECG support
- Advanced arrhythmia simulation

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the ECG module documentation
3. Examine the error logs in the Streamlit console

---

**Note**: This is a demonstration system. For clinical applications, additional validation and safety measures would be required. 