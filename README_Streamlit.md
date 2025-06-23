# Enhanced ECG Dashboard - Streamlit Application

## Overview

This enhanced Streamlit dashboard provides real-time ECG signal comparison and convergence analysis using advanced ECG processing tools. It demonstrates how to use PID control loops to iteratively modify one ECG signal to converge toward another.

## Features

### 🎯 **Real-Time Signal Comparison**
- **Human ECG**: Baseline normal ECG signal
- **Organoid ECG**: Target signal (normal, afib, or tachycardia)
- **Converged Signal**: PID-controlled signal that converges toward target

### 🔬 **Advanced Analysis Tools**
- **Morphology Correlation**: Compare ECG waveform shapes
- **Interval Analysis**: PR, QRS, QT interval measurements
- **Deviation Area**: Quantitative measure of signal differences
- **PID Control Loop**: Real-time chemical modulation simulation

### 📊 **Comprehensive Visualization**
- **Main Comparison Plot**: All three signals overlaid with deviation areas
- **Convergence Progress**: Real-time tracking of deviation reduction
- **Correlation Analysis**: Heartbeat comparison and correlation over time
- **Control System Dashboard**: PID gains, chemical concentrations, error signals

### 🎛️ **Interactive Controls**
- **Signal Mode Selection**: Choose between normal, afib, or tachycardia
- **PID Parameter Tuning**: Adjust Kp, Ki, Kd gains in real-time
- **Analysis Options**: Toggle interval analysis and correlation plots

## Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Verify Installation
```bash
python test_streamlit.py
```

### 3. Run the Dashboard
```bash
streamlit run streamlit-app.py
```

## Usage

### Starting the Dashboard
1. Click **▶️ Start** to begin real-time analysis
2. Select the **Organoid Signal Type** (normal, afib, tachycardia)
3. Adjust **PID Control** parameters as needed
4. Toggle **Analysis Settings** for additional visualizations

### Understanding the Plots

#### **Main Comparison Plot**
- **Blue line**: Human ECG (baseline)
- **Red line**: Organoid ECG (target)
- **Green line**: Converged signal (controlled)
- **Purple shaded area**: Deviation between human and converged signals

#### **Convergence Progress**
- Shows how the deviation area decreases over time
- Target is zero deviation (perfect match)
- Demonstrates PID control effectiveness

#### **Correlation Analysis**
- **Left plot**: Individual heartbeat comparison
- **Right plot**: Correlation coefficient over time
- Higher correlation = better convergence

#### **Control System Dashboard**
- **Chemical Concentrations**: Activator (A) and Inhibitor (B) levels
- **Error Signal**: Deviation area over time
- **PID Gains**: Current control parameters
- **Control Status**: Active/Stable based on error threshold

### PID Control Parameters

- **Kp (Proportional)**: Responds to current error
  - Higher values = faster response but potential overshoot
  - Lower values = slower response but more stable

- **Ki (Integral)**: Eliminates steady-state error
  - Higher values = faster elimination of persistent error
  - Lower values = slower elimination but less oscillation

- **Kd (Derivative)**: Reduces overshoot and oscillation
  - Higher values = more damping, less oscillation
  - Lower values = less damping, potential overshoot

## Technical Details

### Signal Generation
The dashboard uses realistic ECG signal generation with:
- **Fundamental frequency**: Base heart rate
- **Harmonics**: Realistic ECG waveform shape
- **Noise**: Simulated measurement noise
- **Irregularity**: For afib simulation

### PID Control Algorithm
```python
error = deviation_area(human_signal, target_signal)
pid_output = Kp*error + Ki*integral + Kd*derivative
chemical_a += pid_output * 0.5  # Activator
chemical_b -= pid_output * 0.5  # Inhibitor
```

### Chemical Modulation Model
- **Chemical A (Activator)**: Increases ECG amplitude and frequency
- **Chemical B (Inhibitor)**: Decreases ECG amplitude and frequency
- **Dynamic Response**: Real-time simulation of chemical effects

## Applications

### 🧬 **Drug Development**
- Test chemical compounds for arrhythmia treatment
- Optimize drug dosages for individual patients
- Understand chemical effects on cardiac electrophysiology

### 🔬 **Research Applications**
- Study signal convergence in biological systems
- Develop control strategies for cardiac modulation
- Validate mathematical models of cardiac activity

### 📚 **Educational Use**
- Demonstrate control theory in biological systems
- Visualize PID control concepts
- Show real-time signal processing applications

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install -r requirements.txt
   python test_streamlit.py
   ```

2. **Streamlit Not Found**
   ```bash
   pip install streamlit
   ```

3. **ECG Module Import Errors**
   - Ensure the `ecg` folder is in the same directory as `streamlit-app.py`
   - Check that all ECG helper files are present

4. **Performance Issues**
   - Reduce `BUFFER_SECONDS` in the configuration
   - Lower the sampling rate `FS`
   - Close other applications to free up resources

### Performance Optimization

- **Buffer Size**: Adjust `BUFFER_SECONDS` for memory usage vs. visualization quality
- **Sampling Rate**: Lower `FS` for better performance, higher for better resolution
- **Update Frequency**: Modify the sleep time in the main loop

## File Structure

```
├── streamlit-app.py          # Main Streamlit application
├── requirements.txt          # Python dependencies
├── test_streamlit.py         # Installation verification script
├── README_Streamlit.md       # This documentation
└── ecg/                      # ECG analysis modules
    ├── ecg_morphology_correlator.py
    ├── ecg_analyzer.py
    ├── synthetic_ecg_generator.py
    ├── ecg_pid_control.py
    └── ...
```

## Contributing

To extend the dashboard:

1. **Add New Signal Types**: Modify `get_next_sample()` function
2. **Enhance Analysis**: Add new analysis functions and plots
3. **Improve Control**: Modify PID control algorithm or add new control strategies
4. **Add Real Data**: Replace synthetic data generation with real ECG acquisition

## License

This project is part of the Couture Organoids ECG analysis toolkit.

---

**Note**: This dashboard is designed for educational and research purposes. For clinical applications, ensure proper validation and regulatory compliance. 