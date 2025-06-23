/*
 * Arduino ECG Simulator
 * Simulates ECG data for testing the live demo
 * Sends timestamp,ecg_value pairs over Serial
 */

const int SAMPLING_RATE = 500;  // Hz
const unsigned long SAMPLE_INTERVAL = 1000000 / SAMPLING_RATE;  // microseconds
unsigned long lastSampleTime = 0;
unsigned long startTime = 0;

// ECG simulation parameters
float heartRate = 72.0;  // BPM
float time = 0.0;
float dt = 1.0 / SAMPLING_RATE;

void setup() {
  Serial.begin(9600);
  startTime = micros();
  Serial.println("ECG Simulator Started");
  Serial.println("Format: timestamp,ecg_value");
}

void loop() {
  unsigned long currentTime = micros();
  
  if (currentTime - lastSampleTime >= SAMPLE_INTERVAL) {
    // Calculate time since start
    unsigned long elapsed = currentTime - startTime;
    float timestamp = elapsed / 1000.0;  // Convert to milliseconds
    
    // Generate synthetic ECG value
    float ecgValue = generateECG(timestamp / 1000.0);  // Convert to seconds
    
    // Send data
    Serial.print(timestamp);
    Serial.print(",");
    Serial.println(ecgValue);
    
    lastSampleTime = currentTime;
  }
}

float generateECG(float t) {
  // Simple ECG simulation using mathematical functions
  float rr_interval = 60.0 / heartRate;
  float t_mod = fmod(t, rr_interval);
  
  // P wave (atrial depolarization)
  float p_peak = 0.16;  // PR interval
  float p_wave = 0.25 * exp(-pow(t_mod - p_peak, 2) / (2 * pow(0.02, 2)));
  
  // QRS complex (ventricular depolarization)
  float qrs = generateQRS(t_mod);
  
  // T wave (ventricular repolarization)
  float t_peak = 0.20;  // After QRS
  float t_wave = 0.35 * exp(-pow(t_mod - t_peak, 2) / (2 * pow(0.04, 2)));
  
  // Combine waves and add noise
  float ecg = p_wave + qrs + t_wave;
  float noise = 0.05 * (random(-100, 100) / 100.0);
  
  return ecg + noise;
}

float generateQRS(float t) {
  float qrs = 0.0;
  
  // Q wave (negative deflection)
  if (t >= 0.12 && t < 0.14) {
    qrs = -0.1 * sin(PI * (t - 0.12) / 0.02);
  }
  
  // R wave (positive peak)
  if (t >= 0.14 && t < 0.18) {
    qrs = 1.0 * exp(-pow(t - 0.16, 2) / (2 * pow(0.01, 2)));
  }
  
  // S wave (negative deflection)
  if (t >= 0.18 && t < 0.20) {
    qrs = -0.2 * sin(PI * (t - 0.18) / 0.02);
  }
  
  return qrs;
} 