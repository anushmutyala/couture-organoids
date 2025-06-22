/*
 * ECG Reader Arduino Code
 * Reads analog ECG signal from pin A0
 * Provides real-time ECG data output via Serial
 * Includes basic signal processing and filtering
 */

// Pin definitions
const int ECG_PIN = A0;           // Analog pin for ECG sensor
const int LED_PIN = 13;           // Built-in LED for status indication

// Timing variables
const unsigned long SAMPLE_RATE = 100;  // Sample rate in milliseconds (10 Hz)
unsigned long lastSampleTime = 0;

// Signal processing variables
const int BUFFER_SIZE = 50;       // Circular buffer size for averaging
int ecgBuffer[BUFFER_SIZE];       // Circular buffer for ECG values
int bufferIndex = 0;              // Current position in buffer
int bufferSum = 0;                // Sum of all values in buffer
bool bufferFilled = false;        // Flag to indicate if buffer is full

// Calibration variables
int baselineValue = 0;            // Baseline ECG value
const int CALIBRATION_SAMPLES = 100;  // Number of samples for baseline calibration
int minValue = 1023;              // Minimum value observed
int maxValue = 0;                 // Maximum value observed

// Threshold for heartbeat detection
const int HEARTBEAT_THRESHOLD = 50;  // Minimum change to detect heartbeat
int lastPeakValue = 0;            // Last peak value detected
unsigned long lastPeakTime = 0;   // Time of last peak
unsigned long heartbeatInterval = 0;  // Interval between heartbeats

void setup() {
  // Initialize serial communication
  Serial.begin(115200);
  Serial.println("ECG Reader Initialized");
  Serial.println("Starting calibration...");
  
  // Initialize LED pin
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
  
  // Initialize analog pin
  pinMode(ECG_PIN, INPUT);
  
  // Perform baseline calibration
  calibrateBaseline();
  
  Serial.println("Calibration complete!");
  Serial.println("Baseline value: " + String(baselineValue));
  Serial.println("Starting ECG monitoring...");
  Serial.println("Format: Raw,Filtered,Peak,HeartRate");
}

void loop() {
  unsigned long currentTime = millis();
  
  // Sample at specified rate
  if (currentTime - lastSampleTime >= SAMPLE_RATE) {
    lastSampleTime = currentTime;
    
    // Read ECG value
    int rawValue = analogRead(ECG_PIN);
    
    // Update min/max tracking
    if (rawValue < minValue) minValue = rawValue;
    if (rawValue > maxValue) maxValue = rawValue;
    
    // Apply moving average filter
    int filteredValue = applyMovingAverage(rawValue);
    
    // Detect heartbeat peaks
    bool isPeak = detectHeartbeat(filteredValue, currentTime);
    
    // Calculate heart rate
    int heartRate = calculateHeartRate();
    
    // Output data
    outputData(rawValue, filteredValue, isPeak, heartRate);
    
    // Update LED status
    updateLED(isPeak);
  }
}

void calibrateBaseline() {
  long sum = 0;
  
  for (int i = 0; i < CALIBRATION_SAMPLES; i++) {
    int value = analogRead(ECG_PIN);
    sum += value;
    delay(10);  // Small delay between samples
  }
  
  baselineValue = sum / CALIBRATION_SAMPLES;
  
  // Initialize buffer with baseline value
  for (int i = 0; i < BUFFER_SIZE; i++) {
    ecgBuffer[i] = baselineValue;
    bufferSum += baselineValue;
  }
  bufferFilled = true;
}

int applyMovingAverage(int newValue) {
  // Remove old value from sum
  bufferSum -= ecgBuffer[bufferIndex];
  
  // Add new value
  ecgBuffer[bufferIndex] = newValue;
  bufferSum += newValue;
  
  // Update buffer index
  bufferIndex = (bufferIndex + 1) % BUFFER_SIZE;
  
  // Return average
  return bufferSum / BUFFER_SIZE;
}

bool detectHeartbeat(int filteredValue, unsigned long currentTime) {
  // Check if we have a significant peak above baseline
  int signalChange = filteredValue - baselineValue;
  
  if (signalChange > HEARTBEAT_THRESHOLD && 
      filteredValue > lastPeakValue && 
      currentTime - lastPeakTime > 200) {  // Minimum 200ms between peaks
    
    lastPeakValue = filteredValue;
    lastPeakTime = currentTime;
    
    // Calculate heartbeat interval
    if (lastPeakTime > 0) {
      heartbeatInterval = currentTime - lastPeakTime;
    }
    
    return true;
  }
  
  // Reset peak detection if signal drops
  if (signalChange < -HEARTBEAT_THRESHOLD) {
    lastPeakValue = 0;
  }
  
  return false;
}

int calculateHeartRate() {
  if (heartbeatInterval > 0) {
    // Convert interval to heart rate (beats per minute)
    return 60000 / heartbeatInterval;  // 60000ms = 1 minute
  }
  return 0;
}

void outputData(int rawValue, int filteredValue, bool isPeak, int heartRate) {
  // Output in CSV format for easy parsing
  Serial.print(rawValue);
  Serial.print(",");
  Serial.print(filteredValue);
  Serial.print(",");
  Serial.print(isPeak ? "1" : "0");
  Serial.print(",");
  Serial.println(heartRate);
  
  // Alternative: Output in a more readable format
  /*
  Serial.print("Raw: ");
  Serial.print(rawValue);
  Serial.print(" | Filtered: ");
  Serial.print(filteredValue);
  Serial.print(" | Peak: ");
  Serial.print(isPeak ? "YES" : "NO");
  Serial.print(" | HR: ");
  Serial.print(heartRate);
  Serial.println(" BPM");
  */
}

void updateLED(bool isPeak) {
  if (isPeak) {
    digitalWrite(LED_PIN, HIGH);
    delay(50);  // Brief flash
    digitalWrite(LED_PIN, LOW);
  }
}

// Optional: Function to get signal statistics
void printSignalStats() {
  Serial.println("=== Signal Statistics ===");
  Serial.print("Baseline: ");
  Serial.println(baselineValue);
  Serial.print("Min Value: ");
  Serial.println(minValue);
  Serial.print("Max Value: ");
  Serial.println(maxValue);
  Serial.print("Signal Range: ");
  Serial.println(maxValue - minValue);
  Serial.println("========================");
} 