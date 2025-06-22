/*
 * Simple ECG Reader Arduino Code
 * Basic analog ECG reading from pin A0
 * Minimal processing for raw data output
 */

const int ECG_PIN = A0;           // Analog pin for ECG sensor
const int LED_PIN = 13;           // Built-in LED for status

// Timing
const unsigned long SAMPLE_RATE = 50;  // Sample every 50ms (20 Hz)
unsigned long lastSampleTime = 0;

void setup() {
  // Initialize serial communication
  Serial.begin(9600);
  Serial.println("Simple ECG Reader Started");
  
  // Initialize pins
  pinMode(LED_PIN, OUTPUT);
  pinMode(ECG_PIN, INPUT);
  
  Serial.println("Reading ECG from pin A0...");
  Serial.println("Format: Time,Value");
}

void loop() {
  unsigned long currentTime = millis();
  
  // Sample at specified rate
  if (currentTime - lastSampleTime >= SAMPLE_RATE) {
    lastSampleTime = currentTime;
    
    // Read ECG value
    int ecgValue = analogRead(ECG_PIN);
    
    // Output timestamp and value
    Serial.print(currentTime);
    Serial.print(",");
    Serial.println(ecgValue);
    
    // Simple LED indicator (blinks with signal changes)
    if (ecgValue > 512) {  // Threshold at mid-range
      digitalWrite(LED_PIN, HIGH);
    } else {
      digitalWrite(LED_PIN, LOW);
    }
  }
} 