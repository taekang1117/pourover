#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// =====================
// PCA9685 CONFIG
// =====================
Adafruit_PWMServoDriver pca9685(0x40);

#define PWM_FREQ      50
#define NUM_SERVOS    5
const uint8_t servoChannels[NUM_SERVOS] = {0, 1, 2, 3, 4};

// =====================
// SERVO CALIBRATION
// =====================
#define SERVO_MIN_US  500
#define SERVO_MAX_US  2500

// =====================
// JOG SETTINGS
// =====================
#define STEP_DEG      10   // degrees per key press

// H key -> these angles
uint8_t homeAngles[NUM_SERVOS] = {40, 70, 0, 110, 70};
uint8_t cur[NUM_SERVOS]        = {40, 70, 0, 110, 70};

// Option A: keep outputs OFF at boot
void allServosOff() {
  for (uint8_t i = 0; i < NUM_SERVOS; i++) {
    // 4096 = full OFF in Adafruit PCA9685 library
    pca9685.setPWM(servoChannels[i], 0, 4096);
  }
}

// Convert microseconds -> PCA9685 ticks (50Hz frame = 20000us)
uint16_t usToTicks(uint16_t us) {
  return (uint16_t)((us / 20000.0) * 4096.0);
}

uint16_t angleToTicks(uint8_t angle) {
  angle = constrain(angle, 0, 180);
  uint16_t us = map(angle, 0, 180, SERVO_MIN_US, SERVO_MAX_US);
  return usToTicks(us);
}

void applyServo(uint8_t i) {
  if (i >= NUM_SERVOS) return;
  uint8_t a = constrain(cur[i], 0, 180);
  pca9685.setPWM(servoChannels[i], 0, angleToTicks(a));
}

void setServo(uint8_t i, int angle) {
  if (i >= NUM_SERVOS) return;

  cur[i] = (uint8_t)constrain(angle, 0, 180);
  applyServo(i);

  Serial.print("S");
  Serial.print(i);
  Serial.print(" = ");
  Serial.print(cur[i]);
  Serial.println(" deg");
}

void homeAll() {
  Serial.println("HOME: s0=40 s1=70 s2=20 s3=80 s4=0");
  for (uint8_t i = 0; i < NUM_SERVOS; i++) {
    cur[i] = homeAngles[i];
    applyServo(i);
  }
}

void showStatus() {
  Serial.println("----- STATUS -----");
  for (uint8_t i = 0; i < NUM_SERVOS; i++) {
    Serial.print("S");
    Serial.print(i);
    Serial.print(" = ");
    Serial.print(cur[i]);
    Serial.println(" deg");
  }
  Serial.println("------------------");
}

void printHelp() {
  Serial.println("Keys:");
  Serial.println("  H : home (40,70,20,80,0)");
  Serial.println("  0..4 : +10 deg to servo 0..4");
  Serial.println("  Q/W/E/R/T : -10 deg to servo 0..4 (Q->0 W->1 E->2 R->3 T->4)");
  Serial.println("  S : show angles");
}

void handleKey(char c) {
  if (c == '\n' || c == '\r') return;

  // HOME
  if (c == 'H' || c == 'h') { homeAll(); return; }

  // SHOW
  if (c == 'S' || c == 's') { showStatus(); return; }

  // INCREASE with 0..4
  if (c >= '0' && c <= '4') {
    uint8_t i = (uint8_t)(c - '0');
    setServo(i, (int)cur[i] + STEP_DEG);
    return;
  }

  // DECREASE with QWERT (0..4)
  char up = c;
  if (up >= 'a' && up <= 'z') up = (char)(up - 'a' + 'A');

  const char decKeys[NUM_SERVOS] = {'Q', 'W', 'E', 'R', 'T'};
  for (uint8_t i = 0; i < NUM_SERVOS; i++) {
    if (up == decKeys[i]) {
      setServo(i, (int)cur[i] - STEP_DEG);
      return;
    }
  }

  Serial.print("Unknown key: ");
  Serial.println(c);
  printHelp();
}

void setup() {
  Serial.begin(9600);
  Serial.println("PCA9685 Servo Jog (Option A: outputs OFF at boot)");

  Wire.begin();
  pca9685.begin();
  pca9685.setPWMFreq(PWM_FREQ);
  delay(50);

  // Option A: disable outputs to prevent startup twitch
  allServosOff();
  delay(500);

  Serial.println("READY. Press H to move to HOME.");
  printHelp();
}

void loop() {
  if (Serial.available()) {
    char c = (char)Serial.read();
    handleKey(c);
  }
}
