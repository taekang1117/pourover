#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// ================================================================
//  PCA9685 SERVO DRIVER — STATE MACHINE (5 SERVOS)
//  Arduino UNO R3: SDA = A4 | SCL = A5
//  Channels: Servo 0 = CH0, Servo 1 = CH1, ... Servo 4 = CH4
// ================================================================

Adafruit_PWMServoDriver pca9685 = Adafruit_PWMServoDriver(0x40);

// --- Servo pulse width limits in microseconds ---
#define SERVO_MIN_US  500
#define SERVO_MAX_US  2500
#define PWM_FREQ      50
#define NUM_SERVOS    5

// --- PCA9685 Channel assignments ---
const uint8_t servoChannels[NUM_SERVOS] = {
  0,   // Servo 0 → CH0
  1,   // Servo 1 → CH1
  2,   // Servo 2 → CH2
  3,   // Servo 3 → CH3
  4    // Servo 4 → CH4
};

// ================================================================
//  SWEEP SPEED CONTROL
//  Adjust these to make servos move faster or slower
// ================================================================
#define SWEEP_STEP    5     // Degrees per step
#define SWEEP_DELAY   20    // Milliseconds between each step

// ================================================================
//  STATE DEFINITIONS
//  Each state is a named constant for readability.
//  The machine runs top to bottom, then resets to INIT.
// ================================================================
enum RobotState {
  INIT,     // All servos → 0°
  STATE_1,  // Servo 1 & 2 → 0° to 90°
  STATE_2,  // Servo 3 → 0° to 90°
  STATE_3,  // Servo 4 → 0° to 180°
  STATE_4,  // Servo 1 & 2 → 90° back to 0°
  STATE_5,  // Servo 0 → 0° to 180°
  STATE_6   // Servo 4 → 180° back to 0°
};

// --- Current state variable ---
RobotState currentState = INIT;

// ================================================================
//  HELPER FUNCTIONS
// ================================================================

// Converts microseconds → PCA9685 tick count
uint16_t usToPulse(uint16_t us) {
  return (uint16_t)((us / 20000.0) * 4096);
}

// Converts angle (0–180°) → PCA9685 tick count
uint16_t angleToPulse(uint8_t angle) {
  angle = constrain(angle, 0, 180);
  uint16_t us = map(angle, 0, 180, SERVO_MIN_US, SERVO_MAX_US);
  return usToPulse(us);
}

// Moves a single servo to the given angle instantly (no sweep)
void setServoAngle(uint8_t channel, uint8_t angle) {
  uint16_t pulse = angleToPulse(angle);
  pca9685.setPWM(channel, 0, pulse);
}

// ================================================================
//  SWEEP FUNCTION — sweeps ONE servo from startAngle to endAngle
//  Parameters:
//    channel    = PCA9685 channel (0–15)
//    startAngle = angle to begin at
//    endAngle   = angle to end at
//  Handles both forward (0→90) and backward (90→0) automatically.
// ================================================================
void sweepServo(uint8_t channel, uint8_t startAngle, uint8_t endAngle) {
  if (startAngle < endAngle) {
    // Sweeping forward (increasing angle)
    for (int angle = startAngle; angle <= endAngle; angle += SWEEP_STEP) {
      setServoAngle(channel, constrain(angle, 0, 180));
      delay(SWEEP_DELAY);
    }
  } else {
    // Sweeping backward (decreasing angle)
    for (int angle = startAngle; angle >= endAngle; angle -= SWEEP_STEP) {
      setServoAngle(channel, constrain(angle, 0, 180));
      delay(SWEEP_DELAY);
    }
  }
  // Ensure the servo lands exactly on the final angle
  setServoAngle(channel, endAngle);
}

// ================================================================
//  SWEEP FUNCTION — sweeps TWO servos together simultaneously
//  Both servos move in sync, step by step, at the same time.
//  Parameters:
//    channel1/2    = PCA9685 channels for each servo
//    startAngle    = angle both servos begin at
//    endAngle      = angle both servos end at
// ================================================================
void sweepTwoServos(uint8_t channel1, uint8_t channel2,
                    uint8_t startAngle, uint8_t endAngle) {
  if (startAngle < endAngle) {
    // Sweeping forward
    for (int angle = startAngle; angle <= endAngle; angle += SWEEP_STEP) {
      setServoAngle(channel1, constrain(angle, 0, 180));
      setServoAngle(channel2, constrain(angle, 0, 180));
      delay(SWEEP_DELAY);
    }
  } else {
    // Sweeping backward
    for (int angle = startAngle; angle >= endAngle; angle -= SWEEP_STEP) {
      setServoAngle(channel1, constrain(angle, 0, 180));
      setServoAngle(channel2, constrain(angle, 0, 180));
      delay(SWEEP_DELAY);
    }
  }
  // Ensure both servos land exactly on the final angle
  setServoAngle(channel1, endAngle);
  setServoAngle(channel2, endAngle);
}

// ================================================================
//  STATE MACHINE FUNCTIONS
//  Each state is its own function. Clean, readable, easy to edit.
// ================================================================

// --- INIT: Move ALL 5 servos to 0 degrees ---
void runINIT() {
  Serial.println("[INIT] All servos → 0 degrees");
  for (uint8_t i = 0; i < NUM_SERVOS; i++) {
    setServoAngle(servoChannels[i], 0);
  }
  delay(1000); // Hold at INIT for 1 second before moving to State 1
}

// --- STATE 1: Servo 1 and Servo 2 sweep from 0° to 90° together ---
void runState1() {
  Serial.println("[STATE 1] Servo 1 & 2 → 0 to 90 degrees");
  sweepTwoServos(servoChannels[1], servoChannels[2], 0, 90);
  delay(500); // Short pause after state completes
}

// --- STATE 2: Servo 3 sweeps from 0° to 90° ---
void runState2() {
  Serial.println("[STATE 2] Servo 3 → 0 to 90 degrees");
  sweepServo(servoChannels[3], 0, 90);
  delay(500);
}

// --- STATE 3: Servo 4 sweeps from 0° to 180° ---
void runState3() {
  Serial.println("[STATE 3] Servo 4 → 0 to 180 degrees");
  sweepServo(servoChannels[4], 0, 180);
  delay(500);
}

// --- STATE 4: Servo 1 and Servo 2 sweep from 90° back to 0° together ---
void runState4() {
  Serial.println("[STATE 4] Servo 1 & 2 → 90 back to 0 degrees");
  sweepTwoServos(servoChannels[1], servoChannels[2], 90, 0);
  delay(500);
}

// --- STATE 5: Servo 0 sweeps from 0° to 180° ---
void runState5() {
  Serial.println("[STATE 5] Servo 0 → 0 to 180 degrees");
  sweepServo(servoChannels[0], 0, 180);
  delay(500);
}

// --- STATE 6: Servo 4 sweeps from 180° back to 0° ---
void runState6() {
  Serial.println("[STATE 6] Servo 4 → 180 back to 0 degrees");
  sweepServo(servoChannels[4], 180, 0);
  delay(500);
}


// ================================================================
//  SETUP — runs once on power on or reset
// ================================================================
void setup() {
  Serial.begin(9600);
  Serial.println("=================================");
  Serial.println(" PCA9685 — 5 Servo State Machine");
  Serial.println(" SDA = A4 | SCL = A5");
  Serial.println("=================================");

  Wire.begin();                  // Start I2C (SDA=A4, SCL=A5 on UNO R3)
  pca9685.begin();               // Initialize PCA9685
  pca9685.setPWMFreq(PWM_FREQ);  // Set 50Hz PWM for standard servos

  delay(100); // Let PCA9685 stabilize

  // Start all servos at 0 degrees immediately on boot
  for (uint8_t i = 0; i < NUM_SERVOS; i++) {
    setServoAngle(servoChannels[i], 0);
  }

  delay(1000);
  currentState = INIT; // Set starting state
}


// ================================================================
//  LOOP — runs the state machine forever
//  Each iteration of loop() executes one state, then advances
//  to the next. After STATE_6, it resets back to INIT.
// ================================================================
void loop() {

  switch (currentState) {

    case INIT:
      runINIT();
      currentState = STATE_1;   // Advance to next state
      break;

    case STATE_1:
      runState1();
      currentState = STATE_2;
      break;

    case STATE_2:
      runState2();
      currentState = STATE_3;
      break;

    case STATE_3:
      runState3();
      currentState = STATE_4;
      break;

    case STATE_4:
      runState4();
      currentState = STATE_5;
      break;

    case STATE_5:
      runState5();
      currentState = STATE_6;
      break;

    case STATE_6:
      runState6();
      currentState = INIT;      // Cycle complete → back to INIT
      break;
  }
}
