#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pca9685 = Adafruit_PWMServoDriver(0x40);

#define SERVO_MIN_US  500
#define SERVO_MAX_US  2500
#define PWM_FREQ      50
#define NUM_SERVOS    5

const uint8_t servoChannels[NUM_SERVOS] = {0, 1, 2, 3, 4};

#define SWEEP_STEP    5
#define SWEEP_DELAY   20

enum RobotState {
  INIT,
  STATE_1,
  STATE_2,
  STATE_3,
  STATE_4,
  STATE_5,
  STATE_6,
  STATE_7,
  STATE_8,
  DONE
};

RobotState currentState = INIT;
bool sequenceDone = false;

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

void setServoAngle(uint8_t channel, uint8_t angle) {
  uint16_t pulse = angleToPulse(angle);
  pca9685.setPWM(channel, 0, pulse);
}

void sweepServo(uint8_t channel, uint8_t startAngle, uint8_t endAngle) {
  if (startAngle < endAngle) {
    for (int a = startAngle; a <= endAngle; a += SWEEP_STEP) {
      setServoAngle(channel, constrain(a, 0, 180));
      delay(SWEEP_DELAY);
    }
  } else {
    for (int a = startAngle; a >= endAngle; a -= SWEEP_STEP) {
      setServoAngle(channel, constrain(a, 0, 180));
      delay(SWEEP_DELAY);
    }
  }
  setServoAngle(channel, endAngle);
}

void sweepTwoServos(uint8_t ch1, uint8_t ch2, uint8_t start1, uint8_t end1, uint8_t start2, uint8_t end2) {
  int steps = max(abs(end1 - start1), abs(end2 - start2)) / SWEEP_STEP;
  if (steps == 0) steps = 1;

  for (int i = 0; i <= steps; i++) {
    uint8_t a1 = map(i, 0, steps, start1, end1);
    uint8_t a2 = map(i, 0, steps, start2, end2);
    setServoAngle(ch1, a1);
    setServoAngle(ch2, a2);
    delay(SWEEP_DELAY);
  }
  setServoAngle(ch1, end1);
  setServoAngle(ch2, end2);
}

// ===== States =====

// NOTE: your INIT function print message didn’t match action.
// I kept your *action* as written.
void runINIT() {
  Serial.println("[INIT] All servos → 0 degrees (except servo 0 → 45°)");

  for (uint8_t i = 1; i < NUM_SERVOS; i++) {
    setServoAngle(servoChannels[i], 0);
  }
  setServoAngle(servoChannels[0], 45); //so gripper wont hit the basket.
  delay(1500);
}

void runState1() {
  Serial.println("[STATE 1] Servo 1  → 0 to 90 degrees & Servo 2 → 0 to 110 degrees");
  //basket position, need to be fixed.
  sweepServo(servoChannels[1], 0, 90);
  sweepServo(servoChannels[2], 0, 70);

  delay(1500);
}

void runState2() {
  Serial.println("[STATE 2] Servo 3 → 0 to 90 degrees");
  sweepServo(servoChannels[3], 0, 90); //rotate gripper
  delay(1500);
}

void runState3() {
  Serial.println("[STATE 3] Servo 4 → 0 to 180 degrees");
  sweepServo(servoChannels[4], 0, 180); //open gripper
  delay(1500);
}

void runState4() {
  Serial.println("[STATE 4] Servo 0 → 45 to 0 degrees");
  
  sweepServo(servoChannels[0], 45, 0); //To basket's postion.
  delay(1500);
}

void runState5() {
  Serial.println("[STATE 5] Servo 4 → 180 back to 0 degrees");
  sweepServo(servoChannels[4], 180, 0);
  delay(1500);
}

// drag
void runState6() {
  Serial.println("[STATE 6] Coordinated Drag: Servos 1 & 2 → 0 degrees");
  // Coordinated drag pulls both joints simultaneously for more power/reach
  // Starting from a slightly wider position (90, 70) and pulling to 0
  sweepTwoServos(servoChannels[1], servoChannels[2], 90, 0, 70, 0);
  delay(1500);
}

void runState7() {
  Serial.println("[STATE 7] Servo 0 → 0 back to 90 degrees");
  sweepServo(servoChannels[0], 0, 90);// position for user
  delay(1500);
}

void runState8() {
  Serial.println("[STATE 8] Servo 4 → 0 back to 90 degrees");
  sweepServo(servoChannels[4], 0, 180);//open gripper for user to take the item.
  delay(1500);
}

void setup() {
  Serial.begin(9600);
  Serial.println("=================================");
  Serial.println(" PCA9685 — 5 Servo State Machine");
  Serial.println(" Runs ONCE then stops");
  Serial.println(" SDA = A4 | SCL = A5");
  Serial.println("=================================");

  Wire.begin();
  pca9685.begin();
  pca9685.setPWMFreq(PWM_FREQ);

  delay(100);

  // Start all servos at 0 degrees on boot
 
  currentState = INIT;
  sequenceDone = false;
}

void loop() {
  if (sequenceDone) {
    // Stop forever after one full cycle
    // (servos hold last position)
    while (true) { delay(1000); }
  }

  switch (currentState) {
    case INIT:
      runINIT();
      currentState = STATE_1;
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
      currentState = STATE_7;
      break;

    case STATE_7:
      runState7();
      currentState = STATE_8;
      break;
    case STATE_8:
      runState8();
      currentState = DONE;
      break;

    case DONE:
      
      Serial.println("DONE");
      sequenceDone = true;
      break;
  }
}
