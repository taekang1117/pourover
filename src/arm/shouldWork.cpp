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
// MOTION SETTINGS
// =====================
#define SWEEP_STEP    2
#define SWEEP_DELAY   15
#define STATE_DELAY   1000   // 1 sec between states

// =====================
// FSM
// =====================
enum RobotState {
  ST_INIT,
  ST_OPEN,
  ST_MOVE1,
  ST_CLOSE,
  ST_MOVE2,
  ST_DONE
};

RobotState state = ST_INIT;
bool stateRan = false;
unsigned long tState = 0;

// Track current servo angles (for sweeps)
uint8_t cur[NUM_SERVOS] = {0, 0, 0, 0, 0};

// ===== Option A: keep outputs OFF at boot =====
void allServosOff() {
  for (uint8_t i = 0; i < NUM_SERVOS; i++) {
    // 4096 = full OFF in Adafruit PCA9685 library
    pca9685.setPWM(servoChannels[i], 0, 4096);
  }
}

// -------- Helpers --------
uint16_t usToTicks(uint16_t us) {
  return (uint16_t)((us / 20000.0) * 4096.0);  // 20ms frame @ 50Hz
}

uint16_t angleToTicks(uint8_t angle) {
  angle = constrain(angle, 0, 180);
  uint16_t us = map(angle, 0, 180, SERVO_MIN_US, SERVO_MAX_US);
  return usToTicks(us);
}

void setServoAngle(uint8_t i, uint8_t angle) {
  if (i >= NUM_SERVOS) return;
  angle = constrain(angle, 0, 180);
  pca9685.setPWM(servoChannels[i], 0, angleToTicks(angle));
  cur[i] = angle;
}

void sweepServoTo(uint8_t i, uint8_t target) {
  if (i >= NUM_SERVOS) return;
  target = constrain(target, 0, 180);

  uint8_t start = cur[i];
  if (start == target) {
    setServoAngle(i, target);
    return;
  }

  if (start < target) {
    for (int a = start; a <= target; a += SWEEP_STEP) {
      setServoAngle(i, (uint8_t)a);
      delay(SWEEP_DELAY);
    }
  } else {
    for (int a = start; a >= target; a -= SWEEP_STEP) {
      setServoAngle(i, (uint8_t)a);
      delay(SWEEP_DELAY);
    }
  }
  setServoAngle(i, target);
}

void moveInitPose() {
  Serial.println("[INIT] s0=40 s1=70 s2=10 s3=80 s4=0");
  sweepServoTo(0, 40);
  sweepServoTo(1, 170);
  sweepServoTo(2, 0);
  sweepServoTo(3, 100);
  sweepServoTo(4, 65);
}

void doOpen() {
  Serial.println("[OPEN] s4=70");
  sweepServoTo(4, 70);
}

void doMove1() {
  Serial.println("[MOVE1] s0=0");
  sweepServoTo(0, 0);
}

void doClose() {
  Serial.println("[CLOSE] s4=40");
  sweepServoTo(4, 40);
}

void doMove2() {
  Serial.println("[MOVE2] s0=90");
  sweepServoTo(0, 90);
}

void runStateOnce() {
  switch (state) {
    case ST_INIT:  moveInitPose(); break;
    case ST_OPEN:  doOpen();       break;
    case ST_MOVE1: doMove1();      break;
    case ST_CLOSE: doClose();      break;
    case ST_MOVE2: doMove2();      break;
    case ST_DONE:  Serial.println("[DONE] Stop"); break;
  }
}

void advanceState() {
  switch (state) {
    case ST_INIT:  state = ST_OPEN;  break;
    case ST_OPEN:  state = ST_MOVE1; break;
    case ST_MOVE1: state = ST_CLOSE; break;
    case ST_CLOSE: state = ST_MOVE2; break;
    case ST_MOVE2: state = ST_DONE;  break;
    case ST_DONE:  break;
  }
}

void setup() {
  Serial.begin(9600);
  Serial.println("PCA9685 FSM Sequence");

  Wire.begin();
  pca9685.begin();
  pca9685.setPWMFreq(PWM_FREQ);
  delay(50);

  // Option A: disable outputs first to reduce startup twitch
  allServosOff();
  delay(500);

  // IMPORTANT: set cur[] to the FIRST pose targets so sweeps are deterministic
  cur[0] = 40; cur[1] = 70; cur[2] = 10; cur[3] = 80; cur[4] = 0;

  state = ST_INIT;
  stateRan = false;
  tState = millis();
}

void loop() {
  if (state == ST_DONE) {
    // Hold final PWM values (or you can allServosOff() here if you want limp)
    while (true) { delay(1000); }
  }

  // Run state action once
  if (!stateRan) {
    runStateOnce();
    stateRan = true;
    tState = millis();
  }

  // Wait 1 second then advance
  if (millis() - tState >= STATE_DELAY) {
    advanceState();
    stateRan = false;
  }
}
