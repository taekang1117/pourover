// ========== Optical Sorting: LCWS + Robotic Arm (WebSocket GUI protocol) ==========
//
// Serial protocol (Pi <-> Arduino)
//   Pi -> Arduino:
//     g       : handshake (Pi ready)
//     t       : tare
//     r       : request one weighing cycle
//     a       : allow robotic arm to move (optional)
//     w,<g>   : update target weight in grams
//
//   Arduino -> Pi (LCWS part):
//     READY
//     WEIGHT_AVG,<avg_g>
//     FEED,<0|1>         // 1=keep feeding, 0=stop feeding (target reached)
//     WEIGHT_RDY         // weighing cycle finished (success or error)
//     WEIGHT_ERR         // optional (if sampling failed)
//     NOT_READY
//     TARGET_G,<g>        // current target after handshake
//     TARGET_OK,<g>       // target weight updated
//     TARGET_REJECT       // invalid target payload
//
// Weighing behavior on 'r' when piReady==true:
//   1) wait 1000ms (bean settle)
//   2) sample 10 times in 1 second (10Hz => 100ms interval)
//   3) compute average weight
//   4) output WEIGHT_AVG, FEED, WEIGHT_RDY

#include <Arduino.h>
#include "HX711.h"
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// ---------- Load Cell Weight Sensor ----------
#define DOUT_PIN 2
#define SCK_PIN 3

HX711 scale;

const float calibration_factor = 433.467163f;
const float DEFAULT_WEIGHT_STOP_G = 20.0f;
float currentWeightStopG = DEFAULT_WEIGHT_STOP_G;

static const uint32_t BEAN_SETTLE_MS = 1000;
static const uint8_t  WEIGH_SAMPLES = 10;
static const uint32_t WEIGH_INTERVAL_MS = 100;      // 10Hz
static const uint32_t HX711_READY_TIMEOUT_MS = 150; // per sample wait

// ---------- Robotic Arm ----------
Adafruit_PWMServoDriver pca9685 = Adafruit_PWMServoDriver(0x40);

#define SERVO_MIN_US  500
#define SERVO_MAX_US  2500
#define PWM_FREQ      50
#define NUM_SERVOS    5

const uint8_t servoChannels[NUM_SERVOS] = {0, 1, 2, 3, 4};

#define SWEEP_STEP    5
#define SWEEP_DELAY   20
#define STATE_DELAY   2000

enum RobotState {
  WAIT_WEIGHT,
  INIT,
  MOVE_TO_OBJECT,
  CLOSE_GRIPPER,
  MOVE_TO_USER,
  OPEN_GRIPPER,
  DONE
};

RobotState currentState = WAIT_WEIGHT;
bool sequenceDone = false;

// Pi 未发 'g' 之前不响应称重/机械臂动作
bool piReady = false;


static bool parseTargetWeightCmd(const String &line, float &target_g) {
  if (!line.startsWith("w,")) {
    return false;
  }

  String value = line.substring(2);
  value.trim();
  if (value.length() == 0) {
    return false;
  }

  float parsed = value.toFloat();
  if (parsed <= 0.0f) {
    return false;
  }

  target_g = parsed;
  return true;
}

static bool readAverageWeight10Hz(float &avg_g) {
  // Ensure scale is ready at least once
  if (!scale.wait_ready_timeout(1000)) {
    return false;
  }

  double sum_g = 0.0;
  uint8_t got = 0;

  for (uint8_t i = 0; i < WEIGH_SAMPLES; i++) {
    uint32_t t0 = millis();

    if (scale.wait_ready_timeout(HX711_READY_TIMEOUT_MS)) {
      float w = scale.get_units(1);
      sum_g += (double)w;
      got++;
    }

    // keep ~100ms sampling interval
    while ((millis() - t0) < WEIGH_INTERVAL_MS) {
      delay(1);
    }
  }

  if (got == 0) {
    avg_g = 0.0f;
    return false;
  }

  avg_g = (float)(sum_g / (double)got);
  return (got == WEIGH_SAMPLES);
}

uint16_t usToPulse(uint16_t us) {
  return (uint16_t)((us / 20000.0) * 4096);
}

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

void sweepTwoServos(uint8_t ch1, uint8_t ch2, uint8_t startAngle, uint8_t endAngle) {
  if (startAngle < endAngle) {
    for (int a = startAngle; a <= endAngle; a += SWEEP_STEP) {
      setServoAngle(ch1, constrain(a, 0, 180));
      setServoAngle(ch2, constrain(a, 0, 180));
      delay(SWEEP_DELAY);
    }
  } else {
    for (int a = startAngle; a >= endAngle; a -= SWEEP_STEP) {
      setServoAngle(ch1, constrain(a, 0, 180));
      setServoAngle(ch2, constrain(a, 0, 180));
      delay(SWEEP_DELAY);
    }
  }
  setServoAngle(ch1, endAngle);
  setServoAngle(ch2, endAngle);
}

void runINIT() {
  Serial.println("INIT");
  setServoAngle(servoChannels[0], 40);
  setServoAngle(servoChannels[1], 20);
  setServoAngle(servoChannels[2], 95);
  setServoAngle(servoChannels[3], 80);
  setServoAngle(servoChannels[4], 30);   // gripper open
  delay(STATE_DELAY);
}

void runMoveToObject() {
  Serial.println("MOVE TO OBJECT");
  setServoAngle(servoChannels[0], 0);
  delay(STATE_DELAY);
}

void runCloseGripper() {
  Serial.println("CLOSE GRIPPER");
  setServoAngle(servoChannels[4], 0);   // close
  delay(STATE_DELAY);
}

void runMoveToUser() {
  Serial.println("MOVE TO USER");
  setServoAngle(servoChannels[0], 90);
  delay(STATE_DELAY);
}

void runOpenGripper() {
  Serial.println("OPEN GRIPPER");
  setServoAngle(servoChannels[4], 30);  // open again
  delay(STATE_DELAY);
}

void setup() {
  Serial.begin(115200);
  delay(200);

  scale.begin(DOUT_PIN, SCK_PIN);
  scale.set_scale(calibration_factor);
  if (scale.wait_ready_timeout(1000)) {
    scale.tare(10);
    Serial.println("Tared (basket). Ready.");
  } else {
    Serial.println("HX711 not ready at boot.");
  }

  Serial.println("=================================");
  Serial.println(" Pi sends 'g' to start; then:");
  Serial.println("   t = tare");
  Serial.println("   r = weigh (settle 1s, 10 samples @10Hz) -> WEIGHT_AVG + FEED + WEIGHT_RDY");
  Serial.println("   a = allow robotic arm to move");
  Serial.println("   w,<g> = change target weight");
  Serial.print(" Default target weight (g): ");
  Serial.println(currentWeightStopG, 2);
  Serial.println("=================================");

  Wire.begin();
  pca9685.begin();
  pca9685.setPWMFreq(PWM_FREQ);
  delay(100);

  currentState = WAIT_WEIGHT;
  sequenceDone = false;
}

void loop() {
  if (Serial.available() > 0) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    if (line.length() == 0) {
      return;
    }

    if (line == "g") {
      piReady = true;
      Serial.println("READY");
      Serial.print("TARGET_G,");
      Serial.println(currentWeightStopG, 2);
      return;
    }

    if (line == "t") {
      if (scale.wait_ready_timeout(1000)) {
        scale.tare(10);
        Serial.println("TARE_OK");
      } else {
        Serial.println("TARE_FAIL");
      }
      return;
    }

    float requestedTargetG = 0.0f;
    if (parseTargetWeightCmd(line, requestedTargetG)) {
      currentWeightStopG = requestedTargetG;
      Serial.print("TARGET_OK,");
      Serial.println(currentWeightStopG, 2);
      return;
    } else if (line.startsWith("w,")) {
      Serial.println("TARGET_REJECT");
      return;
    }

    if (line == "r") {
      if (!piReady) {
        Serial.println("NOT_READY");
        Serial.println("WEIGHT_RDY");
        return;
      }

      delay(BEAN_SETTLE_MS);

      float avg_g = 0.0f;
      bool ok = readAverageWeight10Hz(avg_g);
      if (!ok) {
        Serial.println("WEIGHT_ERR");
        Serial.println("WEIGHT_RDY");
        return;
      }

      Serial.print("WEIGHT_AVG,");
      Serial.println(avg_g, 2);

      if (avg_g >= currentWeightStopG) {
        Serial.println("FEED,0");
      } else {
        Serial.println("FEED,1");
      }

      Serial.println("WEIGHT_RDY");
      return;
    }

    if (line == "a") {
      if (!piReady) {
        Serial.println("NOT_READY");
        return;
      }
      Serial.println("ARM_START");
      currentState = INIT;
      sequenceDone = false;
      return;
    }
  }

  if (sequenceDone) {
    delay(50);
    return;
  }

  if (currentState == WAIT_WEIGHT) {
    delay(50);
    return;
  }

  switch (currentState) {
    case INIT:           runINIT();          currentState = MOVE_TO_OBJECT; break;
    case MOVE_TO_OBJECT: runMoveToObject();  currentState = CLOSE_GRIPPER; break;
    case CLOSE_GRIPPER:  runCloseGripper();  currentState = MOVE_TO_USER;   break;
    case MOVE_TO_USER:   runMoveToUser();    currentState = OPEN_GRIPPER;   break;
    case OPEN_GRIPPER:   runOpenGripper();   currentState = DONE;           break;
    case DONE:
      Serial.println("ARM_DONE");
      sequenceDone = true;
      break;
    case WAIT_WEIGHT:
      break;
  }
}
