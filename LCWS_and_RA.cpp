// ========== Optical Sorting: LCWS + Robotic Arm ==========
// Pi 发 's'（sort 出一粒豆后）→ 短 delay → 采样称重一段时间 → 平均重量发回 Pi，并与 WEIGHT_STOP_G 比较

#include <Arduino.h>
#include "HX711.h"
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// ---------- Load Cell Weight Sensor ----------
#define DOUT_PIN 2
#define SCK_PIN 3

HX711 scale;

const float calibration_factor = 433.467163f;
const float WEIGHT_STOP_G = 20.0f;

// Pi 发 's' 后：等待多久再开始采样（豆子落稳）
const uint32_t DELAY_BEFORE_WEIGH_MS = 500;
// 采样总时长（毫秒）
const uint32_t WEIGH_DURATION_MS = 2000;
// 采样间隔（毫秒），每次取 scale.get_units(5)
const uint32_t WEIGH_SAMPLE_INTERVAL_MS = 100;

// 称重达到 WEIGHT_STOP_G 后，延迟此时间（毫秒）再启动机械臂
const uint32_t DELAY_AFTER_WEIGHT_MS = 2000;

// ---------- Robotic Arm ----------
Adafruit_PWMServoDriver pca9685 = Adafruit_PWMServoDriver(0x40);

#define SERVO_MIN_US  500
#define SERVO_MAX_US  2500
#define PWM_FREQ      50
#define NUM_SERVOS    5

const uint8_t servoChannels[NUM_SERVOS] = {0, 1, 2, 3, 4};

#define SWEEP_STEP    5
#define SWEEP_DELAY   20

enum RobotState {
  WAIT_WEIGHT,  // 等待称重到 20g，之后 delay 再启动机械臂
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

// Pi 未发 'g' 之前不响应 's'（称重/FEED），实现“all.py 启动后再开始”
bool piReady = false;

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
  Serial.println("[INIT] All servos → 0 degrees");

  for (uint8_t i = 1; i < NUM_SERVOS; i++) {
    setServoAngle(servoChannels[i], 0);
  }
  setServoAngle(servoChannels[0], 45);
  delay(1500);
}

void runState1() {
  Serial.println("[STATE 1] Servo 1 & 2 → 0 to 90 degrees");
  sweepTwoServos(servoChannels[1], servoChannels[2], 0, 90);
  delay(1500);
}

void runState2() {
  Serial.println("[STATE 2] Servo 3 → 0 to 90 degrees");
  sweepServo(servoChannels[3], 0, 90);
  delay(1500);
}

void runState3() {
  Serial.println("[STATE 3] Servo 4 → 0 to 90 degrees");
  sweepServo(servoChannels[4], 0, 180);
  delay(1500);
}

void runState4() {
  Serial.println("[STATE 4] Servo 1 & 2 → 90 back to 0 degrees");

  sweepServo(servoChannels[0], 45, 0);
  delay(1500);
}

void runState5() {
  Serial.println("[STATE 5] Servo 4 → 90 back to 0 degrees");
  sweepServo(servoChannels[4], 180, 0);
  delay(1500);
}

void runState6() {
  Serial.println("[STATE 6] Servo 2 → 0 back to 180 degrees");
  sweepTwoServos(servoChannels[1], servoChannels[2], 90, 0);//drag
  delay(1500);
}

void runState7() {
  Serial.println("[STATE 7] Servo 0 → 0 back to 180 degrees");
  sweepServo(servoChannels[0], 0, 180);
  delay(1500);
}

void runState8() {
  Serial.println("[STATE 8] Servo 4 → 0 back to 90 degrees");
  sweepServo(servoChannels[4], 0, 180);
  delay(1500);
}

// ---------- 合并 setup ----------
void setup() {
  Serial.begin(115200);  // 与 Pi 通信需 115200（原 RA 为 9600，仅打印，现统一 115200）
  delay(200);

  scale.begin(DOUT_PIN, SCK_PIN);
  scale.set_scale(calibration_factor);
  if (scale.wait_ready_timeout(1000)) {
    scale.tare(10);
    Serial.println("Tared (basket). Press 'r' to read weight.");
  } else {
    Serial.println("HX711 not ready. Press 'r' to read when ready.");
  }

  Serial.println("=================================");
  Serial.println(" Pi sends 'g' to start; then t/tare, r/weigh, s=sort done (weight+FEED to Pi)");
  Serial.println(" Arm starts when avg weight >= 20g + delay");
  Serial.println(" SDA = A4 | SCL = A5");
  Serial.println("=================================");

  Wire.begin();
  pca9685.begin();
  pca9685.setPWMFreq(PWM_FREQ);
  delay(100);

  currentState = WAIT_WEIGHT;  // 先等称重到 20g，再启动机械臂
  sequenceDone = false;
}

// ---------- 合并 loop：有 't'/'r' 则执行称重，否则执行机械臂状态机一步 ----------
void loop() {
  if (Serial.available() > 0) {
    char cmd = (char)Serial.read();
    while (Serial.available() && Serial.peek() != '\n' && Serial.peek() != '\r')
      (void)Serial.read();

    if (cmd == 'g') {
      // all.py 启动后发 'g'，Arduino 收到后才开始响应 's'
      piReady = true;
      Serial.println("READY");
      return;
    }

    if (cmd == 't') {
      if (scale.wait_ready_timeout(1000)) {
        scale.tare(10);
        Serial.println("Tared.");
      } else {
        Serial.println("HX711 not ready. Tare failed.");
      }
      return;
    }

    if (cmd == 's') {
      if (!piReady) {
        Serial.println("NOT_READY");
        return;
      }
      // Optical sort 后 Pi 发 's'：短 delay → 采样一段时间 → 平均重量发回 Pi，并与 WEIGHT_STOP_G 比较
      if (!scale.wait_ready_timeout(1000)) {
        Serial.println("HX711 not ready.");
        return;
      }
      delay(DELAY_BEFORE_WEIGH_MS);

      uint32_t nSamples = 0;
      double sum_g = 0;
      uint32_t tEnd = millis() + WEIGH_DURATION_MS;

      while (millis() < tEnd) {
        if (scale.wait_ready_timeout(200)) {
          float w = scale.get_units(5);
          sum_g += (double)w;
          nSamples++;
        }
        delay(WEIGH_SAMPLE_INTERVAL_MS);
      }

      float avg_g = (nSamples > 0) ? (float)(sum_g / nSamples) : 0.0f;
      Serial.print("WEIGHT_AVG,");
      Serial.println(avg_g, 2);

      // 1. 重量已发回 Pi；2. 根据是否达到目标重量决定 slow feeder 是否工作，并发送给 Pi
      if (avg_g >= WEIGHT_STOP_G) {
        Serial.println("FEED,0");   // 达到目标，feeder 停止
        Serial.print("Avg >= ");
        Serial.print(WEIGHT_STOP_G, 0);
        Serial.println(" g. Delay then start arm.");
        delay(DELAY_AFTER_WEIGHT_MS);
        currentState = INIT;
      } else {
        Serial.println("FEED,1");   // 未达到目标，feeder 继续工作
      }
      return;
    }

    if (cmd == 'r') {
      if (!scale.wait_ready_timeout(1000)) {
        Serial.println("HX711 not ready.");
        return;
      }
      Serial.println("Weighing... (stops at 20 g)");
      float weight_g;
      do {
        if (!scale.wait_ready_timeout(1000)) {
          Serial.println("HX711 not ready.");
          return;
        }
        weight_g = scale.get_units(10);
        Serial.print("Weight: ");
        Serial.print(weight_g, 2);
        Serial.println(" g");
        if (weight_g >= WEIGHT_STOP_G) {
          Serial.print("Reached ");
          Serial.print(WEIGHT_STOP_G, 0);
          Serial.println(" g. Stopped.");
          Serial.print("WEIGHT_READY,");
          Serial.println(weight_g, 2);
          Serial.print("Waiting ");
          Serial.print(DELAY_AFTER_WEIGHT_MS);
          Serial.println(" ms, then starting arm.");
          delay(DELAY_AFTER_WEIGHT_MS);
          currentState = INIT;
          break;
        }
        delay(300);
      } while (true);
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
