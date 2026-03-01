#include <Arduino.h>
#include "HX711.h"

// HX711 pins (PD2 = D2, PD3 = D3 on AVR/Uno)
#define DOUT_PIN 2
#define SCK_PIN 3

HX711 scale;

// 校准因子（已用已知重量标定）
const float calibration_factor = 433.467163f;

void setup() {
  Serial.begin(115200);
  delay(200);

  scale.begin(DOUT_PIN, SCK_PIN);
  scale.set_scale(calibration_factor);

  // 程序开始时 basket 已在秤上，自动 tare 把空篮重量清零
  if (scale.wait_ready_timeout(1000)) {
    scale.tare(10);
    Serial.println("Tared (basket). Press 'r' to read weight.");
  } else {
    Serial.println("HX711 not ready. Press 'r' to read when ready.");
  }
}

void loop() {
  if (Serial.available() <= 0) {
    delay(50);
    return;
  }

  char cmd = (char)Serial.read();
  while (Serial.available() && Serial.peek() != '\n' && Serial.peek() != '\r')
    (void)Serial.read();

  if (cmd == 't' || cmd == 'b') {
    if (scale.wait_ready_timeout(1000)) {
      scale.tare(10);
      Serial.println("Tared.");
    } else {
      Serial.println("HX711 not ready. Tare failed.");
    }
    return;
  }

  if (cmd == 'r') {
    if (!scale.wait_ready_timeout(1000)) {
      Serial.println("HX711 not ready.");
      return;
    }
    float weight_g = scale.get_units(10);
    Serial.print("Weight: ");
    Serial.print(weight_g, 2);
    Serial.println(" g");
    return;
  }

  delay(50);
}
