/*
 * Arduino Uno + HX711 + Load Cell 称重示例
 * 接线：5V→VCC, GND→GND, 数字引脚2→DT(DOUT), 数字引脚3→SCK
 */

#include "HX711.h"

// 与 HX711 的接线（Arduino 引脚 = PTD 编号）
// PTD2 = 数字引脚 2 → HX711 的 DT (DOUT，数据线)
// PTD3 = 数字引脚 3 → HX711 的 SCK（时钟线）
const int LOADCELL_DOUT_PIN = 2;  // DT
const int LOADCELL_SCK_PIN = 3;   // SCK

HX711 scale;

// HX711 是 24 位 ADC，理论范围 -8388608 ~ 8388607
// 以下为「明显异常」的读数，通常表示未接负载/通信错/时序问题，予以过滤
const long RAW_INVALID_HIGH = 4194303;   // 0x3FFFFF，半满量程全 1
const long RAW_INVALID_MAX  = 8388607;   // 0x7FFFFF，满量程全 1（DOUT 常高或读错）

void setup() {
  Serial.begin(57600);
  scale.begin(LOADCELL_DOUT_PIN, LOADCELL_SCK_PIN);

  Serial.println("HX711 称重传感器就绪");
  Serial.println("串口发送 't' 执行去皮(tare)");
}

void loop() {
  // 串口收到 't' 时执行去皮（先清空托盘，再发 t）
  if (Serial.available()) {
    if (Serial.read() == 't') {
      scale.tare(10);
      Serial.println("已执行 tare，当前为零点");
    }
  }

  if (scale.is_ready()) {
    long raw = scale.read();

    // 过滤明显无效的读数（通信异常或 DOUT 未正常拉低时的典型值）
    if (raw == RAW_INVALID_MAX || raw == RAW_INVALID_HIGH) {
      Serial.println("原始读数 raw: [可能无效，已忽略]");
      return;
    }

    Serial.print("原始读数 raw: ");
    Serial.println(raw);
  } else {
    Serial.println("HX711 未就绪，请检查接线");
  }

  delay(500);
}

/*
 * 校准步骤（在 setup 或通过串口命令调用）：
 *
 * 1) 去皮（无负载时归零）：
 *    scale.tare(10);  // 取 10 次平均作为零点
 *
 * 2) 标定比例（放已知重量砝码，如 100g）：
 *    scale.tare(10);
 *    // 放上 100g 砝码，等稳定后：
 *    float knownWeight = 100.0;  // 克
 *    long raw = scale.read_average(20);
 *    scale.set_scale(raw / knownWeight);
 *
 * 之后在 loop 里用 scale.get_units(5) 即可得到克数。
 *
 * --- 关于「未接 load cell」时的读数 ---
 * 1) -22000 ～ -23000 左右：正常。未接负载时电桥开路/悬空，HX711 会输出
 *    一个无意义的固定偏置，不是真实重量，接上 load cell 并去皮后才会准。
 * 2) 8388607、4194303：不正常，但很常见。表示「数据无效」：
 *    - 8388607 = 24 位最大正数(0x7FFFFF)，多为 DOUT 一直被读成 1 或读时序错；
 *    - 4194303 = 0x3FFFFF，多为半程全 1，常是时钟/数据不同步。
 *    未接 load cell 时 DOUT 行为不稳定，容易偶尔出现这两种值，已在上方过滤。
 */
