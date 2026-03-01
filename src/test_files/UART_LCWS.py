#!/usr/bin/env python3
# weight_only_ttyacm0.py

import sys
import time
import re

import serial  # pip install pyserial

PORT = "/dev/ttyACM0"
BAUD = 115200

# 兼容多种可能格式：
# WEIGHT_AVG,123.45
# WEIGHT_AVG: 123.45
# WEIGHT=123.45
WEIGHT_RE = re.compile(r"(WEIGHT(?:_AVG)?)\s*[,=:]\s*([-+]?\d*\.?\d+)")

def main():
    print(f"[INFO] Opening {PORT} @ {BAUD} ...")
    ser = serial.Serial(PORT, baudrate=BAUD, timeout=0.2)

    # Arduino 常见：串口打开会触发重启，给它一点启动时间
    time.sleep(2.0)

    print(f"[OK] Opened serial: {PORT} @ {BAUD}")
    print("[INFO] Waiting for WEIGHT... (Ctrl+C to exit)")

    last_any_data_ts = time.time()

    try:
        while True:
            line = ser.readline()  # 读到 '\n' 或 timeout
            if line:
                last_any_data_ts = time.time()

                s = line.decode("utf-8", errors="ignore").strip()
                m = WEIGHT_RE.search(s)
                if m:
                    w = float(m.group(2))
                    # 实时刷新同一行
                    sys.stdout.write(f"\rWeight: {w:.3f} g    ")
                    sys.stdout.flush()

            # 心跳：如果一直没收到任何串口数据，提示你“根本没数据进来”
            if time.time() - last_any_data_ts > 1.0:
                sys.stdout.write("\r[WAIT] No serial data yet...            ")
                sys.stdout.flush()
                last_any_data_ts = time.time()

    except KeyboardInterrupt:
        print("\n[EXIT] Bye.")
    finally:
        try:
            ser.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
