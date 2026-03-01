#!/usr/bin/env python3
# weight_only.py
import sys
import time
import re
import serial  # pip install pyserial

PORT = "/dev/ttyACM0"
BAUD = 115200

# 匹配：Weight: 8.61 g
WEIGHT_RE = re.compile(r"Weight:\s*([-+]?\d*\.?\d+)\s*g", re.IGNORECASE)

def main():
    print(f"[INFO] Opening {PORT} @ {BAUD} ...")
    ser = serial.Serial(PORT, baudrate=BAUD, timeout=0.2)

    # 打开串口通常会让 Arduino reset，给它启动时间
    time.sleep(2.0)

    print(f"[OK] Opened serial: {PORT} @ {BAUD}")
    print("[INFO] Sending 'r' to start weight output ...")
    try:
        ser.write(b"r\n")  # 触发开始输出重量
        ser.flush()
    except Exception as e:
        print(f"[WARN] Failed to send 'r': {e}")

    newline_mode = ("--newline" in sys.argv)

    last_any_line = time.time()
    try:
        while True:
            raw = ser.readline()
            if not raw:
                # 1 秒没收到任何行，提示一下（方便你判断是否还在输出）
                if time.time() - last_any_line > 1.0:
                    sys.stdout.write("\r[WAIT] No serial lines yet...            ")
                    sys.stdout.flush()
                    last_any_line = time.time()
                continue

            last_any_line = time.time()
            s = raw.decode("utf-8", errors="ignore").strip()

            m = WEIGHT_RE.search(s)
            if not m:
                # 只考虑 weight 输出：非 Weight 行全部忽略
                continue

            w = float(m.group(1))

            if newline_mode:
                ts = time.strftime("%H:%M:%S")
                print(f"{ts}  Weight: {w:.2f} g")
            else:
                # 同一行刷新，更像实时仪表
                sys.stdout.write(f"\rWeight: {w:.2f} g    ")
                sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n[EXIT] Bye.")
    finally:
        try:
            ser.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
