#!/usr/bin/env python3
# UART_LCWS_test.py
import sys
import time
import re
import argparse

import serial  # pip install pyserial

# 匹配 Arduino 输出：Weight: 8.61 g
WEIGHT_RE = re.compile(r"Weight:\s*([-+]?\d*\.?\d+)\s*g", re.IGNORECASE)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/ttyACM0", help="Serial port (default: /dev/ttyACM0)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate (default: 115200)")
    parser.add_argument("--interval", type=float, default=1.0, help="Send 'r' every N seconds (default: 1.0)")
    parser.add_argument("--newline", action="store_true", help="Print each weight on a new line")
    parser.add_argument("--raw", action="store_true", help="Also print non-Weight lines (debug)")
    args = parser.parse_args()

    print(f"[INFO] Opening {args.port} @ {args.baud} ...")
    ser = serial.Serial(args.port, baudrate=args.baud, timeout=0.2)

    # 打开串口通常会让 Arduino reset，给它一点启动时间
    time.sleep(2.0)
    try:
        ser.reset_input_buffer()
    except Exception:
        pass

    print(f"[OK] Opened serial: {args.port} @ {args.baud}")
    print(f"[INFO] Sending 'r' every {args.interval:.2f}s. (Ctrl+C to exit)")

    next_send = time.time()  # 立即发送一次
    last_seen_any = time.time()

    try:
        while True:
            now = time.time()

            # 1) 定时发送 'r'
            if now >= next_send:
                try:
                    ser.write(b"r\n")
                    ser.flush()
                except Exception as e:
                    print(f"\n[ERR] write failed: {e}")
                    break
                next_send = now + args.interval

            # 2) 读取 Arduino 输出
            line = ser.readline()
            if line:
                last_seen_any = now
                s = line.decode("utf-8", errors="ignore").strip()

                m = WEIGHT_RE.search(s)
                if m:
                    w = float(m.group(1))
                    if args.newline:
                        ts = time.strftime("%H:%M:%S")
                        print(f"{ts}  Weight: {w:.2f} g")
                    else:
                        sys.stdout.write(f"\rWeight: {w:.2f} g    ")
                        sys.stdout.flush()
                else:
                    if args.raw:
                        print(s)
            else:
                # 连接测试：长时间没任何输出就提示一下
                if now - last_seen_any > 2.0:
                    sys.stdout.write("\r[WAIT] No serial output yet...           ")
                    sys.stdout.flush()
                    last_seen_any = now

    except KeyboardInterrupt:
        print("\n[EXIT] Bye.")
    finally:
        try:
            ser.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
