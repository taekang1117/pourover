#!/usr/bin/env python3
# weight_monitor.py
import sys
import time
import argparse

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    serial = None
    list_ports = None


def auto_detect_port() -> str | None:
    """尽量自动找一个像 Arduino 的串口。"""
    if list_ports is None:
        return None
    ports = list(list_ports.comports())
    if not ports:
        return None

    # 优先挑看起来像 Arduino/USB 串口的
    preferred_keywords = ["Arduino", "CH340", "CP210", "USB Serial", "ttyACM", "ttyUSB"]
    for p in ports:
        desc = (p.description or "") + " " + (p.manufacturer or "")
        if any(k.lower() in desc.lower() for k in preferred_keywords):
            return p.device

    # 退而求其次：只有一个口就用它
    if len(ports) == 1:
        return ports[0].device

    return None


def open_serial(port: str, baud: int):
    while True:
        try:
            ser = serial.Serial(port, baudrate=baud, timeout=0.2)
            # 让 Arduino 上电/重置后有时间开始输出
            time.sleep(1.0)
            ser.reset_input_buffer()
            print(f"[OK] Opened serial: {port} @ {baud}")
            return ser
        except Exception as e:
            print(f"[WARN] Cannot open {port} @ {baud}: {e}")
            print("       Retrying in 1s ...")
            time.sleep(1.0)


def main():
    parser = argparse.ArgumentParser(description="Print Arduino WEIGHT_AVG to terminal in real time.")
    parser.add_argument("--port", default="", help='Serial port, e.g. "/dev/ttyACM0" or "COM3"')
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate (default: 115200)")
    parser.add_argument("--newline", action="store_true",
                        help="Print each weight on a new line (default: overwrite same line)")
    parser.add_argument("--raw", action="store_true",
                        help="Also print non-weight lines (debug)")
    args = parser.parse_args()

    if serial is None:
        print("pyserial is not installed. Install it with: pip install pyserial")
        sys.exit(1)

    port = args.port.strip()
    if not port:
        port = auto_detect_port()
        if not port:
            print("No serial port detected. Please specify --port.")
            print('Examples:  python3 weight_monitor.py --port /dev/ttyACM0')
            print('           python3 weight_monitor.py --port COM3')
            sys.exit(1)

    ser = open_serial(port, args.baud)

    last_print_time = 0.0
    try:
        while True:
            try:
                line = ser.readline()
                if not line:
                    continue

                s = line.decode("utf-8", errors="ignore").strip()
                if s.startswith("WEIGHT_AVG,"):
                    try:
                        w = float(s.split(",", 1)[1])
                    except Exception:
                        continue

                    # 你可以按需调小/调大格式精度
                    if args.newline:
                        ts = time.strftime("%H:%M:%S")
                        print(f"{ts}  {w:.3f} g")
                    else:
                        # 同一行刷新（更“实时”）
                        sys.stdout.write(f"\rWeight: {w:.3f} g    ")
                        sys.stdout.flush()

                    last_print_time = time.time()
                else:
                    if args.raw:
                        print(s)

            except serial.SerialException as e:
                print(f"\n[ERR] Serial error: {e}")
                try:
                    ser.close()
                except Exception:
                    pass
                ser = open_serial(port, args.baud)

    except KeyboardInterrupt:
        print("\n[EXIT] Bye.")
    finally:
        try:
            ser.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()