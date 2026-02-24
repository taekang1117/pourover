#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load Cell Weight Sensor - Raspberry Pi version.
Converted from Arduino (C); uses HX711 load cell amplifier.
"""

import time
import sys

try:
    from hx711 import HX711
    import RPi.GPIO as GPIO
except ImportError as e:
    print("Please install dependencies: pip3 install hx711 RPi.GPIO")
    print("  (On Raspberry Pi: sudo pip3 install hx711 RPi.GPIO)")
    sys.exit(1)

# ------------ Wiring and calibration ------------
# Arduino D2 -> GPIO connected to HX711 DOUT (e.g. BCM 5)
# Arduino D3 -> GPIO connected to HX711 SCK (e.g. BCM 6)
DOUT_BCM = 5   # Data pin (DOUT)
SCK_BCM = 6    # Clock pin (SCK)

# Calibration factor: same value used on Arduino (unit depends on your calibration, e.g. grams)
CALIBRATION_FACTOR = -7050.0

# Number of samples (matches Arduino get_units(10))
NUM_SAMPLES = 10

# Ready-check timeout (ms), matches Arduino wait_ready_timeout(1000)
READY_TIMEOUT_MS = 1000

# Loop interval (seconds)
LOOP_DELAY_SEC = 0.5


def read_raw_with_timeout(hx, num_measures, timeout_sec):
    """Read raw values with timeout (mimics Arduino wait_ready_timeout)."""
    import threading
    result = [None]
    err = [None]

    def read():
        try:
            result[0] = hx.get_raw_data(num_measures=num_measures)
        except Exception as e:
            err[0] = e

    t = threading.Thread(target=read, daemon=True)
    t.start()
    t.join(timeout=timeout_sec)
    if t.is_alive():
        return None  # Timeout, sensor not ready
    if err[0] is not None:
        raise err[0]
    return result[0]


def main():
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)

    print("HX711 Weight Sensor starting...")
    time.sleep(0.2)

    hx = HX711(
        dout_pin=DOUT_BCM,
        pd_sck_pin=SCK_BCM,
        channel="A",
        gain=64,
    )
    hx.reset()

    # Tare: set current reading as zero offset
    timeout_sec = READY_TIMEOUT_MS / 1000.0
    raw_list = read_raw_with_timeout(hx, 5, timeout_sec)
    if raw_list is None:
        print("HX711 not found or not ready. Check wiring and GPIO pins.")
        GPIO.cleanup()
        return
    offset = sum(raw_list) / len(raw_list)
    print("Scale initialized. Tare complete.")

    try:
        while True:
            raw_list = read_raw_with_timeout(hx, NUM_SAMPLES, timeout_sec)
            if raw_list is None:
                print("HX711 not found or not ready.")
            else:
                avg_raw = sum(raw_list) / len(raw_list)
                # Same as Arduino: units = (raw - offset) / calibration_factor
                weight = (avg_raw - offset) / CALIBRATION_FACTOR
                print("Weight: {:.2f} g".format(weight))
            time.sleep(LOOP_DELAY_SEC)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        GPIO.cleanup()


if __name__ == "__main__":
    main()
