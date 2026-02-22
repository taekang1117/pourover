#!/usr/bin/env python3
import sys
import signal
from rpi_ws281x import PixelStrip, Color

# ====== CONFIG ======
LED_COUNT = 7
LED_PIN = 18         # BCM 18 = physical pin 12
LED_FREQ_HZ = 800000
LED_DMA = 10
LED_INVERT = False
LED_BRIGHTNESS = 255
LED_CHANNEL = 0
# ====================

strip = PixelStrip(LED_COUNT, LED_PIN, LED_FREQ_HZ, LED_DMA, LED_INVERT, LED_BRIGHTNESS, LED_CHANNEL)

def show_white(v: int):
    v = max(0, min(255, v))
    c = Color(v, v, v)
    for i in range(LED_COUNT):
        strip.setPixelColor(i, c)
    strip.show()

def map_1_to_10(level: int) -> int:
    return int((level - 1) * (255 - 25) / 9 + 25)

def cleanup(*_args):
    show_white(0)
    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    strip.begin()
    show_white(25)
    print("System Ready. Enter brightness 1-10:")

    while True:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.strip()

        try:
            level = int(line)
        except ValueError:
            print("Please enter a number between 1 and 10.")
            continue

        if 1 <= level <= 10:
            white_value = map_1_to_10(level)
            show_white(white_value)
            print(f"Level: {level} (RGB Value: {white_value})")
        else:
            print("Please enter a number between 1 and 10.")

    cleanup()

if __name__ == "__main__":
    main()
