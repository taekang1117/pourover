#!/usr/bin/env python3
# dual_tmc2209_stepdir.py
# Run: sudo python3 dual_tmc2209_stepdir.py
#
# Controls 2x TMC2209 in STEP/DIR mode (no UART).
# EN is active-low on TMC2209 boards: LOW=enable, HIGH=disable.

import time
import sys
import termios
import tty
import select
import RPi.GPIO as GPIO

# =========================
# Pin map (BCM numbering)
# =========================
M1_STEP = 17
M1_DIR  = 27
M1_EN   = 24

M2_STEP = 22
M2_DIR  = 23
M2_EN   = 25

# =========================
# Tuning parameters
# =========================
STEP_PULSE_US = 8          # step high time (>=2us is typically ok; 5-10us is safe)
STEP_GAP_US   = 800        # time between steps => sets speed. 800us => ~1250 steps/s
DOSE_STEPS    = 300        # "one dose" amount; tune this for 1–2 beans
SETTLE_SEC    = 0.25       # pause after dosing so the part settles

# If your motor direction is reversed, flip this sign or swap DIR logic:
M1_DIR_NORMAL = True
M2_DIR_NORMAL = True

# =========================
# Non-blocking key input
# =========================
def getch_nonblocking():
    """Return one character if available, else None."""
    dr, _, _ = select.select([sys.stdin], [], [], 0)
    if dr:
        return sys.stdin.read(1)
    return None

class RawTerminal:
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self
    def __exit__(self, exc_type, exc, tb):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

# =========================
# Stepper helper
# =========================
class Stepper:
    def __init__(self, step_pin, dir_pin, en_pin, dir_normal=True, name="M"):
        self.step_pin = step_pin
        self.dir_pin = dir_pin
        self.en_pin = en_pin
        self.dir_normal = dir_normal
        self.name = name

    def enable(self):
        # EN is active-low
        GPIO.output(self.en_pin, GPIO.LOW)

    def disable(self):
        GPIO.output(self.en_pin, GPIO.HIGH)

    def set_dir(self, forward: bool):
        # forward=True means "normal" direction
        level = GPIO.HIGH if (forward == self.dir_normal) else GPIO.LOW
        GPIO.output(self.dir_pin, level)

    def step_n(self, steps: int, step_gap_us=STEP_GAP_US, pulse_us=STEP_PULSE_US, stop_flag=None, paused_flag=None):
        """
        Step a fixed number of steps.
        stop_flag: callable returning True if should stop immediately.
        paused_flag: callable returning True if paused (wait until unpaused).
        """
        self.enable()
        # Small delay after enabling driver
        time.sleep(0.002)

        pulse_s = pulse_us / 1_000_000.0
        gap_s   = step_gap_us / 1_000_000.0

        for _ in range(steps):
            if stop_flag and stop_flag():
                break

            # Pause handling
            while paused_flag and paused_flag():
                time.sleep(0.01)
                if stop_flag and stop_flag():
                    self.disable()
                    return

            GPIO.output(self.step_pin, GPIO.HIGH)
            time.sleep(pulse_s)
            GPIO.output(self.step_pin, GPIO.LOW)
            time.sleep(gap_s)

        self.disable()

# =========================
# Main
# =========================
def main():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)

    pins = [M1_STEP, M1_DIR, M1_EN, M2_STEP, M2_DIR, M2_EN]
    for p in pins:
        GPIO.setup(p, GPIO.OUT)

    # Default states
    GPIO.output(M1_STEP, GPIO.LOW)
    GPIO.output(M2_STEP, GPIO.LOW)

    # Disable drivers initially (EN high)
    GPIO.output(M1_EN, GPIO.HIGH)
    GPIO.output(M2_EN, GPIO.HIGH)

    m1 = Stepper(M1_STEP, M1_DIR, M1_EN, dir_normal=M1_DIR_NORMAL, name="M1")
    m2 = Stepper(M2_STEP, M2_DIR, M2_EN, dir_normal=M2_DIR_NORMAL, name="M2")

    stop = {"flag": False}
    paused = {"flag": False}

    def stop_flag():
        return stop["flag"]

    def paused_flag():
        return paused["flag"]

    print("\nDual TMC2209 STEP/DIR test")
    print("Keys:")
    print("  1: dose Motor 1 (DOSE_STEPS)")
    print("  2: dose Motor 2 (DOSE_STEPS)")
    print("  a: jog Motor 1 forward (hold)")
    print("  z: jog Motor 1 reverse (hold)")
    print("  k: jog Motor 2 forward (hold)")
    print("  m: jog Motor 2 reverse (hold)")
    print("  p: pause/resume (pauses stepping loops)")
    print("  q: qu
