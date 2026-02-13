#!/usr/bin/env python3
import time
import RPi.GPIO as GPIO

M1_STEP, M1_DIR = 17, 27
M2_STEP, M2_DIR = 22, 23

PULSE_US = 20
GAP_US   = 2000   # slow on purpose
STEPS    = 400

def stepper(step_pin, dir_pin, forward=True, steps=STEPS):
    GPIO.output(dir_pin, GPIO.HIGH if forward else GPIO.LOW)
    time.sleep(0.05)  # DIR setup time
    pulse = PULSE_US / 1_000_000
    gap   = GAP_US / 1_000_000
    for _ in range(steps):
        GPIO.output(step_pin, 1)
        time.sleep(pulse)
        GPIO.output(step_pin, 0)
        time.sleep(gap)

def main():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    for p in [M1_STEP, M1_DIR, M2_STEP, M2_DIR]:
        GPIO.setup(p, GPIO.OUT)
        GPIO.output(p, 0)

    print("M1 forward...")
    stepper(M1_STEP, M1_DIR, True)
    time.sleep(1)

    print("M1 reverse...")
    stepper(M1_STEP, M1_DIR, False)
    time.sleep(1)

    print("M2 forward...")
    stepper(M2_STEP, M2_DIR, True)
    time.sleep(1)

    print("M2 reverse...")
    stepper(M2_STEP, M2_DIR, False)

    GPIO.cleanup()
    print("done")

if __name__ == "__main__":
    main()
