#!/usr/bin/env python3
import time
import threading
import RPi.GPIO as GPIO

# Physical pins you gave:
# 11->GPIO17, 12->GPIO18, 13->GPIO27, 15->GPIO22
PINS = [17, 18, 27, 22]  # IN1, IN2, IN3, IN4

# 2-phase full-step sequence (stronger torque than half-step ripple)
FULL_STEP_2PH = [
    [1, 1, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 1, 1],
    [1, 0, 0, 1],
]

def setup_gpio():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    for pin in PINS:
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, 0)

def set_step(pattern):
    for pin, val in zip(PINS, pattern):
        GPIO.output(pin, val)

def release():
    set_step([0, 0, 0, 0])

class StepperRunner:
    def __init__(self, direction=1):
        self.direction = 1 if direction >= 0 else -1

        # Speed control (seconds per step)
        # Smaller = faster. Too small = stall/skip.
        self.delay = 0.0012          # target running speed (try 0.0012 -> 0.0009)
        self.ramp_start_delay = 0.004  # start slower for ramp
        self.ramp_steps = 250         # how many steps to ramp down

        self._run = False
        self._stop = False
        self._do_ramp = False
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._worker, daemon=True)

    def start(self):
        self._thread.start()

    def rotate_on(self):
        with self._lock:
            self._run = True
            self._do_ramp = True  # ramp each time you start

    def rotate_off(self):
        with self._lock:
            self._run = False
        release()

    def shutdown(self):
        with self._lock:
            self._stop = True
            self._run = False
        self._thread.join(timeout=2.0)
        release()

    def _worker(self):
        seq = FULL_STEP_2PH if self.direction > 0 else list(reversed(FULL_STEP_2PH))
        step_index = 0

        while True:
            with self._lock:
                if self._stop:
                    break
                running = self._run
                do_ramp = self._do_ramp
                target_delay = self.delay

            if not running:
                time.sleep(0.05)
                continue

            # Optional acceleration ramp (helps prevent stalling when starting fast)
            if do_ramp:
                d0 = self.ramp_start_delay
                d1 = target_delay
                n = max(1, self.ramp_steps)
                for i in range(n):
                    with self._lock:
                        if not self._run or self._stop:
                            break
                    # linear ramp
                    d = d0 + (d1 - d0) * (i / (n - 1))
                    set_step(seq[step_index])
                    step_index = (step_index + 1) % len(seq)
                    time.sleep(d)

                with self._lock:
                    self._do_ramp = False

            # Continuous run at target speed
            set_step(seq[step_index])
            step_index = (step_index + 1) % len(seq)
            time.sleep(target_delay)

def main():
    setup_gpio()
    runner = StepperRunner(direction=1)
    runner.start()

    print("Commands:")
    print("  r + Enter  -> rotate continuously (fast)")
    print("  q + Enter  -> stop and quit")

    try:
        while True:
            cmd = input("> ").strip().lower()
            if cmd == "r":
                runner.rotate_on()
                print("Rotating... (type q to stop/quit)")
            elif cmd == "q":
                print("Stopping and exiting...")
                runner.rotate_off()
                break
            else:
                print("Unknown command. Use 'r' or 'q'.")
    finally:
        runner.shutdown()
        GPIO.cleanup()

if __name__ == "__main__":
    main()
