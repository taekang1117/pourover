#!/usr/bin/env python3
import time
import threading
import RPi.GPIO as GPIO

# Your wiring (physical -> BCM):
# Pin 11 -> GPIO17, Pin 12 -> GPIO18, Pin 13 -> GPIO27, Pin 15 -> GPIO22
PINS = [17, 18, 27, 22]  # IN1, IN2, IN3, IN4

# Half-step sequence (smooth) for 28BYJ-48 + ULN2003
HALF_STEP_SEQ = [
    [1, 0, 0, 0],
    [1, 1, 0, 0],
    [0, 1, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 1, 0],
    [0, 0, 1, 1],
    [0, 0, 0, 1],
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
    def __init__(self, delay_s=0.002, direction=1):
        self.delay_s = delay_s
        self.direction = 1 if direction >= 0 else -1
        self._run = False
        self._stop = False
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._worker, daemon=True)

    def start(self):
        self._thread.start()

    def rotate_on(self):
        with self._lock:
            self._run = True

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
        seq = HALF_STEP_SEQ if self.direction > 0 else list(reversed(HALF_STEP_SEQ))
        step_index = 0

        while True:
            with self._lock:
                if self._stop:
                    break
                running = self._run

            if not running:
                time.sleep(0.05)
                continue

            # Output one half-step at a time
            set_step(seq[step_index])
            step_index = (step_index + 1) % len(seq)
            time.sleep(self.delay_s)

def main():
    setup_gpio()

    # Tune delay_s if needed:
    # bigger = slower/stronger, smaller = faster/more likely to skip
    runner = StepperRunner(delay_s=0.002, direction=1)
    runner.start()

    print("Commands:")
    print("  r + Enter  -> rotate continuously")
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
