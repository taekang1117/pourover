# Flipper.py - ULN2003 Stepper (28BYJ-48 5V) for flip mechanism
# Run on Raspberry Pi: sudo python Flipper.py
#
# Keys: r = rotate right 135°, l = rotate left 135°, q = quit
# After each r/l move: wait 0.5s then return to original position. Input ignored until return.

import sys
import time

# =========================
# ULN2003 Stepper Setup (28BYJ-48 5V)
# =========================
FLIP_IN1 = 5
FLIP_IN2 = 6
FLIP_IN3 = 13
FLIP_IN4 = 19

FLIP_STEP_DELAY = 0.0018

# Flipper positions (steps relative to CENTER) — tune these
POS_RIGHT = +650     # BEAN side (RIGHT)
POS_LEFT  = -650     # ROCK side (LEFT)
DROP_WAIT_SEC = 0.35

# 135° rotation (28BYJ-48: 2048 half-steps/rev → 135° ≈ 768 steps; tune if needed)
STEPS_135_DEG = 768
RETURN_WAIT_SEC = 0.5


class ULN2003Stepper:
    HALF_SEQ = [
        (1, 0, 0, 0),
        (1, 1, 0, 0),
        (0, 1, 0, 0),
        (0, 1, 1, 0),
        (0, 0, 1, 0),
        (0, 0, 1, 1),
        (0, 0, 0, 1),
        (1, 0, 0, 1),
    ]

    def __init__(self, in1, in2, in3, in4, step_delay=0.002):
        import RPi.GPIO as GPIO
        self.GPIO = GPIO
        self.pins = [in1, in2, in3, in4]
        self.step_delay = step_delay
        self._idx = 0

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        for p in self.pins:
            GPIO.setup(p, GPIO.OUT)
            GPIO.output(p, 0)

    def _write(self, a, b, c, d):
        self.GPIO.output(self.pins[0], a)
        self.GPIO.output(self.pins[1], b)
        self.GPIO.output(self.pins[2], c)
        self.GPIO.output(self.pins[3], d)

    def step(self, steps, direction=1):
        direction = 1 if direction >= 0 else -1
        for _ in range(abs(int(steps))):
            self._idx = (self._idx + direction) % len(self.HALF_SEQ)
            self._write(*self.HALF_SEQ[self._idx])
            time.sleep(self.step_delay)

    def release(self):
        self._write(0, 0, 0, 0)

    def cleanup(self):
        self.release()
        self.GPIO.cleanup()


def move_to(stepper, target_pos, current_pos):
    """Move flipper from current_pos to target_pos (in steps). Returns new position."""
    delta = int(target_pos - current_pos)
    if delta == 0:
        return current_pos
    stepper.step(abs(delta), direction=+1 if delta > 0 else -1)
    return target_pos


def _getch_unix():
    """Read one character from stdin (Linux/RPi, raw mode)."""
    import tty
    import termios
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


def _getch_win():
    """Read one character (Windows)."""
    import msvcrt
    return msvcrt.getch().decode("utf-8", errors="replace")


def getch():
    """Read one keypress. Unix (RPi) or Windows."""
    if sys.platform == "win32":
        return _getch_win()
    return _getch_unix()


def drain_stdin():
    """Discard any pending keypresses so input during move is ignored."""
    if sys.platform == "win32":
        import msvcrt
        while msvcrt.kbhit():
            msvcrt.getch()
        return
    import select
    while select.select([sys.stdin], [], [], 0)[0]:
        sys.stdin.read(1)


if __name__ == "__main__":
    stepper = ULN2003Stepper(FLIP_IN1, FLIP_IN2, FLIP_IN3, FLIP_IN4,
                             step_delay=FLIP_STEP_DELAY)
    try:
        pos = 0
        print("r: rotate right 135° | l: rotate left 135° | q: quit")
        while True:
            key = getch().lower()
            if key == "q":
                break
            if key != "r" and key != "l":
                continue

            start_pos = pos
            if key == "r":
                pos = move_to(stepper, start_pos + STEPS_135_DEG, pos)
            else:
                pos = move_to(stepper, start_pos - STEPS_135_DEG, pos)

            time.sleep(RETURN_WAIT_SEC)
            pos = move_to(stepper, start_pos, pos)
            stepper.release()

            drain_stdin()
    finally:
        stepper.cleanup()
