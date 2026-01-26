import time
import sys

import RPi.GPIO as GPIO
from adafruit_servokit import ServoKit

# ---- Settings ----
OE_PIN = 4                 # PCA9685 OE (active-low)
I2C_ADDR = 0x40            # your i2cdetect shows 0x40
NUM_SERVOS = 5             # channels 0-4
FREQ_HZ = 50               # standard servos
STEP_DELAY = 0.6           # seconds between moves


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def outputs_enable(enable: bool) -> None:
    # OE is active-low
    GPIO.output(OE_PIN, GPIO.LOW if enable else GPIO.HIGH)


def main() -> None:
    log("Program start")
    log(f"Python executable: {sys.executable}")
    log(f"Python version: {sys.version.split()[0]}")

    # Init GPIO for OE
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(OE_PIN, GPIO.OUT)
    log(f"GPIO OE_PIN={OE_PIN} configured as OUTPUT")

    # Create ServoKit (talks to PCA9685 over I2C)
    log(f"Initializing PCA9685 at I2C address 0x{I2C_ADDR:02X} ...")
    kit = ServoKit(channels=16, address=I2C_ADDR)
    kit.frequency = FREQ_HZ
    log(f"PCA9685 initialized; frequency set to {FREQ_HZ} Hz")

    # Enable outputs
    outputs_enable(True)
    log("OE set LOW => outputs ENABLED")
    time.sleep(0.2)

    # Basic motion sequence to prove commands are executing
    sequence = [0, 90, 180, 90]

    for angle in sequence:
        log(f"Commanding channels 0-{NUM_SERVOS-1} to angle={angle}")
        for ch in range(NUM_SERVOS):
            try:
                kit.servo[ch].angle = angle
                log(f"  ch{ch}: angle set to {angle}")
            except Exception as e:
                log(f"  ch{ch}: ERROR setting angle -> {e}")
        time.sleep(STEP_DELAY)

    log("Motion sequence complete; holding position for 2 seconds")
    time.sleep(2.0)

    # Optional: disable outputs at end
    # outputs_enable(False)
    # log("OE set HIGH => outputs DISABLED")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"FATAL ERROR: {e}")
        raise
    finally:
        try:
            GPIO.cleanup()
            log("GPIO cleanup done")
        except Exception:
            pass
        log("Program end")
