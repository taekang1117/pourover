import time
import RPi.GPIO as GPIO
from adafruit_servokit import ServoKit

# ---- Pins (BCM numbering) ----
OE_PIN = 4  # connected to PCA9685 OE (active-low)

# ---- PCA9685 setup ----
# Most PCA9685 boards default to I2C address 0x40.
# Change address=0x41, 0x42, etc. if you changed the address jumpers.
kit = ServoKit(channels=16, address=0x40)
kit.frequency = 50  # standard servo PWM frequency

# ---- OE control ----
GPIO.setmode(GPIO.BCM)
GPIO.setup(OE_PIN, GPIO.OUT)

def outputs_enable(enable: bool):
    # OE is active-low
    GPIO.output(OE_PIN, GPIO.LOW if enable else GPIO.HIGH)

try:
    # Enable PWM outputs
    outputs_enable(True)
    time.sleep(0.1)

    # Move 5 servos (channels 0-4) to 90 degrees
    for ch in range(5):
        kit.servo[ch].angle = 90

    # Hold position for a bit (optional)
    time.sleep(2.0)

    # If you want to disable outputs after moving (optional):
    # outputs_enable(False)

finally:
    # Safety cleanup
    GPIO.cleanup()
