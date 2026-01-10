import RPi.GPIO as GPIO
import time

# Use Broadcom pin numbering
GPIO.setmode(GPIO.BCM)

# Define the GPIO pins as per your setup
# Note: GPIO 1 is physical pin 28, GPIO 2 is pin 3, etc. 
# Ensure these match your actual wiring.
pins = [1, 2, 3, 4]

# Set all pins as output
for pin in pins:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, False)

# Sequence for Clockwise rotation (4-step)
# Each sub-list represents [Pin1, Pin2, Pin3, Pin4]
sequence = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
]

def rotate_clockwise(steps, delay=0.005):
    for i in range(steps):
        for step in sequence:
            for pin_index in range(4):
                GPIO.output(pins[pin_index], step[pin_index])
            time.sleep(delay)

try:
    print("Rotating clockwise...")
    # Adjust 512 based on your motor's gear ratio (usually 512 for 28BYJ-48)
    rotate_clockwise(512)
    print("Done!")

except KeyboardInterrupt:
    print("Stopping...")
finally:
    GPIO.cleanup()
