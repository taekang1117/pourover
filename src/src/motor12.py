import RPi.GPIO as GPIO
import time

STEP = 17   # GPIO17 (pin 11)
DIR  = 27   # GPIO27 (pin 13)

GPIO.setmode(GPIO.BCM)
GPIO.setup(STEP, GPIO.OUT)
GPIO.setup(DIR, GPIO.OUT)

GPIO.output(DIR, GPIO.HIGH)  # set direction

print("Motor spinning forever...")

while True:
    GPIO.output(STEP, GPIO.HIGH)
    time.sleep(0.001)   # speed control
    GPIO.output(STEP, GPIO.LOW)
    time.sleep(0.001)
