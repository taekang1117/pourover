import RPi.GPIO as GPIO
import time
import threading

STEP = 17
DIR  = 27

GPIO.setwarnings(False)          # remove warnings
GPIO.setmode(GPIO.BCM)
GPIO.setup(STEP, GPIO.OUT)
GPIO.setup(DIR, GPIO.OUT)

GPIO.output(DIR, GPIO.HIGH)

running = True

def motor_loop():
    global running
    while running:
        GPIO.output(STEP, GPIO.HIGH)
        time.sleep(0.0002)
        GPIO.output(STEP, GPIO.LOW)
        time.sleep(0.0002)

# Run motor in background thread
t = threading.Thread(target=motor_loop)
t.start()

print("Motor spinning... press 'q' then Enter to stop")

# Wait for user input
while True:
    cmd = input()
    if cmd.lower() == 'q':
        running = False
        break

t.join()
GPIO.cleanup()
print("Motor stopped cleanly.")

