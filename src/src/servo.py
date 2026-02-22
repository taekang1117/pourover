import time
from adafruit_servokit import ServoKit

kit = ServoKit(channels=16, address=0x40)
kit.frequency = 50

for ch in range(5):
    try:
        kit.servo[ch].set_pulse_width_range(500, 2500)
    except Exception:
        pass

BASE = 0
SHOULDER = 1
ELBOW = 2
WRIST = 3
GRIPPER = 4

MOVE_STEP = 2
MOVE_DELAY = 0.01

def move(channel, target):
    cur = kit.servo[channel].angle
    if cur is None:
        kit.servo[channel].angle = target
        time.sleep(0.2)
        return
    cur = int(cur)
    target = int(target)
    if cur == target:
        return
    step = 1 if target > cur else -1
    for a in range(cur, target + step, step * MOVE_STEP):
        kit.servo[channel].angle = a
        time.sleep(MOVE_DELAY)

def pose(p):
    for ch in [BASE, SHOULDER, ELBOW, WRIST, GRIPPER]:
        if ch in p:
            move(ch, p[ch])
    time.sleep(0.3)

GRIP_OPEN = 20
GRIP_CLOSE = 80

RESET = {
    BASE: 0,
    SHOULDER: 40,
    ELBOW: 160,
    WRIST: 20,
    GRIPPER: GRIP_OPEN
}

HOME = {
    BASE: 90,
    SHOULDER: 90,
    ELBOW: 90,
    WRIST: 90,
    GRIPPER: GRIP_OPEN
}

PICK_APPROACH = {
    BASE: 110,
    SHOULDER: 70,
    ELBOW: 120,
    WRIST: 90
}

PICK_DOWN = {
    BASE: 110,
    SHOULDER: 82,
    ELBOW: 135,
    WRIST: 95
}

PICK_LIFT = {
    BASE: 110,
    SHOULDER: 70,
    ELBOW: 120,
    WRIST: 90
}

def main():
    print("RESET")
    pose(RESET)
    time.sleep(0.5)

    print("HOME")
    pose(HOME)
    time.sleep(0.5)

    print("APPROACH")
    pose(PICK_APPROACH)

    print("DOWN")
    pose(PICK_DOWN)

    print("GRAB")
    move(GRIPPER, GRIP_CLOSE)
    time.sleep(0.5)

    print("LIFT")
    pose(PICK_LIFT)

    print("RETURN HOME")
    pose(HOME)

if __name__ == "__main__":
    main()
