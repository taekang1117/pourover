import time
from adafruit_servokit import ServoKit

# ----------------------------
# PCA9685 / ServoKit Setup
# ----------------------------
kit = ServoKit(channels=16, address=0x40)
kit.frequency = 50

# Optional: expand pulse range if your servos move too little.
# Typical ranges: (500, 2500) or (600, 2400).
for ch in range(5):
    try:
        kit.servo[ch].set_pulse_width_range(500, 2500)
    except Exception:
        pass

# ----------------------------
# Channel Mapping (EDIT THIS)
# ----------------------------
BASE     = 4 # rotate left/right
SHOULDER = 3  # up/down
ELBOW    = 2  # up/down
WRIST    = 1  # wrist angle
GRIPPER  = 0  # open/close

# ----------------------------
# Timing / Motion Helpers
# ----------------------------
MOVE_STEP_DEG = 2
MOVE_STEP_S   = 0.01

def move_servo_smooth(channel: int, target: int, step: int = MOVE_STEP_DEG, delay: float = MOVE_STEP_S):
    """Smoothly move one servo to target angle."""
    cur = kit.servo[channel].angle
    if cur is None:
        # First command after boot: set directly
        kit.servo[channel].angle = target
        time.sleep(0.2)
        return

    cur = int(cur)
    target = int(target)
    if target == cur:
        return

    direction = 1 if target > cur else -1
    for a in range(cur, target + direction, direction * step):
        kit.servo[channel].angle = max(0, min(180, a))
        time.sleep(delay)

def move_pose(pose: dict, settle: float = 0.3):
    """Move multiple servos to a pose. Order matters for arms."""
    # A safe typical order: base -> shoulder -> elbow -> wrist -> gripper
    order = [BASE, SHOULDER, ELBOW, WRIST, GRIPPER]
    for ch in order:
        if ch in pose:
            move_servo_smooth(ch, pose[ch])
    time.sleep(settle)

# ----------------------------
# Gripper Helpers (EDIT THESE)
# ----------------------------
GRIP_OPEN_ANGLE  = 20   # adjust to your gripper
GRIP_CLOSE_ANGLE = 80   # adjust to your gripper

def gripper_open():
    move_servo_smooth(GRIPPER, GRIP_OPEN_ANGLE)
    time.sleep(0.2)

def gripper_close():
    move_servo_smooth(GRIPPER, GRIP_CLOSE_ANGLE)
    time.sleep(0.2)

# ----------------------------
# Calibrated Poses (YOU MUST TUNE THESE)
# ----------------------------
HOME = {
    BASE: 90,
    SHOULDER: 90,
    ELBOW: 90,
    WRIST: 90,
    GRIPPER: GRIP_OPEN_ANGLE
}

# These three poses represent:
# - approach above the target (safe height)
# - down at the target (grab height)
# - lift after closing gripper
#
# "9,4" is a label here; you must tune the angles so the end-effector is at your (x,y).
PICK_9_4_APPROACH = {
    BASE: 110,
    SHOULDER: 70,
    ELBOW: 120,
    WRIST: 90,
}

PICK_9_4_DOWN = {
    BASE: 110,
    SHOULDER: 82,
    ELBOW: 135,
    WRIST: 95,
}

PICK_9_4_LIFT = {
    BASE: 110,
    SHOULDER: 70,
    ELBOW: 120,
    WRIST: 90,
}

# ----------------------------
# Main Routine
# ----------------------------
def pick_at_9_4():
    print("Going HOME")
    move_pose(HOME)
    gripper_open()

    print("Approach (9,4)")
    move_pose(PICK_9_4_APPROACH)

    print("Down to grab")
    move_pose(PICK_9_4_DOWN)

    print("Close gripper")
    gripper_close()

    print("Lift")
    move_pose(PICK_9_4_LIFT)

    print("Return HOME")
    move_pose(HOME)

if __name__ == "__main__":
    try:
        pick_at_9_4()
        print("Done.")
    except KeyboardInterrupt:
        print("Stopped.")
