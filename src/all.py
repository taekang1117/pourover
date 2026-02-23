# main_fsm_rf_dualfeed.py
# Full FSM: INIT -> FEED -> SETTLE -> DETECT -> FLIP -> WAIT_CLEAR -> WEIGHT -> ARM -> FEED ...
# RandomForest inference + dual 12V TMC2209 feeders (STEP/DIR, EN=GND) + ULN2003 flipper
# Run: sudo python3 main_fsm_rf_dualfeed.py

import time
import cv2
import numpy as np
import pandas as pd
import pickle
import os
import sys
from enum import Enum, auto

from picamera2 import Picamera2
from rpi_ws281x import PixelStrip, Color

# ============================================================
# ===================== PIN MAP / HARDWARE ===================
# ============================================================

# --- 12V TMC2209 feeders (EN hard-grounded => always enabled) ---
M1_STEP = 17
M1_DIR  = 27

M2_STEP = 22
M2_DIR  = 23

# Direction "normal" for each motor (flip True/False if it runs backwards)
M1_DIR_NORMAL = True
M2_DIR_NORMAL = True

# --- ULN2003 Flipper (28BYJ-48 5V) ---
FLIP_IN1 = 5
FLIP_IN2 = 6
FLIP_IN3 = 13
FLIP_IN4 = 19

# ============================================================
# ===================== FEEDER TUNING ========================
# ============================================================

STEP_PULSE_US  = 20
STEP_GAP_US    = 2000
DIR_SETTLE_SEC = 0.05

DOSE_STEPS_M1  = 300   # tune
DOSE_STEPS_M2  = 300   # tune

FEEDER_SETTLE_SEC = 0.35  # after feeder movement, wait for vibration to stop

# Which feeder is used for the next FEED state (toggle with 'm')
AUTO_FEED_MOTOR = 1  # 1 or 2

# ============================================================
# ===================== FLIPPER TUNING =======================
# ============================================================

FLIP_STEP_DELAY = 0.0018
STEPS_135_DEG   = 768
RETURN_WAIT_SEC = 0.5

# Which way to flip for each decision
# +1 means "right", -1 means "left"
FLIP_DIR_FOR_BEAN = +1
FLIP_DIR_FOR_ROCK = -1

# ============================================================
# ===================== WS2812 / LED =========================
# ============================================================

LED_COUNT = 7
LED_PIN = 18
LED_FREQ_HZ = 800000
LED_DMA = 10
LED_INVERT = False
LED_BRIGHTNESS = 255
LED_CHANNEL = 0

strip = PixelStrip(LED_COUNT, LED_PIN, LED_FREQ_HZ, LED_DMA, LED_INVERT,
                   LED_BRIGHTNESS, LED_CHANNEL)

def set_max_white():
    strip.begin()
    white = Color(255, 255, 255)
    for i in range(LED_COUNT):
        strip.setPixelColor(i, white)
    strip.show()

def set_led_off():
    off = Color(0, 0, 0)
    for i in range(LED_COUNT):
        strip.setPixelColor(i, off)
    strip.show()

# ============================================================
# ===================== VISION CONFIG ========================
# ============================================================

FRAME_W, FRAME_H = 960, 540
ROI_X, ROI_Y, ROI_W, ROI_H = 280, 23, 431, 450

BLUR_K = 5
MORPH_K = 5
OPEN_ITERS = 2
CLOSE_ITERS = 2

MIN_AREA = 1000
MAX_AREA = 8000

MODEL_FILE = "bean_model.pkl"

FEATURE_COLS_16 = [
    "area", "aspect_ratio", "circularity", "solidity", "perimeter",
    "mean_hue", "mean_saturation",
    "gabor_mean", "gabor_std",
    "hu1", "hu2", "hu3", "hu4", "hu5", "hu6", "hu7",
]
FEATURE_COLS_5 = ["area", "aspect_ratio", "circularity", "solidity", "perimeter"]

GABOR_KERNEL = cv2.getGaborKernel((9, 9), 3.0, np.pi / 4, 8.0, 0.5, 0, ktype=cv2.CV_32F)

# ============================================================
# ===================== FSM TUNING ===========================
# ============================================================

# DETECT stability: require same decision N consecutive frames
DECISION_STREAK_N = 3

# In DETECT, if no object for too long -> re-feed
DETECT_TIMEOUT_SEC = 4.0

# WAIT_CLEAR stability: require zero objects N consecutive frames
CLEAR_STREAK_N = 4
CLEAR_TIMEOUT_SEC = 4.0

# If you want to skip WEIGHT/ARM for now but keep structure:
ENABLE_WEIGHT = False
ENABLE_ARM = False

# ============================================================
# ===================== HELPERS: GPIO STEPPER ================
# ============================================================

class StepperTMC2209_EN_GND:
    """STEP/DIR stepper where EN is wired to GND (always enabled)."""
    def __init__(self, step_pin, dir_pin, dir_normal=True, name="M"):
        self.step_pin = step_pin
        self.dir_pin = dir_pin
        self.dir_normal = dir_normal
        self.name = name

    def set_dir(self, forward: bool):
        import RPi.GPIO as GPIO
        level = GPIO.HIGH if (forward == self.dir_normal) else GPIO.LOW
        GPIO.output(self.dir_pin, level)
        time.sleep(DIR_SETTLE_SEC)

    def step_n(self, steps: int, step_gap_us=STEP_GAP_US, pulse_us=STEP_PULSE_US):
        import RPi.GPIO as GPIO
        pulse_s = pulse_us / 1_000_000.0
        gap_s = step_gap_us / 1_000_000.0
        for _ in range(int(steps)):
            GPIO.output(self.step_pin, GPIO.HIGH)
            time.sleep(pulse_s)
            GPIO.output(self.step_pin, GPIO.LOW)
            time.sleep(gap_s)

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

def move_to(stepper, target_pos, current_pos):
    delta = int(target_pos - current_pos)
    if delta == 0:
        return current_pos
    stepper.step(abs(delta), direction=+1 if delta > 0 else -1)
    return target_pos

def run_flip(stepper, current_pos, direction):
    start_pos = current_pos
    target_pos = start_pos + (STEPS_135_DEG if direction > 0 else -STEPS_135_DEG)
    current_pos = move_to(stepper, target_pos, current_pos)
    time.sleep(RETURN_WAIT_SEC)
    current_pos = move_to(stepper, start_pos, current_pos)
    stepper.release()
    return current_pos

# ============================================================
# ===================== HELPERS: CV / ML =====================
# ============================================================

def load_model():
    if not os.path.exists(MODEL_FILE):
        print(f"ERROR: {MODEL_FILE} not found. Run train_model.py first.")
        return None
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    print(f"Loaded model from {MODEL_FILE}")
    return model

def get_model_feature_cols(model):
    if hasattr(model, "feature_names_in_"):
        cols = [str(c) for c in model.feature_names_in_]
        if cols:
            return cols
    n_features = int(getattr(model, "n_features_in_", 0))
    if n_features == len(FEATURE_COLS_16):
        return FEATURE_COLS_16
    if n_features == len(FEATURE_COLS_5):
        return FEATURE_COLS_5
    raise ValueError(f"Unsupported model feature count: {n_features}")

def clamp_roi(x, y, w, h, W, H):
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    return x, y, w, h

def morph_cleanup(mask):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_K, MORPH_K))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=OPEN_ITERS)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=CLOSE_ITERS)
    return mask

def capture_background_gray(picam2, roi_rect, n=20):
    rx, ry, rw, rh = roi_rect
    acc = None
    for _ in range(n):
        frame_rgb = picam2.capture_array()
        full_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        roi = full_bgr[ry:ry+rh, rx:rx+rw]
        g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
        g = cv2.GaussianBlur(g, (BLUR_K, BLUR_K), 0)
        acc = g if acc is None else acc + g
        time.sleep(0.02)
    return (acc / n).astype(np.uint8)

def get_object_mask(roi_bgr, bg_gray):
    g1 = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    g1 = cv2.GaussianBlur(g1, (BLUR_K, BLUR_K), 0)
    diff = cv2.subtract(bg_gray, g1)  # suppress reflections
    _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return morph_cleanup(mask)

def is_valid_contour(cnt):
    a = cv2.contourArea(cnt)
    return MIN_AREA <= a <= MAX_AREA

def get_features_dict(cnt, roi_bgr):
    area = float(cv2.contourArea(cnt))
    perim = float(cv2.arcLength(cnt, True))
    if perim == 0:
        return None

    circularity = (4.0 * np.pi * area) / (perim * perim)

    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull))
    solidity = area / hull_area if hull_area > 0 else 0

    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio_invariant = float(max(w, h)) / (min(w, h) + 1e-9)

    obj_mask = np.zeros(roi_bgr.shape[:2], dtype=np.uint8)
    cv2.drawContours(obj_mask, [cnt], -1, 255, thickness=-1)

    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    mean_h, mean_s, _ = cv2.mean(hsv, mask=obj_mask)[:3]

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gabor_resp = cv2.filter2D(gray, cv2.CV_32F, GABOR_KERNEL)
    gabor_abs = np.abs(gabor_resp)
    gabor_mean, gabor_std = cv2.meanStdDev(gabor_abs, mask=obj_mask)
    gabor_mean = float(gabor_mean[0][0])
    gabor_std = float(gabor_std[0][0])

    hu = cv2.HuMoments(cv2.moments(cnt)).flatten()
    hu_log = [float(-np.sign(v) * np.log10(abs(v) + 1e-12)) for v in hu]

    return {
        "area": area,
        "aspect_ratio": aspect_ratio_invariant,
        "circularity": circularity,
        "solidity": solidity,
        "perimeter": perim,
        "mean_hue": float(mean_h),
        "mean_saturation": float(mean_s),
        "gabor_mean": gabor_mean,
        "gabor_std": gabor_std,
        "hu1": hu_log[0],
        "hu2": hu_log[1],
        "hu3": hu_log[2],
        "hu4": hu_log[3],
        "hu5": hu_log[4],
        "hu6": hu_log[5],
        "hu7": hu_log[6],
    }

# ============================================================
# ===================== FSM DEFINITIONS ======================
# ============================================================

class State(Enum):
    INIT = auto()
    FEED = auto()
    SETTLE = auto()
    DETECT = auto()
    FLIP = auto()
    WAIT_CLEAR = auto()
    WEIGHT = auto()
    ARM = auto()

class Decision(Enum):
    NONE = auto()
    BEAN = auto()
    ROCK = auto()
    UNCERTAIN = auto()

def majority_decision(preds):
    """preds are 0/1. Return Decision.BEAN/ROCK."""
    if len(preds) == 0:
        return Decision.NONE
    beans = int(np.sum(preds == 1))
    rocks = int(np.sum(preds == 0))
    if beans == 0 and rocks == 0:
        return Decision.NONE
    return Decision.BEAN if beans >= rocks else Decision.ROCK

# ============================================================
# ===================== PLACEHOLDERS =========================
# ============================================================

def do_weight_check():
    """Placeholder for HX711 weighing. Return True if ready to proceed to ARM, else False."""
    # Example:
    # w = hx711.get_units(10)
    # if w >= target: return True
    # else: return False
    return True

def do_arm_action():
    """Placeholder for arm routine."""
    # Example: arm.pick_and_place()
    time.sleep(0.2)

# ============================================================
# ============================ MAIN ==========================
# ============================================================

def main():
    global AUTO_FEED_MOTOR

    model = load_model()
    if model is None:
        sys.exit(1)

    try:
        model_cols = get_model_feature_cols(model)
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    print(f"Using {len(model_cols)} features: {model_cols}")

    # ---------- GPIO init ----------
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)

    for p in [M1_STEP, M1_DIR, M2_STEP, M2_DIR]:
        GPIO.setup(p, GPIO.OUT)
        GPIO.output(p, GPIO.LOW)

    m1 = StepperTMC2209_EN_GND(M1_STEP, M1_DIR, dir_normal=M1_DIR_NORMAL, name="M1")
    m2 = StepperTMC2209_EN_GND(M2_STEP, M2_DIR, dir_normal=M2_DIR_NORMAL, name="M2")

    flip = ULN2003Stepper(FLIP_IN1, FLIP_IN2, FLIP_IN3, FLIP_IN4, step_delay=FLIP_STEP_DELAY)
    flip_pos = 0

    # ---------- Camera init ----------
    roi_rect = clamp_roi(ROI_X, ROI_Y, ROI_W, ROI_H, FRAME_W, FRAME_H)
    rx, ry, rw, rh = roi_rect

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (FRAME_W, FRAME_H)})
    picam2.configure(config)
    picam2.start()

    time.sleep(1.5)
    try:
        picam2.set_controls({"AeEnable": False, "AwbEnable": False})
    except:
        pass

    set_max_white()

    # ---------- FSM vars ----------
    state = State.INIT
    bg_gray = None

    t_state_enter = time.time()
    t_detect_start = 0.0
    t_clear_start = 0.0

    last_decision = Decision.NONE
    decision_streak = 0

    clear_streak = 0

    # Runtime overlay / debug
    beans_count = 0
    rocks_count = 0
    visible_count = 0

    def goto(new_state):
        nonlocal state, t_state_enter
        state = new_state
        t_state_enter = time.time()
        print(f"[FSM] -> {state.name}")

    print("\n=== FSM sorter (RF + dual feeders + flipper) ===")
    print("Keys:")
    print("  q : quit")
    print("  b : capture background now (plate empty)")
    print("  m : toggle AUTO_FEED_MOTOR (1<->2)")
    print("  1 : manual dose M1 (debug)")
    print("  2 : manual dose M2 (debug)")
    print("  r/l : manual flipper right/left (debug)\n")

    goto(State.INIT)

    try:
        while True:
            # --- grab frame ---
            frame_rgb = picam2.capture_array()
            full_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            cv2.rectangle(full_bgr, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 2)
            roi_bgr = full_bgr[ry:ry + rh, rx:rx + rw]
            vis_roi = roi_bgr.copy()

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                set_led_off()
                break
            elif key == ord('b'):
                print("[MANUAL] Capturing background...")
                bg_gray = capture_background_gray(picam2, roi_rect)
                print("[MANUAL] Background captured.")
            elif key == ord('m'):
                AUTO_FEED_MOTOR = 2 if AUTO_FEED_MOTOR == 1 else 1
                print(f"[MODE] AUTO_FEED_MOTOR -> {AUTO_FEED_MOTOR}")
            elif key == ord('1'):
                print("[MANUAL] dose M1")
                m1.set_dir(True)
                m1.step_n(DOSE_STEPS_M1)
            elif key == ord('2'):
                print("[MANUAL] dose M2")
                m2.set_dir(True)
                m2.step_n(DOSE_STEPS_M2)
            elif key == ord('r'):
                flip_pos = run_flip(flip, flip_pos, direction=+1)
            elif key == ord('l'):
                flip_pos = run_flip(flip, flip_pos, direction=-1)

            # --- compute mask/contours only if we have bg ---
            mask = None
            contours = []
            valid_contours = []
            coords = []

            beans_count = 0
            rocks_count = 0
            visible_count = 0

            if bg_gray is not None:
                mask = get_object_mask(roi_bgr, bg_gray)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    if not is_valid_contour(cnt):
                        continue
                    valid_contours.append(cnt)
                    coords.append(cv2.boundingRect(cnt))

                visible_count = len(valid_contours)

            # --- Draw mask ---
            if mask is not None:
                cv2.imshow("Mask", mask)

            # =====================================================
            # ===================== FSM LOGIC ======================
            # =====================================================

            now = time.time()

            if state == State.INIT:
                # Need background first
                if bg_gray is None:
                    cv2.putText(full_bgr, "FSM INIT: Press 'b' capture BACKGROUND (empty plate)",
                                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    goto(State.FEED)

            elif state == State.FEED:
                # Dose from selected feeder
                feeder = m1 if AUTO_FEED_MOTOR == 1 else m2
                steps = DOSE_STEPS_M1 if AUTO_FEED_MOTOR == 1 else DOSE_STEPS_M2

                print(f"[FEED] M{AUTO_FEED_MOTOR} steps={steps}")
                feeder.set_dir(True)
                feeder.step_n(steps)

                goto(State.SETTLE)

            elif state == State.SETTLE:
                # quiet window so image/features stabilize
                if (now - t_state_enter) >= FEEDER_SETTLE_SEC:
                    # reset detect trackers
                    t_detect_start = now
                    last_decision = Decision.NONE
                    decision_streak = 0
                    goto(State.DETECT)

            elif state == State.DETECT:
                # if no bg somehow, return init
                if bg_gray is None:
                    goto(State.INIT)
                else:
                    # If nothing visible for too long, re-feed
                    if visible_count == 0:
                        if (now - t_detect_start) > DETECT_TIMEOUT_SEC:
                            print("[DETECT] timeout (no object) -> FEED")
                            goto(State.FEED)
                    else:
                        # Extract features and predict for all objects
                        feature_rows = []
                        for cnt in valid_contours:
                            feats = get_features_dict(cnt, roi_bgr)
                            if feats is None:
                                continue
                            feature_rows.append({c: feats.get(c, 0.0) for c in model_cols})

                        if feature_rows:
                            df = pd.DataFrame(feature_rows, columns=model_cols)
                            preds = model.predict(df)
                            # count for display
                            beans_count = int(np.sum(preds == 1))
                            rocks_count = int(np.sum(preds == 0))

                            decision = majority_decision(preds)

                            # Stability / streak logic
                            if decision == last_decision:
                                decision_streak += 1
                            else:
                                last_decision = decision
                                decision_streak = 1

                            if decision_streak >= DECISION_STREAK_N:
                                print(f"[DETECT] stable decision={decision.name} streak={decision_streak}")
                                goto(State.FLIP)
                        else:
                            # no usable features -> keep waiting but respect timeout
                            if (now - t_detect_start) > DETECT_TIMEOUT_SEC:
                                print("[DETECT] timeout (no features) -> FEED")
                                goto(State.FEED)

            elif state == State.FLIP:
                # Execute flip based on last stable decision
                if last_decision == Decision.BEAN:
                    flip_dir = FLIP_DIR_FOR_BEAN
                elif last_decision == Decision.ROCK:
                    flip_dir = FLIP_DIR_FOR_ROCK
                else:
                    # unexpected: if NONE/UNCERTAIN, just don't flip, go refuel
                    print(f"[FLIP] decision={last_decision.name} -> no flip -> FEED")
                    goto(State.FEED)
                    flip_dir = 0

                if flip_dir != 0:
                    print(f"[FLIP] direction={flip_dir} (BEAN->{FLIP_DIR_FOR_BEAN}, ROCK->{FLIP_DIR_FOR_ROCK})")
                    flip_pos = run_flip(flip, flip_pos, direction=flip_dir)

                # start clear tracking
                clear_streak = 0
                t_clear_start = now
                goto(State.WAIT_CLEAR)

            elif state == State.WAIT_CLEAR:
                if bg_gray is None:
                    goto(State.INIT)
                else:
                    if visible_count == 0:
                        clear_streak += 1
                    else:
                        clear_streak = 0

                    if clear_streak >= CLEAR_STREAK_N:
                        print("[CLEAR] plate is clear")
                        if ENABLE_WEIGHT:
                            goto(State.WEIGHT)
                        elif ENABLE_ARM:
                            goto(State.ARM)
                        else:
                            goto(State.FEED)
                    else:
                        if (now - t_clear_start) > CLEAR_TIMEOUT_SEC:
                            print("[CLEAR] timeout -> proceed anyway")
                            if ENABLE_WEIGHT:
                                goto(State.WEIGHT)
                            elif ENABLE_ARM:
                                goto(State.ARM)
                            else:
                                goto(State.FEED)

            elif state == State.WEIGHT:
                ok = do_weight_check()
                if ok:
                    goto(State.ARM if ENABLE_ARM else State.FEED)
                else:
                    goto(State.FEED)

            elif state == State.ARM:
                do_arm_action()
                goto(State.FEED)

            # --- overlay ---
            header = f"FSM:{state.name} | AutoFeed:M{AUTO_FEED_MOTOR} | Visible:{visible_count} | B:{beans_count} R:{rocks_count}"
            cv2.putText(full_bgr, header, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            if bg_gray is None:
                cv2.putText(full_bgr, "No background. Press 'b' with empty plate.",
                            (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Draw predicted boxes (only for visualization)
            for i, (x, y, w, h) in enumerate(coords):
                cv2.rectangle(vis_roi, (x, y), (x+w, y+h), (255, 255, 0), 2)

            cv2.imshow("Review", full_bgr)
            cv2.imshow("Inference", vis_roi)

    finally:
        try:
            picam2.stop()
        except:
            pass
        cv2.destroyAllWindows()
        try:
            flip.release()
        except:
            pass
        try:
            import RPi.GPIO as GPIO
            GPIO.cleanup()
        except:
            pass
        set_led_off()
        print("Clean exit.")

if __name__ == "__main__":
    main()
