#!/usr/bin/env python3
# combo_sorter.py
# Unified: TMC2209 feeders (STEP/DIR) + PiCamera2 CV + WS2812 light + ULN2003 flipper
#
# Run:
#   sudo /home/pi/venv/bin/python combo_sorter.py
#
# Keys (in OpenCV window):
#   q : quit
#   b : capture background (ROI must be empty)
#   1 : manual dose feeder motor 1
#   2 : manual dose feeder motor 2
#   r : re-center flipper (move to 0)
#
# Notes:
# - TMC2209 EN is active-low: LOW=enable, HIGH=disable
# - All grounds must be common (Pi GND, ULN2003 GND, TMC logic GND, 12V-)

import time
import os
import sys
import cv2
import numpy as np
import pickle
from picamera2 import Picamera2
from rpi_ws281x import PixelStrip, Color
import RPi.GPIO as GPIO

# ============================================================
# ===================== GPIO PIN MAP =========================
# ============================================================

# ---- TMC2209 Motor 1 ----
M1_STEP = 17
M1_DIR  = 27
M1_EN   = 24

# ---- TMC2209 Motor 2 ----
M2_STEP = 22
M2_DIR  = 23
M2_EN   = 25

# ---- ULN2003 Flipper (28BYJ-48) ----
FLIP_IN1 = 5
FLIP_IN2 = 6
FLIP_IN3 = 13
FLIP_IN4 = 19

# ---- WS2812 LED ----
LED_COUNT = 7
LED_PIN = 18
LED_FREQ_HZ = 800000
LED_DMA = 10
LED_INVERT = False
LED_BRIGHTNESS = 255
LED_CHANNEL = 0

# ============================================================
# ===================== FEEDER TUNING ========================
# ============================================================

STEP_PULSE_US = 8       # STEP high time
STEP_GAP_US   = 800     # speed: 800us gap -> ~1250 steps/s
DOSE_STEPS    = 300     # tune for "1-2 beans"
SETTLE_SEC_FEED = 0.25  # pause after dosing

M1_DIR_NORMAL = True
M2_DIR_NORMAL = True

# Choose which feeder motor auto-feeds each cycle:
AUTO_FEED_MOTOR = 1  # 1 or 2

# ============================================================
# ===================== FLIPPER TUNING =======================
# ============================================================

FLIP_STEP_DELAY = 0.0018

POS_RIGHT = +650     # BEAN side (RIGHT)
POS_LEFT  = -650     # ROCK side (LEFT)
DROP_WAIT_SEC = 0.35

# ============================================================
# ===================== VISION CONFIG ========================
# ============================================================

FRAME_W, FRAME_H = 960, 540
ROI_X, ROI_Y, ROI_W, ROI_H = 260, 90, 440, 360

BLUR_K = 7
MORPH_K = 5
OPEN_ITERS = 2
CLOSE_ITERS = 2

MIN_AREA = 800
MAX_AREA = 40000

MODEL_FILE = "bean_model.pkl"

# Presence / stability tuning
PRESENT_THRESH = 3000
CLEAR_THRESH   = 1500
STABLE_FRAMES  = 6
SETTLE_SEC_CV  = 0.08

# WAIT_CLEAR timeout (prevents stuck state)
MAX_WAIT_CLEAR_SEC = 2.0

# ============================================================
# ===================== WS2812 HELPERS =======================
# ============================================================

strip = PixelStrip(LED_COUNT, LED_PIN, LED_FREQ_HZ, LED_DMA,
                   LED_INVERT, LED_BRIGHTNESS, LED_CHANNEL)

def set_max_white():
    strip.begin()
    white = Color(255, 255, 255)
    for i in range(LED_COUNT):
        strip.setPixelColor(i, white)
    strip.show()

# ============================================================
# ===================== TMC2209 STEPPER ======================
# ============================================================

class StepperTMC2209:
    def __init__(self, step_pin, dir_pin, en_pin, dir_normal=True, name="M"):
        self.step_pin = step_pin
        self.dir_pin  = dir_pin
        self.en_pin   = en_pin
        self.dir_normal = dir_normal
        self.name = name

    def enable(self):
        GPIO.output(self.en_pin, GPIO.LOW)   # active-low

    def disable(self):
        GPIO.output(self.en_pin, GPIO.HIGH)

    def set_dir(self, forward: bool):
        level = GPIO.HIGH if (forward == self.dir_normal) else GPIO.LOW
        GPIO.output(self.dir_pin, level)

    def step_n(self, steps: int, step_gap_us=STEP_GAP_US, pulse_us=STEP_PULSE_US):
        self.enable()
        time.sleep(0.002)  # driver enable settle

        pulse_s = pulse_us / 1_000_000.0
        gap_s   = step_gap_us / 1_000_000.0

        for _ in range(int(steps)):
            GPIO.output(self.step_pin, GPIO.HIGH)
            time.sleep(pulse_s)
            GPIO.output(self.step_pin, GPIO.LOW)
            time.sleep(gap_s)

        self.disable()

# ============================================================
# ===================== ULN2003 STEPPER ======================
# ============================================================

class ULN2003Stepper:
    HALF_SEQ = [
        (1,0,0,0),
        (1,1,0,0),
        (0,1,0,0),
        (0,1,1,0),
        (0,0,1,0),
        (0,0,1,1),
        (0,0,0,1),
        (1,0,0,1),
    ]

    def __init__(self, in1, in2, in3, in4, step_delay=0.002):
        self.pins = [in1, in2, in3, in4]
        self.step_delay = step_delay
        self._idx = 0
        for p in self.pins:
            GPIO.setup(p, GPIO.OUT)
            GPIO.output(p, 0)

    def _write(self, a, b, c, d):
        GPIO.output(self.pins[0], a)
        GPIO.output(self.pins[1], b)
        GPIO.output(self.pins[2], c)
        GPIO.output(self.pins[3], d)

    def step(self, steps, direction=1):
        direction = 1 if direction >= 0 else -1
        for _ in range(abs(int(steps))):
            self._idx = (self._idx + direction) % len(self.HALF_SEQ)
            self._write(*self.HALF_SEQ[self._idx])
            time.sleep(self.step_delay)

    def release(self):
        self._write(0, 0, 0, 0)

def move_to(stepper: ULN2003Stepper, target_pos: int, current_pos: int) -> int:
    delta = int(target_pos - current_pos)
    if delta == 0:
        return current_pos
    stepper.step(abs(delta), direction=+1 if delta > 0 else -1)
    return target_pos

# ============================================================
# ===================== CV HELPERS ===========================
# ============================================================

def load_model():
    if not os.path.exists(MODEL_FILE):
        print(f"ERROR: {MODEL_FILE} not found in current directory!")
        return None
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    print(f"Loaded model from {MODEL_FILE}")
    return model

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
    diff = cv2.absdiff(g1, bg_gray)
    _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = morph_cleanup(mask)
    return mask

def get_features_vector(cnt):
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

    return [area, aspect_ratio_invariant, circularity, solidity, perim]

# ============================================================
# ===================== MAIN (FSM) ===========================
# ============================================================

def main():
    model = load_model()
    if model is None:
        sys.exit(1)

    # ---- GPIO init ----
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)

    # Setup TMC pins
    tmc_pins = [M1_STEP, M1_DIR, M1_EN, M2_STEP, M2_DIR, M2_EN]
    for p in tmc_pins:
        GPIO.setup(p, GPIO.OUT)
    GPIO.output(M1_STEP, GPIO.LOW)
    GPIO.output(M2_STEP, GPIO.LOW)
    GPIO.output(M1_EN, GPIO.HIGH)  # disabled
    GPIO.output(M2_EN, GPIO.HIGH)  # disabled

    m1 = StepperTMC2209(M1_STEP, M1_DIR, M1_EN, dir_normal=M1_DIR_NORMAL, name="M1")
    m2 = StepperTMC2209(M2_STEP, M2_DIR, M2_EN, dir_normal=M2_DIR_NORMAL, name="M2")

    # Setup ULN2003 pins
    stepper_flip = ULN2003Stepper(FLIP_IN1, FLIP_IN2, FLIP_IN3, FLIP_IN4,
                                  step_delay=FLIP_STEP_DELAY)
    flip_pos = 0

    # ---- Camera init ----
    roi_rect = clamp_roi(ROI_X, ROI_Y, ROI_W, ROI_H, FRAME_W, FRAME_H)
    rx, ry, rw, rh = roi_rect

    picam2 = Picamera2()
    set_max_white()

    config = picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (FRAME_W, FRAME_H)}
    )
    picam2.configure(config)
    picam2.start()

    time.sleep(1.5)
    try:
        picam2.set_controls({"AeEnable": False, "AwbEnable": False})
    except:
        pass

    bg_gray = None

    # ---- Unified FSM ----
    state = "FEED"
    stable_cnt = 0
    last_label = None
    beans_max = 0
    rocks_max = 0
    clear_start_t = 0.0

    print("\n=== Unified Sorter Mode ===")
    print("Keys: b=background | 1/2=manual dose | r=re-center | q=quit")
    print(f"AUTO_FEED_MOTOR={AUTO_FEED_MOTOR}, DOSE_STEPS={DOSE_STEPS}, STEP_GAP_US={STEP_GAP_US}")

    try:
        while True:
            # ========== FEED / SETTLE states don't need camera ==========
            if state == "FEED":
                if bg_gray is None:
                    # Don't feed until background captured, otherwise CV mask is meaningless
                    state = "WAIT_BG"
                else:
                    feeder = m1 if AUTO_FEED_MOTOR == 1 else m2
                    print(f"[FEED] Motor {AUTO_FEED_MOTOR} dose ({DOSE_STEPS} steps)")
                    feeder.set_dir(True)
                    feeder.step_n(DOSE_STEPS)
                    state = "SETTLE"
                    settle_t = time.time()

            if state == "SETTLE":
                if (time.time() - settle_t) >= SETTLE_SEC_FEED:
                    state = "WAIT_OBJECT"

            # ========== Camera frame ==========
            frame_rgb = picam2.capture_array()
            full_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            cv2.rectangle(full_bgr, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 2)
            roi_bgr = full_bgr[ry:ry + rh, rx:rx + rw]
            vis_roi = roi_bgr.copy()

            beans_count = 0
            rocks_count = 0
            mask_area = 0

            if bg_gray is None:
                cv2.putText(full_bgr, "Press 'b' for BACKGROUND (empty ROI)",
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            else:
                mask = get_object_mask(roi_bgr, bg_gray)
                mask_area = int(cv2.countNonZero(mask))
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                feature_list = []
                coords = []

                for cnt in contours:
                    a = cv2.contourArea(cnt)
                    if a < MIN_AREA or a > MAX_AREA:
                        continue
                    vec = get_features_vector(cnt)
                    if vec:
                        feature_list.append(vec)
                        coords.append(cv2.boundingRect(cnt))

                if feature_list:
                    preds = model.predict(feature_list)
                    for i, label in enumerate(preds):
                        x, y, w, h = coords[i]
                        if label == 1:
                            beans_count += 1
                            color = (0, 255, 0)
                            text = "BEAN"
                        else:
                            rocks_count += 1
                            color = (0, 0, 255)
                            text = "ROCK"
                        cv2.rectangle(vis_roi, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(vis_roi, text, (x, y - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                header = f"Beans: {beans_count} | Rocks: {rocks_count}"
                cv2.putText(full_bgr, header, (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

                # ===== Unified FSM (vision-driven part) =====
                if state == "WAIT_BG":
                    # just wait for 'b'
                    pass

                elif state == "WAIT_OBJECT":
                    stable_cnt = 0
                    last_label = None
                    beans_max = 0
                    rocks_max = 0
                    if mask_area > PRESENT_THRESH:
                        state = "CLASSIFY"

                elif state == "CLASSIFY":
                    if mask_area > PRESENT_THRESH:
                        stable_cnt += 1
                        beans_max = max(beans_max, beans_count)
                        rocks_max = max(rocks_max, rocks_count)
                    else:
                        stable_cnt = 0
                        beans_max = 0
                        rocks_max = 0
                        state = "WAIT_OBJECT"

                    if stable_cnt >= STABLE_FRAMES:
                        time.sleep(SETTLE_SEC_CV)

                        # Rule: rock wins (covers "both present")
                        if rocks_max > 0:
                            last_label = "ROCK"
                            state = "FLIP"
                        elif beans_max > 0:
                            last_label = "BEAN"
                            state = "FLIP"
                        else:
                            stable_cnt = 0
                            beans_max = 0
                            rocks_max = 0
                            state = "WAIT_OBJECT"

                elif state == "FLIP":
                    if last_label == "BEAN":
                        print("Flip RIGHT (BEAN)")
                        flip_pos = move_to(stepper_flip, POS_RIGHT, flip_pos)
                    else:
                        print("Flip LEFT (ROCK)")
                        flip_pos = move_to(stepper_flip, POS_LEFT, flip_pos)

                    time.sleep(DROP_WAIT_SEC)

                    flip_pos = move_to(stepper_flip, 0, flip_pos)
                    stepper_flip.release()

                    clear_start_t = time.time()
                    state = "WAIT_CLEAR"

                elif state == "WAIT_CLEAR":
                    # Re-arm when cleared OR after timeout
                    if mask_area < CLEAR_THRESH or (time.time() - clear_start_t) > MAX_WAIT_CLEAR_SEC:
                        state = "FEED"

                # Debug overlay
                cv2.putText(full_bgr,
                            f"State:{state} mask={mask_area} stable={stable_cnt} bMax={beans_max} rMax={rocks_max}",
                            (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

                cv2.imshow("Mask", mask)

            cv2.imshow("Review", full_bgr)
            cv2.imshow("Inference", vis_roi)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord('b'):
                print("Capturing background (make sure ROI is empty)...")
                bg_gray = capture_background_gray(picam2, roi_rect)
                print("Background captured.")
                state = "FEED"
                stable_cnt = 0
                last_label = None
                beans_max = 0
                rocks_max = 0

            elif key == ord('1'):
                print("[MANUAL] dose M1")
                m1.set_dir(True)
                m1.step_n(DOSE_STEPS)

            elif key == ord('2'):
                print("[MANUAL] dose M2")
                m2.set_dir(True)
                m2.step_n(DOSE_STEPS)

            elif key == ord('r'):
                print("[MANUAL] flipper -> center")
                flip_pos = move_to(stepper_flip, 0, flip_pos)
                stepper_flip.release()

    finally:
        # Safety shutdown
        try:
            picam2.stop()
        except:
            pass
        cv2.destroyAllWindows()

        # Disable TMC drivers
        GPIO.output(M1_EN, GPIO.HIGH)
        GPIO.output(M2_EN, GPIO.HIGH)

        # Release flipper coils
        try:
            stepper_flip.release()
        except:
            pass

        GPIO.cleanup()
        print("\nClean exit. Drivers disabled, GPIO cleaned up.")

if __name__ == "__main__":
    main()
