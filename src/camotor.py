# camotor.py - RandomForest Inference + WS2812 Light + ULN2003 Flipper
# FSM:
#   WAIT_OBJECT -> CLASSIFY -> FLIP -> WAIT_CLEAR -> WAIT_OBJECT
# WAIT_CLEAR has TIMEOUT so it won't get stuck forever.
#
# Rule:
#   If ANY rock is seen during stable window => ROCK side (LEFT)
#   Else if any bean is seen => BEAN side (RIGHT)
#
# Run:
#   sudo /home/pi/venv/bin/python camotor.py

import time
import cv2
import numpy as np
import pickle
from picamera2 import Picamera2
from rpi_ws281x import PixelStrip, Color
import os
import sys

# =========================
# WS2812 / NeoPixel Setup
# =========================
LED_COUNT = 7
LED_PIN = 18
LED_FREQ_HZ = 800000
LED_DMA = 10
LED_INVERT = False
LED_BRIGHTNESS = 255
LED_CHANNEL = 0

strip = PixelStrip(LED_COUNT, LED_PIN, LED_FREQ_HZ, LED_DMA,
                   LED_INVERT, LED_BRIGHTNESS, LED_CHANNEL)

def set_max_white():
    strip.begin()
    white = Color(255, 255, 255)
    for i in range(LED_COUNT):
        strip.setPixelColor(i, white)
    strip.show()

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
    delta = int(target_pos - current_pos)
    if delta == 0:
        return current_pos
    stepper.step(abs(delta), direction=+1 if delta > 0 else -1)
    return target_pos

# =========================
# Camera / Vision Configuration
# =========================
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
SETTLE_SEC     = 0.08

# WAIT_CLEAR timeout (recommended option A)
MAX_WAIT_CLEAR_SEC = 2.0

# =========================
# Helpers
# =========================
def load_model():
    if not os.path.exists(MODEL_FILE):
        print(f"ERROR: {MODEL_FILE} not found!")
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

# =========================
# Main
# =========================
def main():
    model = load_model()
    if model is None:
        sys.exit(1)

    stepper = ULN2003Stepper(FLIP_IN1, FLIP_IN2, FLIP_IN3, FLIP_IN4,
                             step_delay=FLIP_STEP_DELAY)
    flip_pos = 0

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

    state = "WAIT_OBJECT"
    stable_cnt = 0
    last_label = None

    beans_max = 0
    rocks_max = 0

    clear_start_t = 0.0

    print("Inference + Flip Mode (WAIT_CLEAR + TIMEOUT)")
    print("b: Capture Background | q: Quit")

    try:
        while True:
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

                # ===== FSM =====
                if state == "WAIT_OBJECT":
                    stable_cnt = 0
                    last_label = None
                    beans_max = 0
                    rocks_max = 0
                    if mask_area > PRESENT_THRESH:
                        state = "CLASSIFY"

                elif state == "CLASSIFY":
                    if mask_area > PRESENT_THRESH:
                        stable_cnt += 1
                        if beans_count > beans_max:
                            beans_max = beans_count
                        if rocks_count > rocks_max:
                            rocks_max = rocks_count
                    else:
                        stable_cnt = 0
                        beans_max = 0
                        rocks_max = 0
                        state = "WAIT_OBJECT"

                    if stable_cnt >= STABLE_FRAMES:
                        time.sleep(SETTLE_SEC)

                        # Rock wins (also covers "both present")
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

                elif state == "FLIP":
                    if last_label == "BEAN":
                        print("Flip RIGHT (BEAN)")
                        flip_pos = move_to(stepper, POS_RIGHT, flip_pos)
                    else:
                        print("Flip LEFT (ROCK)")
                        flip_pos = move_to(stepper, POS_LEFT, flip_pos)

                    time.sleep(DROP_WAIT_SEC)

                    flip_pos = move_to(stepper, 0, flip_pos)
                    stepper.release()

                    clear_start_t = time.time()
                    state = "WAIT_CLEAR"

                elif state == "WAIT_CLEAR":
                    # Re-arm when cleared OR after timeout (option A)
                    if mask_area < CLEAR_THRESH or (time.time() - clear_start_t) > MAX_WAIT_CLEAR_SEC:
                        state = "WAIT_OBJECT"

                # Debug overlay (your format)
                cv2.putText(full_bgr, f"State:{state} mask={mask_area} stable={stable_cnt} bMax={beans_max} rMax={rocks_max}",
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
                state = "WAIT_OBJECT"
                stable_cnt = 0
                last_label = None
                beans_max = 0
                rocks_max = 0

    finally:
        try:
            picam2.stop()
        except:
            pass
        cv2.destroyAllWindows()
        stepper.cleanup()

if __name__ == "__main__":
    main()
