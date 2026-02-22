# main_rf.py - RandomForest Inference for Bean vs Rock
# Run: sudo python3 main_rf.py

import time
import cv2
import numpy as np
import pandas as pd
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

strip = PixelStrip(LED_COUNT, LED_PIN, LED_FREQ_HZ, LED_DMA, LED_INVERT, LED_BRIGHTNESS, LED_CHANNEL)

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

# =========================
# Configuration
# =========================
FRAME_W, FRAME_H = 960, 540
# ROI_X, ROI_Y, ROI_W, ROI_H = 260, 90, 440, 360
ROI_X, ROI_Y, ROI_W, ROI_H = 280, 23, 431, 450

BLUR_K = 5
MORPH_K = 5
OPEN_ITERS = 2
CLOSE_ITERS = 2
# MIN_AREA = 800
# MAX_AREA = 40000
MIN_AREA = 1000
MAX_AREA = 8000

MODEL_FILE = "bean_model.pkl"

# Feature columns used by train_model.py (16-feature model).
FEATURE_COLS_16 = [
    "area",
    "aspect_ratio",
    "circularity",
    "solidity",
    "perimeter",
    "mean_hue",
    "mean_saturation",
    "gabor_mean",
    "gabor_std",
    "hu1",
    "hu2",
    "hu3",
    "hu4",
    "hu5",
    "hu6",
    "hu7",
]

# Backward compatibility for old 5-feature models.
FEATURE_COLS_5 = [
    "area",
    "aspect_ratio",
    "circularity",
    "solidity",
    "perimeter",
]

# Texture kernel reused for per-object texture features.
GABOR_KERNEL = cv2.getGaborKernel((9, 9), 3.0, np.pi / 4, 8.0, 0.5, 0, ktype=cv2.CV_32F)

# =========================
# ULN2003 Stepper Setup (28BYJ-48 5V)
# =========================
FLIP_IN1 = 5
FLIP_IN2 = 6
FLIP_IN3 = 13
FLIP_IN4 = 19

FLIP_STEP_DELAY = 0.0018
STEPS_135_DEG = 768
RETURN_WAIT_SEC = 0.5

# =========================
# Helpers
# =========================
def load_model():
    if not os.path.exists(MODEL_FILE):
        print(f"ERROR: {MODEL_FILE} not found!")
        print("Please run collect_data.py then train_model.py first.")
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
    return (acc / n).astype(np.uint8)

def get_object_mask(roi_bgr, bg_gray):
    g1 = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    g1 = cv2.GaussianBlur(g1, (BLUR_K, BLUR_K), 0)
    # Suppress bright reflections: keep only pixels that became darker than background.
    diff = cv2.subtract(bg_gray, g1)
    otsu_t, _ = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    min_diff_t = 12
    final_t = max(min_diff_t, int(otsu_t))
    _, mask = cv2.threshold(diff, final_t, 255, cv2.THRESH_BINARY)
    mask = morph_cleanup(mask)
    return mask


def is_valid_contour(cnt):
    area = cv2.contourArea(cnt)
    return MIN_AREA <= area <= MAX_AREA

def get_model_feature_cols(model):
    if hasattr(model, "feature_names_in_"):
        cols = [str(c) for c in model.feature_names_in_]
        if len(cols) == 0:
            raise ValueError("Model has empty feature_names_in_.")
        return cols
    n_features = int(getattr(model, "n_features_in_", 0))
    if n_features == len(FEATURE_COLS_16):
        return FEATURE_COLS_16
    if n_features == len(FEATURE_COLS_5):
        return FEATURE_COLS_5
    raise ValueError(
        f"Unsupported model feature count: {n_features}. "
        f"Expected {len(FEATURE_COLS_5)} or {len(FEATURE_COLS_16)}."
    )


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


def run_flip(stepper, current_pos, direction):
    start_pos = current_pos
    target_pos = start_pos + (STEPS_135_DEG if direction > 0 else -STEPS_135_DEG)
    current_pos = move_to(stepper, target_pos, current_pos)
    time.sleep(RETURN_WAIT_SEC)
    current_pos = move_to(stepper, start_pos, current_pos)
    stepper.release()
    return current_pos

# =========================
# Main
# =========================
def main():
    model = load_model()
    if model is None:
        sys.exit(1)
    try:
        model_feature_cols = get_model_feature_cols(model)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
    print(f"Using {len(model_feature_cols)} model features: {model_feature_cols}")

    roi_rect = clamp_roi(ROI_X, ROI_Y, ROI_W, ROI_H, FRAME_W, FRAME_H)
    rx, ry, rw, rh = roi_rect

    picam2 = Picamera2()
    stepper = None
    stepper_pos = 0
    set_max_white()
    
    config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (FRAME_W, FRAME_H)})
    picam2.configure(config)
    picam2.start()

    time.sleep(1.5)
    try:
        picam2.set_controls({"AeEnable": False, "AwbEnable": False})
    except:
        pass

    bg_gray = None
    try:
        try:
            stepper = ULN2003Stepper(
                FLIP_IN1, FLIP_IN2, FLIP_IN3, FLIP_IN4,
                step_delay=FLIP_STEP_DELAY
            )
            print("Flipper stepper initialized.")
        except Exception as exc:
            print(f"Flipper stepper unavailable: {exc}")
            print("Continuing without flipper controls.")

        print("Inference Mode")
        print("b: Capture Background | r/l: Flip 135 deg and return | q: Quit")

        while True:
            frame_rgb = picam2.capture_array()
            full_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            cv2.rectangle(full_bgr, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 2)
            roi_bgr = full_bgr[ry:ry + rh, rx:rx + rw]
            vis_roi = roi_bgr.copy()

            if bg_gray is None:
                cv2.putText(full_bgr, "Press 'b' for BACKGROUND", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                mask = get_object_mask(roi_bgr, bg_gray)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                beans_count = 0
                rocks_count = 0

                feature_list = []
                valid_contours = []
                coords = []

                # 1. Collect features for batch prediction (faster than one by one, though for <50 objs it matters little)
                for cnt in contours:
                    if not is_valid_contour(cnt):
                        continue

                    feats = get_features_dict(cnt, roi_bgr)
                    if feats:
                        row = {name: feats.get(name, 0.0) for name in model_feature_cols}
                        feature_list.append(row)
                        valid_contours.append(cnt)
                        coords.append(cv2.boundingRect(cnt)) # (x,y,w,h)

                # 2. Predict
                if feature_list:
                    feature_df = pd.DataFrame(feature_list, columns=model_feature_cols)
                    preds = model.predict(feature_df)
                    # probs = model.predict_proba(feature_list) # for confidence calc if needed

                    for i, label in enumerate(preds):
                        cnt = valid_contours[i]
                        x, y, w, h = coords[i]
                        
                        if label == 1: # BEAN
                            color = (0, 255, 0)
                            text = "BEAN"
                            beans_count += 1
                        else: # ROCK
                            color = (0, 0, 255)
                            text = "ROCK"
                            rocks_count += 1
                        
                        cv2.rectangle(vis_roi, (x, y), (x+w, y+h), color, 2)
                        cv2.putText(vis_roi, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw stats
                header = f"Beans: {beans_count} | Rocks: {rocks_count}"
                cv2.putText(full_bgr, header, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                if len(contours) > 0 and len(feature_list) == 0:
                     # Objects detected but filtered out by area
                     pass

                cv2.imshow("Mask", mask)

            cv2.imshow("Review", full_bgr)
            cv2.imshow("Inference", vis_roi)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                set_led_off()
                break
            elif key == ord('b'):
                print("Capturing background...")
                bg_gray = capture_background_gray(picam2, roi_rect)
                print("Background captured.")
            elif key == ord('r'):
                if stepper is None:
                    print("Flipper not initialized.")
                    continue
                stepper_pos = run_flip(stepper, stepper_pos, direction=+1)
                print("Flipper rotated right and returned.")
            elif key == ord('l'):
                if stepper is None:
                    print("Flipper not initialized.")
                    continue
                stepper_pos = run_flip(stepper, stepper_pos, direction=-1)
                print("Flipper rotated left and returned.")
    finally:
        picam2.stop()
        cv2.destroyAllWindows()
        if stepper is not None:
            stepper.cleanup()

if __name__ == "__main__":
    main()
