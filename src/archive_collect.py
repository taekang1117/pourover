# collect_data.py
# Run: sudo python3 collect_data.py

import time
import cv2
import numpy as np
import pandas as pd
from picamera2 import Picamera2
from rpi_ws281x import PixelStrip, Color
# RGB LED Strip
import os

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
# === DONT CHANGE ==== 

ROI_X, ROI_Y, ROI_W, ROI_H = 272, 44, 418, 417 # From provided corner points (axis-aligned bounds)

# Image Processing Tunables
BLUR_K = 5
MORPH_K = 5
OPEN_ITERS = 2
CLOSE_ITERS = 2
# MIN_AREA = 800
# MAX_AREA = 40000

MIN_AREA = 1000
# MAX_AREA = 8000
MAX_AREA = 10000

# File to save data
DATA_FILE = "training_data.csv"

# Texture kernel (reused for every contour).
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
    diff = cv2.absdiff(g1, bg_gray)
    _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = morph_cleanup(mask)
    return mask

def get_features(cnt, roi_bgr):
    # Base geometric features.
    
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

    # Create per-object mask inside ROI for color/texture statistics.
    obj_mask = np.zeros(roi_bgr.shape[:2], dtype=np.uint8)
    cv2.drawContours(obj_mask, [cnt], -1, 255, thickness=-1)

    # Color analysis: mean Hue and Saturation (HSV space).
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    mean_h, mean_s, _ = cv2.mean(hsv, mask=obj_mask)[:3]

    # Texture analysis: masked Gabor response statistics.
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gabor_resp = cv2.filter2D(gray, cv2.CV_32F, GABOR_KERNEL)
    gabor_abs = np.abs(gabor_resp)
    gabor_mean, gabor_std = cv2.meanStdDev(gabor_abs, mask=obj_mask)
    gabor_mean = float(gabor_mean[0][0])
    gabor_std = float(gabor_std[0][0])

    # Hu moments: log-scaled for numerical stability and dynamic-range compression.
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

# =========================
# Main
# =========================
def main():
    roi_rect = clamp_roi(ROI_X, ROI_Y, ROI_W, ROI_H, FRAME_W, FRAME_H)
    rx, ry, rw, rh = roi_rect

    picam2 = Picamera2()
    stepper = None
    stepper_pos = 0
    set_max_white()
    
    config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (FRAME_W, FRAME_H)})
    picam2.configure(config)
    picam2.start()

    time.sleep(1.5) # Warmup
    try:
        picam2.set_controls({"AeEnable": False, "AwbEnable": False})
    except:
        pass

    bg_gray = None
    samples_collected = []
    contours = []

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

        print("="*60)
        print("DATA COLLECTION MODE")
        print("1. Clear plate, press 'b' to capture BACKGROUND.")
        print("2. Place BEANS, press '1' to collect BEAN samples.")
        print("3. Place ROCKS, press '2' to collect ROCK samples.")
        print("4. Press 'r' (right) or 'l' (left) to rotate 135 deg and return.")
        print("5. Press 's' to SAVE to CSV.")
        print("6. Press 'q' to QUIT.")
        print("="*60)

        while True:
            frame_rgb = picam2.capture_array()
            full_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # Draw ROI
            cv2.rectangle(full_bgr, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 2)
            roi_bgr = full_bgr[ry:ry + rh, rx:rx + rw]
            vis_roi = roi_bgr.copy()

            if bg_gray is None:
                cv2.putText(full_bgr, "Press 'b' for BACKGROUND", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                mask = get_object_mask(roi_bgr, bg_gray)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                count_visible = 0
                for cnt in contours:
                    if cv2.contourArea(cnt) < MIN_AREA or cv2.contourArea(cnt) > MAX_AREA:
                        continue

                    count_visible += 1
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(vis_roi, (x, y), (x+w, y+h), (0, 255, 0), 2)

                cv2.putText(full_bgr, f"Visible Objects: {count_visible}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(full_bgr, f"Collected Total: {len(samples_collected)}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.imshow("Mask", mask)

            cv2.imshow("Data Collector", full_bgr)
            cv2.imshow("ROI", vis_roi)

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

            elif key == ord('1'): # BEAN
                if bg_gray is None:
                    print("!! Capture background first (b) !!")
                    continue

                added = 0
                for cnt in contours:
                    if cv2.contourArea(cnt) < MIN_AREA:
                        continue
                    feats = get_features(cnt, roi_bgr)
                    if feats:
                        feats['label'] = 1 # BEAN
                        samples_collected.append(feats)
                        added += 1
                print(f"Added {added} BEAN samples.")

            elif key == ord('2'): # ROCK
                if bg_gray is None:
                    print("!! Capture background first (b) !!")
                    continue

                added = 0
                for cnt in contours:
                    if cv2.contourArea(cnt) < MIN_AREA:
                        continue
                    feats = get_features(cnt, roi_bgr)
                    if feats:
                        feats['label'] = 0 # ROCK
                        samples_collected.append(feats)
                        added += 1
                print(f"Added {added} ROCK samples.")

            elif key == ord('s'):
                if len(samples_collected) > 0:
                    new_df = pd.DataFrame(samples_collected)
                    if os.path.exists(DATA_FILE):
                        try:
                            existing_df = pd.read_csv(DATA_FILE)
                            merged_df = pd.concat([existing_df, new_df], ignore_index=True, sort=False)
                            merged_df.to_csv(DATA_FILE, index=False)
                            print(
                                f"Merged {len(new_df)} new samples with {len(existing_df)} existing samples "
                                f"and saved {len(merged_df)} total to {DATA_FILE}"
                            )
                        except Exception as exc:
                            print(f"Could not merge with existing {DATA_FILE}: {exc}")
                            new_df.to_csv(DATA_FILE, index=False)
                            print(f"Saved {len(new_df)} new samples to {DATA_FILE} (overwrite fallback)")
                    else:
                        new_df.to_csv(DATA_FILE, index=False)
                        print(f"Saved {len(new_df)} samples to {DATA_FILE}")
                else:
                    print("No data to save!")
    finally:
        picam2.stop()
        cv2.destroyAllWindows()
        if stepper is not None:
            stepper.cleanup()

if __name__ == "__main__":
    main()
