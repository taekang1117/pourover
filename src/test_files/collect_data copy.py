# collect_data.py
# Run: sudo python3 collect_data.py
import sys
import time
import cv2
import numpy as np
import pandas as pd
from picamera2 import Picamera2
from rpi_ws281x import PixelStrip, Color
import os

# Flipper (Stepper)
from flipper import (
    ULN2003Stepper,
    FLIP_IN1, FLIP_IN2, FLIP_IN3, FLIP_IN4,
    STEPS_135_DEG,
    RETURN_WAIT_SEC,
    FLIP_STEP_DELAY
)

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

# These constants are redefined here to ensure local configuration overrides
FLIP_IN1 = 5
FLIP_IN2 = 6
FLIP_IN3 = 13
FLIP_IN4 = 19
FLIP_STEP_DELAY = 0.0018

strip = PixelStrip(
    LED_COUNT, LED_PIN, LED_FREQ_HZ, LED_DMA,
    LED_INVERT, LED_BRIGHTNESS, LED_CHANNEL
)

def set_max_white():
    strip.begin()
    white = Color(255, 255, 255)
    for i in range(LED_COUNT):
        strip.setPixelColor(i, white)
    strip.show()

# =========================
# Configuration
# =========================
FRAME_W, FRAME_H = 960, 540
ROI_X, ROI_Y, ROI_W, ROI_H = 272, 44, 418, 417 

BLUR_K = 5
MORPH_K = 5
OPEN_ITERS = 2
CLOSE_ITERS = 2
MIN_AREA = 800
MAX_AREA = 40000

DATA_FILE = "training_data.csv"

# =========================
# Helpers
# =========================
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

def get_features(cnt):
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

    return {
        "area": area,
        "aspect_ratio": aspect_ratio_invariant,
        "circularity": circularity,
        "solidity": solidity,
        "perimeter": perim
    }

def empty_plate(flipper):
    flipper.step(STEPS_135_DEG, direction=+1)
    time.sleep(RETURN_WAIT_SEC)
    flipper.step(STEPS_135_DEG, direction=-1)
    flipper.release()

# =========================
# Main
# =========================
def main():
    roi_rect = clamp_roi(ROI_X, ROI_Y, ROI_W, ROI_H, FRAME_W, FRAME_H)
    rx, ry, rw, rh = roi_rect

    flipper = ULN2003Stepper(
        FLIP_IN1, FLIP_IN2, FLIP_IN3, FLIP_IN4,
        step_delay=FLIP_STEP_DELAY
    )

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
    samples_collected = []

    print("=" * 60)
    print("DATA COLLECTION MODE")
    print("1. Clear plate, press 'b' to capture BACKGROUND.")
    print("2. Place BEANS, press '1' to collect BEAN samples.")
    print("3. Place ROCKS, press '2' to collect ROCK samples.")
    print("4. Press 'e' to EMPTY (flip/clear plate).")
    print("5. Press 's' to SAVE to CSV.")
    print("6. Press 'q' to QUIT.")
    print("=" * 60)

    try:
        while True:
            frame_rgb = picam2.capture_array()
            full_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            cv2.rectangle(full_bgr, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 2)
            roi_bgr = full_bgr[ry:ry + rh, rx:rx + rw]
            vis_roi = roi_bgr.copy()
            contours = []

            if bg_gray is None:
                cv2.putText(full_bgr, "Press 'b' for BACKGROUND", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                mask = get_object_mask(roi_bgr, bg_gray)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                count_visible = 0
                for cnt in contours:
                    a = cv2.contourArea(cnt)
                    if MIN_AREA < a < MAX_AREA:
                        count_visible += 1
                        x, y, w, h = cv2.boundingRect(cnt)
                        cv2.rectangle(vis_roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

                cv2.putText(full_bgr, f"Visible Objects: {count_visible}", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(full_bgr, f"Collected Total: {len(samples_collected)}", (20, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.imshow("Mask", mask)

            cv2.imshow("Data Collector", full_bgr)
            cv2.imshow("ROI", vis_roi)

            key = cv2.waitKey(1) & 0xFF

            # --- INDEPENDENT KEY CHECKS (Using IF instead of ELIF) ---
            
            if key == ord('q'):
                break

            if key == ord('b'):
                print("Capturing background...")
                bg_gray = capture_background_gray(picam2, roi_rect)
                print("Background captured.")

            if key == ord('e'):
                print("Emptying plate (flip)...")
                empty_plate(flipper)
                print("Done.")

            if key == ord('1'):  # BEAN
                if bg_gray is not None:
                    added = 0
                    for cnt in contours:
                        if MIN_AREA < cv2.contourArea(cnt) < MAX_AREA:
                            feats = get_features(cnt)
                            if feats:
                                feats["label"] = 1
                                samples_collected.append(feats)
                                added += 1
                    print(f"Added {added} BEAN samples.")
                else:
                    print("!! Capture background first (b) !!")

            if key == ord('2'):  # ROCK
                if bg_gray is not None:
                    added = 0
                    for cnt in contours:
                        if MIN_AREA < cv2.contourArea(cnt) < MAX_AREA:
                            feats = get_features(cnt)
                            if feats:
                                feats["label"] = 0
                                samples_collected.append(feats)
                                added += 1
                    print(f"Added {added} ROCK samples.")
                else:
                    print("!! Capture background first (b) !!")

            if key == ord('s'):
                if len(samples_collected) > 0:
                    df = pd.DataFrame(samples_collected)
                    df.to_csv(DATA_FILE, index=False)
                    print(f"Saved {len(df)} samples to {DATA_FILE}")
                else:
                    print("No data to save!")

    finally:
        picam2.stop()
        cv2.destroyAllWindows()
        try:
            flipper.release()
            flipper.cleanup()
        except:
            pass

if __name__ == "__main__":
    main()