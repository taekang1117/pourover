# collect_data.py
# Run: sudo python3 collect_data.py

import time
import cv2
import numpy as np
import pandas as pd
from picamera2 import Picamera2
from rpi_ws281x import PixelStrip, Color
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

# =========================
# Configuration
# =========================
FRAME_W, FRAME_H = 960, 540
ROI_X, ROI_Y, ROI_W, ROI_H = 260, 90, 440, 360  # Default ROI

# Image Processing Tunables
BLUR_K = 5
MORPH_K = 5
OPEN_ITERS = 2
CLOSE_ITERS = 2
MIN_AREA = 800
MAX_AREA = 40000

# File to save data
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
    # Calculate features requested: [area, aspect_ratio, circularity, solidity, perimeter]
    area = float(cv2.contourArea(cnt))
    perim = float(cv2.arcLength(cnt, True))
    if perim == 0:
        return None

    circularity = (4.0 * np.pi * area) / (perim * perim)

    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull))
    solidity = area / hull_area if hull_area > 0 else 0

    x, y, w, h = cv2.boundingRect(cnt)

    # Rotation-robust-ish aspect ratio: max/min
    aspect_ratio_invariant = float(max(w, h)) / (min(w, h) + 1e-9)

    return {
        "area": area,
        "aspect_ratio": aspect_ratio_invariant,
        "circularity": circularity,
        "solidity": solidity,
        "perimeter": perim
    }

def save_samples_merged(samples, csv_path):
    """
    Append new samples to csv_path if it exists, otherwise create it.
    Also does optional near-duplicate removal by rounding float columns.
    Returns: (prev_rows, new_rows, total_rows_after_save)
    """
    if not samples:
        return (0, 0, 0)

    new_df = pd.DataFrame(samples)

    prev_rows = 0
    existing_df = None

    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        try:
            existing_df = pd.read_csv(csv_path)
            prev_rows = len(existing_df)
        except Exception as e:
            print(f"!! Warning: could not read existing '{csv_path}' ({e}). Overwriting it.")

    if existing_df is not None:
        merged = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        merged = new_df

    # ---- Optional de-duplication ----
    float_cols = [c for c in ["area", "aspect_ratio", "circularity", "solidity", "perimeter"] if c in merged.columns]
    key_cols = float_cols + (["label"] if "label" in merged.columns else [])

    if key_cols:
        tmp = merged.copy()
        for c in float_cols:
            tmp[c] = tmp[c].round(4)  # tweak decimals if you want stricter/looser dedupe
        keep_idx = tmp.drop_duplicates(subset=key_cols).index
        merged = merged.loc[keep_idx].reset_index(drop=True)

    # Consistent column order
    desired = ["area", "aspect_ratio", "circularity", "solidity", "perimeter", "label"]
    cols = [c for c in desired if c in merged.columns] + [c for c in merged.columns if c not in desired]
    merged = merged[cols]

    merged.to_csv(csv_path, index=False)
    return (prev_rows, len(new_df), len(merged))

# =========================
# Main
# =========================
def main():
    roi_rect = clamp_roi(ROI_X, ROI_Y, ROI_W, ROI_H, FRAME_W, FRAME_H)
    rx, ry, rw, rh = roi_rect

    picam2 = Picamera2()
    set_max_white()

    config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (FRAME_W, FRAME_H)})
    picam2.configure(config)
    picam2.start()

    time.sleep(1.5)  # Warmup
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
    print("4. Press 's' to SAVE to CSV (MERGE if exists).")
    print("5. Press 'q' to QUIT.")
    print("=" * 60)

    # IMPORTANT: ensure contours exists even before background is captured
    contours = []

    while True:
        frame_rgb = picam2.capture_array()
        full_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Draw ROI
        cv2.rectangle(full_bgr, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 2)
        roi_bgr = full_bgr[ry:ry + rh, rx:rx + rw]
        vis_roi = roi_bgr.copy()

        if bg_gray is None:
            cv2.putText(full_bgr, "Press 'b' for BACKGROUND", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            mask = get_object_mask(roi_bgr, bg_gray)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            count_visible = 0
            for cnt in contours:
                a = cv2.contourArea(cnt)
                if a < MIN_AREA or a > MAX_AREA:
                    continue

                count_visible += 1
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(vis_roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.putText(full_bgr, f"Visible Objects: {count_visible}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(full_bgr, f"Collected Batch: {len(samples_collected)}", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.imshow("Mask", mask)

        cv2.imshow("Data Collector", full_bgr)
        cv2.imshow("ROI", vis_roi)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('b'):
            print("Capturing background...")
            bg_gray = capture_background_gray(picam2, roi_rect)
            print("Background captured.")

        elif key == ord('1'):  # BEAN
            if bg_gray is None:
                print("!! Capture background first (b) !!")
                continue

            added = 0
            for cnt in contours:
                a = cv2.contourArea(cnt)
                if a < MIN_AREA or a > MAX_AREA:
                    continue
                feats = get_features(cnt)
                if feats:
                    feats["label"] = 1  # BEAN
                    samples_collected.append(feats)
                    added += 1
            print(f"Added {added} BEAN samples.")

        elif key == ord('2'):  # ROCK
            if bg_gray is None:
                print("!! Capture background first (b) !!")
                continue

            added = 0
            for cnt in contours:
                a = cv2.contourArea(cnt)
                if a < MIN_AREA or a > MAX_AREA:
                    continue
                feats = get_features(cnt)
                if feats:
                    feats["label"] = 0  # ROCK
                    samples_collected.append(feats)
                    added += 1
            print(f"Added {added} ROCK samples.")

        elif key == ord('s'):
            if len(samples_collected) > 0:
                prev_n, added_n, total_n = save_samples_merged(samples_collected, DATA_FILE)
                print(f"Saved {added_n} NEW samples to {DATA_FILE}. Total now {total_n} (was {prev_n}).")
                samples_collected.clear()  # prevents re-saving same batch again
            else:
                print("No data to save!")

    picam2.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
