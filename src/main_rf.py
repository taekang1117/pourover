# main_rf.py - RandomForest Inference for Bean vs Rock
# Run: sudo python3 main_rf.py

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
ROI_X, ROI_Y, ROI_W, ROI_H = 260, 90, 440, 360

BLUR_K = 5
MORPH_K = 5
OPEN_ITERS = 2
CLOSE_ITERS = 2
MIN_AREA = 800
MAX_AREA = 40000

MODEL_FILE = "bean_model.pkl"

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
    diff = cv2.absdiff(g1, bg_gray)
    _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = morph_cleanup(mask)
    return mask

def get_features_vector(cnt):
    # MUST MATCH train_model.py order:
    # ['area', 'aspect_ratio', 'circularity', 'solidity', 'perimeter']
    
    area = float(cv2.contourArea(cnt))
    perim = float(cv2.arcLength(cnt, True))
    
    if perim == 0: return None

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

    roi_rect = clamp_roi(ROI_X, ROI_Y, ROI_W, ROI_H, FRAME_W, FRAME_H)
    rx, ry, rw, rh = roi_rect

    picam2 = Picamera2()
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
    
    print("Inference Mode")
    print("b: Capture Background | q: Quit")

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
                if cv2.contourArea(cnt) < MIN_AREA or cv2.contourArea(cnt) > MAX_AREA:
                    continue
                
                vec = get_features_vector(cnt)
                if vec:
                    feature_list.append(vec)
                    valid_contours.append(cnt)
                    coords.append(cv2.boundingRect(cnt)) # (x,y,w,h)

            # 2. Predict
            if feature_list:
                preds = model.predict(feature_list)
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
            break
        elif key == ord('b'):
            print("Capturing background...")
            bg_gray = capture_background_gray(picam2, roi_rect)
            print("Background captured.")

    picam2.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
