# main.py (PiCamera2) - Ellipse = Bean | Irregular = Rock
# Run: sudo python3 main.py

import time
import cv2
import numpy as np
from picamera2 import Picamera2
from rpi_ws281x import PixelStrip, Color

# =========================
# WS2812 / NeoPixel
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
# Tunables
# =========================
FRAME_W, FRAME_H = 960, 540
ROI_X, ROI_Y, ROI_W, ROI_H = 260, 90, 440, 360
BG_FRAMES = 20
BLUR_K, MORPH_K = 5, 5
OPEN_ITERS, CLOSE_ITERS = 2, 2

MIN_AREA = 800
MAX_AREA = 60000
MIN_W, MIN_H = 18, 18
MIN_EXTENT = 0.30
SHOW_DEBUG = True

# =========================
# NEW CLASSIFICATION LOGIC
# =========================
# Bean = Smooth/Ellipse (High Solidity)
# Rock = Jagged/Irregular (Low Solidity)

BEAN_SOL_MIN = 0.92      # Solidity > 0.92 means very smooth/convex (Bean)
BEAN_CIRC_MIN = 0.65     # Beans aren't usually extremely thin/jagged
BEAN_AR_MIN = 1.10       # Bounding Box Aspect Ratio
BEAN_AR_MAX = 2.40

# =========================
# Helpers
# =========================
def clamp_roi(x, y, w, h, W, H):
    return max(0, min(x, W-1)), max(0, min(y, H-1)), max(1, min(w, W-x)), max(1, min(h, H-y))

def morph_cleanup(mask):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_K, MORPH_K))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=OPEN_ITERS)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=CLOSE_ITERS)

def capture_background_gray(picam2, roi_rect, n=BG_FRAMES):
    rx, ry, rw, rh = roi_rect
    acc = None
    for _ in range(n):
        frame_rgb = picam2.capture_array()
        roi = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)[ry:ry+rh, rx:rx+rw]
        g = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32), (BLUR_K, BLUR_K), 0)
        acc = g if acc is None else acc + g
    return (acc / n).astype(np.uint8)

def get_object_mask(roi_bgr, bg_gray):
    g1 = cv2.GaussianBlur(cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY), (BLUR_K, BLUR_K), 0)
    diff = cv2.absdiff(g1, bg_gray)
    
    # We use Otsu's to find the base level, then increase it by 10 to ignore shadows
    thresh_val, _ = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, mask = cv2.threshold(diff, thresh_val + 10, 255, cv2.THRESH_BINARY)
    
    return morph_cleanup(mask), diff

def contour_stats(cnt):
    x, y, w, h = cv2.boundingRect(cnt)
    area = float(cv2.contourArea(cnt))
    perim = float(cv2.arcLength(cnt, True))
    circularity = float((4.0 * np.pi * area) / (perim * perim + 1e-9))
    hull = cv2.convexHull(cnt)
    solidity = float(area / (cv2.contourArea(hull) + 1e-9))
    aspect = float(max(w, h) / (min(w, h) + 1e-9))
    
    ellipse = cv2.fitEllipse(cnt) if len(cnt) >= 5 else None
    ell_ar = None
    if ellipse:
        (_, _), (MA, ma), _ = ellipse
        ell_ar = float(max(MA, ma) / (min(MA, ma) + 1e-9))

    return {"bbox": (x, y, w, h), "area": area, "circularity": circularity, 
            "solidity": solidity, "aspect": aspect, "ellipse": ellipse, "ell_ar": ell_ar}

def classify_bean_or_rock(stats):
    """
    Classifies based on smoothness.
    Smooth Ellipse (High Solidity) -> BEAN
    Irregular Shape (Low Solidity) -> ROCK
    """
    sol = stats["solidity"]
    circ = stats["circularity"]
    ar = stats["aspect"]

    # If it's smooth, filled-in, and not a weirdly thin line
    if sol >= BEAN_SOL_MIN and circ >= BEAN_CIRC_MIN and (BEAN_AR_MIN <= ar <= BEAN_AR_MAX):
        return "COFFEE BEAN"
    return "ROCK"

def main():
    rx, ry, rw, rh = clamp_roi(ROI_X, ROI_Y, ROI_W, ROI_H, FRAME_W, FRAME_H)
    picam2 = Picamera2()
    set_max_white()
    picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888", "size": (FRAME_W, FRAME_H)}))
    picam2.start()
    time.sleep(1.5)

    bg_gray = None
    while True:
        full_bgr = cv2.cvtColor(picam2.capture_array(), cv2.COLOR_RGB2BGR)
        cv2.rectangle(full_bgr, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 2)
        roi_bgr = full_bgr[ry:ry + rh, rx:rx + rw]

        if bg_gray is None:
            cv2.putText(full_bgr, "Press 'b' to capture BACKGROUND", (20, 40), 2, 0.8, (0, 255, 255), 2)
            cv2.imshow("Detection", full_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('b'): bg_gray = capture_background_gray(picam2, (rx, ry, rw, rh))
            elif key == ord('q'): break
            continue

        obj_mask, diff = get_object_mask(roi_bgr, bg_gray)
        contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        beans, rocks = 0, 0

        for cnt in contours:
            stats = contour_stats(cnt)
            if stats["area"] < MIN_AREA: continue

            label = classify_bean_or_rock(stats)
            color = (0, 255, 0) if label == "COFFEE BEAN" else (0, 0, 255)
            if label == "COFFEE BEAN": beans += 1
            else: rocks += 1

            x, y, w, h = stats["bbox"]
            cv2.rectangle(full_bgr, (x+rx, y+ry), (x+w+rx, y+h+ry), color, 2)
            if stats["ellipse"]:
                (cx, cy), (MA, ma), ang = stats["ellipse"]
                cv2.ellipse(full_bgr, ((cx+rx, cy+ry), (MA, ma), ang), color, 2)

            txt = f"{label} Sol:{stats['solidity']:.2f}"
            cv2.putText(full_bgr, txt, (x+rx, y+ry-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.putText(full_bgr, f"Beans: {beans} | Rocks: {rocks}", (20, 30), 2, 0.9, (255, 255, 255), 2)
        cv2.imshow("Detection", full_bgr)
        if SHOW_DEBUG: cv2.imshow("Mask", obj_mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('r'): bg_gray = None

    picam2.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
