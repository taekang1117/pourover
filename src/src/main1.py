# main.py (PiCamera2) - Stable Bean vs Rock + ROI + Full-frame Ellipse + NO UNKNOWN
# Run: sudo python3 main.py
#
# Notes:
# - This version draws ellipses on the FULL window (not only ROI view).
# - This version uses a "bean = less round / more oval" rule because your rocks look rounder/smoother.
# - NO UNKNOWN: if not BEAN => ROCK.
#
# Controls:
#   b = capture background (empty plate, inside ROI)
#   r = reset background
#   q = quit

import time
import cv2
import numpy as np
from picamera2 import Picamera2
from rpi_ws281x import PixelStrip, Color

# =========================
# WS2812 / NeoPixel
# =========================
LED_COUNT = 7
LED_PIN = 18          # GPIO18 = physical pin 12 (PWM0)
LED_FREQ_HZ = 800000
LED_DMA = 10
LED_INVERT = False
LED_BRIGHTNESS = 255
LED_CHANNEL = 0

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
# Tunables
# =========================
FRAME_W, FRAME_H = 960, 540

# ---- ROI (adjust once to tightly cover only where objects appear) ----
ROI_X = 260
ROI_Y = 90
ROI_W = 440
ROI_H = 360

# Background averaging
BG_FRAMES = 20

# Mask robustness
BLUR_K = 5
MORPH_K = 5
OPEN_ITERS = 2
CLOSE_ITERS = 2

# Noise rejection
MIN_AREA = 800
MAX_AREA = 60000
MIN_W = 18
MIN_H = 18
MIN_EXTENT = 0.30

SHOW_DEBUG = True

# =========================
# CLASSIFICATION (IMPORTANT)
# =========================
# Your observation: rocks look MORE round/smooth than beans.
# Therefore:
#   - ROCK tends to have higher circularity (closer to 1.0) and lower aspect ratio (closer to 1.0)
#   - BEAN tends to be more oval => higher aspect ratio and/or lower circularity

# Strong "rock" cue: too circular
ROCK_CIRC_MIN = 0.82     # if circularity >= this, treat as ROCK

# Strong "bean" cue: oval-ish (aspect ratio)
BEAN_AR_MIN = 1.20       # if aspect ratio >= this (and not too circular), treat as BEAN

# Optional extra bean cue: ellipse ratio (if fitEllipse works)
ELL_AR_MIN = 1.15        # ellipse major/minor ratio threshold for bean

# Safety: if segmentation makes objects look too round, loosen/tighten:
# - If beans become rocks: increase ROCK_CIRC_MIN to 0.86 OR lower BEAN_AR_MIN to 1.15
# - If rocks become beans: lower ROCK_CIRC_MIN to 0.78 OR increase BEAN_AR_MIN to 1.25


# =========================
# Helpers
# =========================
def clamp_roi(x, y, w, h, W, H):
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    return x, y, w, h

def morph_cleanup(mask: np.ndarray) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_K, MORPH_K))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=OPEN_ITERS)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=CLOSE_ITERS)
    return mask

def capture_background_gray(picam2: Picamera2, roi_rect, n=BG_FRAMES) -> np.ndarray:
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

def get_object_mask(roi_bgr: np.ndarray, bg_gray: np.ndarray):
    g1 = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    g1 = cv2.GaussianBlur(g1, (BLUR_K, BLUR_K), 0)

    diff = cv2.absdiff(g1, bg_gray)

    # Otsu threshold adapts to lighting drift
    _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = morph_cleanup(mask)
    return mask, diff

def contour_stats(cnt) -> dict:
    x, y, w, h = cv2.boundingRect(cnt)

    area = float(cv2.contourArea(cnt))
    perim = float(cv2.arcLength(cnt, True))
    circularity = float((4.0 * np.pi * area) / (perim * perim + 1e-9))

    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull)) + 1e-9
    solidity = float(area / hull_area)

    aspect = float(max(w, h) / (min(w, h) + 1e-9))
    extent = float(area / (w * h + 1e-9))

    ellipse = None
    ell_ar = None
    if len(cnt) >= 5:
        try:
            ellipse = cv2.fitEllipse(cnt)  # ((cx,cy),(MA,ma),angle)
            (_, _), (MA, ma), _ = ellipse
            major = max(MA, ma)
            minor = min(MA, ma) + 1e-9
            ell_ar = float(major / minor)
        except cv2.error:
            ellipse = None
            ell_ar = None

    return {
        "bbox": (x, y, w, h),
        "area": area,
        "circularity": circularity,
        "solidity": solidity,
        "aspect": aspect,
        "extent": extent,
        "ellipse": ellipse,
        "ell_ar": ell_ar,
    }

def is_noise(stats: dict) -> bool:
    x, y, w, h = stats["bbox"]
    if stats["area"] < MIN_AREA or stats["area"] > MAX_AREA:
        return True
    if w < MIN_W or h < MIN_H:
        return True
    if stats["extent"] < MIN_EXTENT:
        return True
    return False

def classify_bean_or_rock(stats: dict) -> str:
    """
    NO UNKNOWN:
      If it is very circular -> ROCK
      Else if it is oval-ish -> BEAN
      Else -> ROCK (default)
    """
    ar = stats["aspect"]
    circ = stats["circularity"]
    ell_ar = stats["ell_ar"]

    # Strong rock cue: round
    if circ >= ROCK_CIRC_MIN:
        return "ROCK"

    # Bean cue: oval by bbox AR, and optionally by ellipse AR
    if ar >= BEAN_AR_MIN:
        if ell_ar is None:
            return "COFFEE BEAN"
        return "COFFEE BEAN" if ell_ar >= ELL_AR_MIN else "ROCK"

    # Default
    return "ROCK"


# =========================
# Main
# =========================
def main():
    roi_rect = clamp_roi(ROI_X, ROI_Y, ROI_W, ROI_H, FRAME_W, FRAME_H)
    rx, ry, rw, rh = roi_rect

    picam2 = Picamera2()
    set_max_white()

    config = picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (FRAME_W, FRAME_H)}
    )
    picam2.configure(config)
    picam2.start()

    # settle AE/AWB under LED, then lock to stabilize subtraction
    time.sleep(1.5)
    try:
        picam2.set_controls({"AeEnable": False, "AwbEnable": False})
    except Exception:
        pass

    bg_gray = None
    print("Controls: b=capture background(empty plate) | r=reset | q=quit")
    print(f"ROI: x={rx}, y={ry}, w={rw}, h={rh}")

    while True:
        frame_rgb = picam2.capture_array()
        full_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Draw ROI box on full frame
        cv2.rectangle(full_bgr, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 2)

        roi_bgr = full_bgr[ry:ry + rh, rx:rx + rw]
        vis_roi = roi_bgr.copy()

        if bg_gray is None:
            cv2.putText(full_bgr, "Press 'b' to capture BACKGROUND (empty plate)",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("PiCam - Bean vs Rock (ROI)", full_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('b'):
                print("Capturing ROI background... keep plate empty & steady")
                bg_gray = capture_background_gray(picam2, roi_rect)
                print("Background captured.")
            elif key == ord('q'):
                break
            continue

        obj_mask, diff = get_object_mask(roi_bgr, bg_gray)
        contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        beans = 0
        rocks = 0

        for cnt in contours:
            stats = contour_stats(cnt)
            if is_noise(stats):
                continue

            label = classify_bean_or_rock(stats)
            x, y, w, h = stats["bbox"]

            if label == "COFFEE BEAN":
                color = (0, 255, 0); beans += 1
            else:
                color = (0, 0, 255); rocks += 1

            # Draw on ROI view
            cv2.rectangle(vis_roi, (x, y), (x + w, y + h), color, 2)
            if stats["ellipse"] is not None:
                cv2.ellipse(vis_roi, stats["ellipse"], color, 2)

            # Draw on FULL frame with ROI offset
            cv2.rectangle(full_bgr, (x + rx, y + ry), (x + w + rx, y + h + ry), color, 2)
            if stats["ellipse"] is not None:
                (cx, cy), (MA, ma), ang = stats["ellipse"]
                ellipse_full = ((cx + rx, cy + ry), (MA, ma), ang)
                cv2.ellipse(full_bgr, ellipse_full, color, 2)

            txt = (f"{label} AR:{stats['aspect']:.2f} "
                   f"C:{stats['circularity']:.2f} "
                   f"Sol:{stats['solidity']:.2f} "
                   f"A:{stats['area']:.0f}")
            cv2.putText(full_bgr, txt, (x + rx, max(20, y + ry - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

        header = f"Beans: {beans} | Rocks: {rocks}"
        cv2.putText(full_bgr, header, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.imshow("PiCam - Bean vs Rock (ROI)", full_bgr)
        cv2.imshow("ROI View", vis_roi)

        if SHOW_DEBUG:
            cv2.imshow("Object Mask", obj_mask)
            cv2.imshow("Diff", diff)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            bg_gray = None
            print("Background reset.")

    picam2.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
