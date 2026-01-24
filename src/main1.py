# main.py (PiCamera2) - Stable Bean vs Rock (NO UNKNOWN)
import time
import cv2
import numpy as np
from picamera2 import Picamera2
from rpi_ws281x import PixelStrip, Color

# =========================
# WS2812 / NeoPixel
# =========================
LED_COUNT = 7
LED_PIN = 18          # GPIO18 = physical pin 12
LED_FREQ_HZ = 800000
LED_DMA = 10
LED_INVERT = False
LED_BRIGHTNESS = 255
LED_CHANNEL = 0

strip = PixelStrip(LED_COUNT, LED_PIN, LED_FREQ_HZ, LED_DMA, LED_INVERT, LED_BRIGHTNESS, LED_CHANNEL)

def set_max_white():
    strip.begin()
    # If your "white" looks greenish, your strip may be GRB. Then use Color(255,255,255) still
    # but you'd need a GRB mapping function. Leave for now unless color is wrong.
    white = Color(255, 255, 255)
    for i in range(LED_COUNT):
        strip.setPixelColor(i, white)
    strip.show()

# =========================
# Tunables
# =========================
FRAME_W, FRAME_H = 960, 540

# Background capture averaging (important)
BG_FRAMES = 20

# Mask robustness
BLUR_K = 5
MORPH_K = 5
OPEN_ITERS = 2
CLOSE_ITERS = 2

# Noise rejection filters
MIN_AREA = 800          # start here; raise to 1200 if still noisy
MAX_AREA = 30000        # your 7000 is too small if bean bbox changes; use safer bound
MIN_W = 18
MIN_H = 18
MIN_EXTENT = 0.30       # area/(w*h)
BORDER_MARGIN = 8       # ignore blobs touching edges

SHOW_DEBUG = True

# ---- Bean shape thresholds (tune) ----
BEAN_AR_MIN = 1.20
BEAN_AR_MAX = 2.40
BEAN_SOL_MIN = 0.85     # lower than 0.88 to reduce "unknown"
BEAN_CIRC_MIN = 0.35
BEAN_CIRC_MAX = 0.92
ELL_AR_MIN = 1.15
ELL_AR_MAX = 2.90


# =========================
# Helpers
# =========================
def morph_cleanup(mask: np.ndarray) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_K, MORPH_K))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=OPEN_ITERS)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=CLOSE_ITERS)
    return mask

def capture_background_gray(picam2: Picamera2, n=BG_FRAMES) -> np.ndarray:
    """Average multiple frames to get a stable background reference."""
    acc = None
    for _ in range(n):
        frame_rgb = picam2.capture_array()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        g = cv2.GaussianBlur(g, (BLUR_K, BLUR_K), 0)
        acc = g if acc is None else acc + g
    return (acc / n).astype(np.uint8)

def get_object_mask(frame_bgr: np.ndarray, bg_gray: np.ndarray):
    """Robust mask: absdiff + Otsu threshold (adapts to lighting)."""
    g1 = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    g1 = cv2.GaussianBlur(g1, (BLUR_K, BLUR_K), 0)

    diff = cv2.absdiff(g1, bg_gray)

    # Dynamic threshold (better than fixed DIFF_THRESH)
    _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    mask = morph_cleanup(mask)
    return mask, diff

def touches_border(x, y, w, h, W, H, margin=BORDER_MARGIN):
    return (x <= margin or y <= margin or (x + w) >= (W - margin) or (y + h) >= (H - margin))

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
            ellipse = cv2.fitEllipse(cnt)
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
    if touches_border(x, y, w, h, FRAME_W, FRAME_H):
        return True
    return False

def is_bean(stats: dict) -> bool:
    """Bean decision. If false => ROCK (no UNKNOWN)."""
    ar = stats["aspect"]
    sol = stats["solidity"]
    circ = stats["circularity"]
    ell_ar = stats["ell_ar"]

    bean_like = (
        (BEAN_AR_MIN <= ar <= BEAN_AR_MAX) and
        (sol >= BEAN_SOL_MIN) and
        (BEAN_CIRC_MIN <= circ <= BEAN_CIRC_MAX)
    )

    # Ellipse ratio helps stabilize on smooth ovals
    if ell_ar is not None:
        bean_like = bean_like and (ELL_AR_MIN <= ell_ar <= ELL_AR_MAX)

    return bean_like


# =========================
# Main
# =========================
def main():
    picam2 = Picamera2()
    set_max_white()

    config = picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (FRAME_W, FRAME_H)}
    )
    picam2.configure(config)
    picam2.start()

    # Let AE/AWB settle under LED light, then lock for stable subtraction
    time.sleep(1.5)
    try:
        picam2.set_controls({"AeEnable": False, "AwbEnable": False})
    except Exception:
        pass

    bg_gray = None
    print("Controls: b=capture background(empty) | r=reset | q=quit")

    while True:
        frame_rgb = picam2.capture_array()
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        vis = frame.copy()

        if bg_gray is None:
            cv2.putText(vis, "Press 'b' to capture BACKGROUND (empty plate)",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("PiCam - Bean vs Rock (Stable)", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('b'):
                print("Capturing background... keep plate empty & steady")
                bg_gray = capture_background_gray(picam2)
                print("Background captured.")
            elif key == ord('q'):
                break
            continue

        obj_mask, diff = get_object_mask(frame, bg_gray)

        contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        beans = 0
        rocks = 0

        for cnt in contours:
            stats = contour_stats(cnt)
            if is_noise(stats):
                continue

            label = "COFFEE BEAN" if is_bean(stats) else "ROCK"
            x, y, w, h = stats["bbox"]

            if label == "COFFEE BEAN":
                color = (0, 255, 0); beans += 1
            else:
                color = (0, 0, 255); rocks += 1

            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)

            if stats["ellipse"] is not None:
                cv2.ellipse(vis, stats["ellipse"], color, 2)

            txt = (f"{label} AR:{stats['aspect']:.2f} "
                   f"C:{stats['circularity']:.2f} "
                   f"Sol:{stats['solidity']:.2f} "
                   f"A:{stats['area']:.0f}")
            cv2.putText(vis, txt, (x, max(20, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        header = f"Beans: {beans} | Rocks: {rocks}"
        cv2.putText(vis, header, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.imshow("PiCam - Bean vs Rock (Stable)", vis)

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
