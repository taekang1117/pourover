# main.py (PiCamera2) - Shape-based Bean vs Rock
import cv2
import numpy as np

from rpi_ws281x import PixelStrip, Color

LED_COUNT = 7
LED_PIN = 18          # GPIO18 = physical pin 12
LED_FREQ_HZ = 800000
LED_DMA = 10
LED_INVERT = False
LED_BRIGHTNESS = 255  # max overall brightness
LED_CHANNEL = 0

strip = PixelStrip(LED_COUNT, LED_PIN, LED_FREQ_HZ, LED_DMA, LED_INVERT, LED_BRIGHTNESS, LED_CHANNEL)

def set_max_white():
    strip.begin()
    white = Color(255, 255, 255)
    for i in range(LED_COUNT):
        strip.setPixelColor(i, white)
    strip.show()

from picamera2 import Picamera2

# =========================
# Tunables
# =========================
FRAME_W, FRAME_H = 960, 540

MIN_AREA = 300
MAX_AREA = 250000

DIFF_THRESH = 25
BLUR_K = 5

MORPH_K = 5
OPEN_ITERS = 1
CLOSE_ITERS = 2

# Erosion/Dilation to separate close beans and remove noise
ERODE_ITERS = 2
DILATE_ITERS = 2

SHOW_DEBUG = True

# ---- Shape thresholds (tune with your real beans/rocks) ----
# Coffee bean tends to be oval:
BEAN_AR_MIN = 1.25        # aspect ratio lower bound (max(w,h)/min(w,h))
BEAN_AR_MAX = 2.10        # upper bound
BEAN_SOL_MIN = 0.88       # beans usually quite solid (smooth-ish boundary)
BEAN_CIRC_MIN = 0.45      # circularity range
BEAN_CIRC_MAX = 0.85

# Rock tends to be irregular:
ROCK_SOL_MAX = 0.84       # below this -> likely rock
ROCK_CIRC_MAX = 0.55      # very jagged often lower circularity (not always)
ROCK_AR_EXTREME = 2.30    # very elongated or extreme aspect -> likely not bean

# Optional: if your rocks are much larger/smaller than beans, use area
# BEAN_AREA_MIN = 600
# BEAN_AREA_MAX = 20000


# =========================
# Helpers
# =========================
def morph_cleanup(mask: np.ndarray) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_K, MORPH_K))
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=OPEN_ITERS)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=CLOSE_ITERS)
    
    # 1. Erode to peel away noise and potential bridges between beans
    mask = cv2.erode(mask, k, iterations=ERODE_ITERS)
    # 2. Dilate to restore size
    mask = cv2.dilate(mask, k, iterations=DILATE_ITERS)
    
    return mask

def get_object_mask(frame_bgr: np.ndarray, bg_bgr: np.ndarray):
    g1 = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    g0 = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2GRAY)

    g1 = cv2.GaussianBlur(g1, (BLUR_K, BLUR_K), 0)
    g0 = cv2.GaussianBlur(g0, (BLUR_K, BLUR_K), 0)

    diff = cv2.absdiff(g1, g0)
    _, mask = cv2.threshold(diff, DIFF_THRESH, 255, cv2.THRESH_BINARY)
    mask = morph_cleanup(mask)
    return mask, diff

def find_contours(mask: np.ndarray):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        a = cv2.contourArea(c)
        if MIN_AREA <= a <= MAX_AREA:
            out.append(c)
    return out

def contour_stats(frame_bgr: np.ndarray, cnt) -> dict:
    x, y, w, h = cv2.boundingRect(cnt)

    area = float(cv2.contourArea(cnt))
    perim = float(cv2.arcLength(cnt, True))
    circularity = float((4.0 * np.pi * area) / (perim * perim + 1e-9))

    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull)) + 1e-9
    solidity = float(area / hull_area)

    # Aspect ratio (>=1)
    aspect = float(max(w, h) / (min(w, h) + 1e-9))

    # Ellipse fit (needs at least 5 points)
    ellipse = None
    if len(cnt) >= 5:
        try:
            ellipse = cv2.fitEllipse(cnt)  # ((cx,cy),(MA,ma), angle)
        except cv2.error:
            ellipse = None

    return {
        "bbox": (x, y, w, h),
        "area": area,
        "perim": perim,
        "circularity": circularity,
        "solidity": solidity,
        "aspect": aspect,
        "ellipse": ellipse,
    }

def classify(stats: dict) -> str:
    """
    Shape-first classifier.
    Works when beans/rocks have similar color under current lighting.
    """
    area = stats["area"]
    circ = stats["circularity"]
    sol = stats["solidity"]
    ar = stats["aspect"]
    ellipse = stats["ellipse"]

    # Basic sanity
    if area < MIN_AREA:
        return "UNKNOWN"

    # Strong rock cues (irregular)
    if sol <= ROCK_SOL_MAX:
        return "ROCK"
    if ar >= ROCK_AR_EXTREME and sol < 0.90:
        return "ROCK"

    # Strong bean cues (oval + smooth)
    # If ellipse exists, we can use its major/minor ratio too
    if ellipse is not None:
        (_, _), (MA, ma), _ = ellipse
        # MA = major axis length, ma = minor axis length (sometimes swapped depending on OpenCV)
        major = max(MA, ma)
        minor = min(MA, ma) + 1e-9
        ell_ar = float(major / minor)
    else:
        ell_ar = None

    bean_like = (
        (BEAN_AR_MIN <= ar <= BEAN_AR_MAX) and
        (sol >= BEAN_SOL_MIN) and
        (BEAN_CIRC_MIN <= circ <= BEAN_CIRC_MAX)
    )

    if ell_ar is not None:
        # Ellipse ratio is usually a good bean indicator
        # Typical beans: ~1.3-2.2; rocks can be all over but often worse fit
        bean_like = bean_like and (1.2 <= ell_ar <= 2.5)

    if bean_like:
        return "COFFEE BEAN"

    # If it doesn't clearly match bean but also not strongly rock, mark unknown
    # You can later push unknown into rock/bean depending on your sorting policy.
    return "UNKNOWN"


# =========================
# Main (PiCamera2)
# =========================
def main():
    picam2 = Picamera2()
    set_max_white()
    config = picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (FRAME_W, FRAME_H)}
    )
    picam2.configure(config)
    picam2.start()

    bg = None
    print("Controls: b=capture background(empty) | r=reset | q=quit")

    while True:
        frame_rgb = picam2.capture_array()
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        vis = frame.copy()

        if bg is None:
            cv2.putText(vis, "Press 'b' to capture BACKGROUND (empty plate)",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("PiCam - Bean vs Rock (Shape)", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('b'):
                bg = frame.copy()
                print("Background captured.")
            elif key == ord('q'):
                break
            continue

        obj_mask, diff = get_object_mask(frame, bg)
        contours = find_contours(obj_mask)

        beans = rocks = unknown = 0

        for cnt in contours:
            stats = contour_stats(frame, cnt)
            label = classify(stats)

            x, y, w, h = stats["bbox"]

            # Draw contour and bbox
            if label == "COFFEE BEAN":
                color = (0, 255, 0); beans += 1
            elif label == "ROCK":
                color = (0, 0, 255); rocks += 1
            else:
                color = (255, 255, 0); unknown += 1

            cv2.rectangle(vis, (x, y), (x+w, y+h), color, 2)

            # Optional: draw fitted ellipse
            if stats["ellipse"] is not None:
                cv2.ellipse(vis, stats["ellipse"], color, 2)

            txt = (f"{label} "
                   f"AR:{stats['aspect']:.2f} "
                   f"C:{stats['circularity']:.2f} "
                   f"Sol:{stats['solidity']:.2f} "
                   f"A:{stats['area']:.0f}")
            cv2.putText(vis, txt, (x, max(20, y-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        header = f"Beans: {beans} | Rocks: {rocks} | Unknown: {unknown}"
        cv2.putText(vis, header, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.imshow("PiCam - Bean vs Rock (Shape)", vis)

        if SHOW_DEBUG:
            cv2.imshow("Object Mask", obj_mask)
            cv2.imshow("Diff", diff)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            bg = None
            print("Background reset.")

    picam2.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
