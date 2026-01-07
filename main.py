# main.py

import cv2
import numpy as np
from picamera2 import Picamera2

# =========================
# Tunables
# =========================
FRAME_W, FRAME_H = 960, 540

MIN_AREA = 300
MAX_AREA = 200000

DIFF_THRESH = 25
BLUR_K = 5

MORPH_K = 5
OPEN_ITERS = 1
CLOSE_ITERS = 2

# Classification heuristics
ROCK_S_MAX = 45
ROCK_TEX_MIN = 120.0

BEAN_H_MIN = 5
BEAN_H_MAX = 40
BEAN_S_MIN = 40
BEAN_TEX_MAX = 350.0

SHOW_DEBUG = True


# =========================
# Helpers
# =========================
def morph_cleanup(mask: np.ndarray) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_K, MORPH_K))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=OPEN_ITERS)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=CLOSE_ITERS)
    return mask

def laplacian_texture(gray_roi: np.ndarray, mask_roi: np.ndarray) -> float:
    if gray_roi.size == 0:
        return 0.0
    gray_masked = cv2.bitwise_and(gray_roi, gray_roi, mask=mask_roi)
    ys, xs = np.where(mask_roi > 0)
    if len(xs) == 0 or len(ys) == 0:
        return 0.0
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    crop = gray_masked[y0:y1+1, x0:x1+1]
    if crop.size == 0:
        return 0.0
    lap = cv2.Laplacian(crop, cv2.CV_64F)
    return float(lap.var())

def contour_stats(frame_bgr: np.ndarray, cnt) -> dict:
    x, y, w, h = cv2.boundingRect(cnt)
    roi = frame_bgr[y:y+h, x:x+w]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    mask = np.zeros((h, w), dtype=np.uint8)
    cnt_local = cnt.copy()
    cnt_local[:, 0, 0] -= x
    cnt_local[:, 0, 1] -= y
    cv2.drawContours(mask, [cnt_local], -1, 255, -1)

    mean_h = cv2.mean(hsv[:, :, 0], mask=mask)[0]
    mean_s = cv2.mean(hsv[:, :, 1], mask=mask)[0]
    mean_v = cv2.mean(hsv[:, :, 2], mask=mask)[0]

    area = float(cv2.contourArea(cnt))
    perim = float(cv2.arcLength(cnt, True))
    circularity = float((4.0 * np.pi * area) / (perim * perim + 1e-9))

    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull)) + 1e-9
    solidity = float(area / hull_area)

    tex = laplacian_texture(gray, mask)

    return {
        "bbox": (x, y, w, h),
        "mean_h": float(mean_h),
        "mean_s": float(mean_s),
        "mean_v": float(mean_v),
        "area": area,
        "circularity": circularity,
        "solidity": solidity,
        "texture": float(tex),
    }

def classify(stats: dict) -> str:
    h = stats["mean_h"]
    s = stats["mean_s"]
    v = stats["mean_v"]
    t = stats["texture"]
    sol = stats["solidity"]

    if v < 20:
        return "UNKNOWN"

    rock_score = 0
    bean_score = 0

    if s <= ROCK_S_MAX:
        rock_score += 2
    if t >= ROCK_TEX_MIN:
        rock_score += 2
    if sol < 0.82:
        rock_score += 1

    if BEAN_H_MIN <= h <= BEAN_H_MAX and s >= BEAN_S_MIN:
        bean_score += 3
    if t <= BEAN_TEX_MAX:
        bean_score += 1
    if 0.82 <= sol <= 0.99:
        bean_score += 1

    if rock_score > bean_score:
        return "ROCK"
    if bean_score > rock_score:
        return "COFFEE BEAN"
    return "UNKNOWN"

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


# =========================
# Main (PiCamera2)
# =========================
def main():
    # --- Init PiCamera2 ---
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (FRAME_W, FRAME_H)}
    )
    picam2.configure(config)

    # 可選：鎖曝光/白平衡（讓顏色更穩）
    # picam2.set_controls({"AeEnable": True, "AwbEnable": True})
    # 如果要完全鎖定，通常需要先讓它自動跑一下再鎖住，後面可再加。

    picam2.start()

    bg = None
    print("Controls: b=capture background(empty) | r=reset | q=quit")

    while True:
        # Picamera2 gives RGB; OpenCV uses BGR for display/processing
        frame_rgb = picam2.capture_array()
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        vis = frame.copy()

        if bg is None:
            cv2.putText(vis, "Press 'b' to capture BACKGROUND (empty plate)",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("PiCam - Bean vs Rock", vis)

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
            if label == "COFFEE BEAN":
                color = (0, 255, 0); beans += 1
            elif label == "ROCK":
                color = (0, 0, 255); rocks += 1
            else:
                color = (255, 255, 0); unknown += 1

            cv2.rectangle(vis, (x, y), (x+w, y+h), color, 2)
            txt = (f"{label} H:{stats['mean_h']:.0f} S:{stats['mean_s']:.0f} "
                   f"V:{stats['mean_v']:.0f} T:{stats['texture']:.0f} Sol:{stats['solidity']:.2f}")
            cv2.putText(vis, txt, (x, max(20, y-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

        header = f"Beans: {beans} | Rocks: {rocks} | Unknown: {unknown}"
        cv2.putText(vis, header, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.imshow("PiCam - Bean vs Rock", vis)

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
