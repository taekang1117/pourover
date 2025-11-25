import time
import cv2
import numpy as np
from picamera2 import Picamera2

def get_general_color_name(hsv_value):
    h, s, v = hsv_value
    if v < 40 and s < 50:
        return "BLACK / VERY DARK"
    if v > 230 and s < 30:
        return "WHITE / VERY LIGHT"
    if s < 50 and v < 200:
        return "GRAY / NEUTRAL"
    if h < 5 or h >= 170:
        return "RED"
    elif h < 15:
        return "RED-ORANGE"
    elif h < 25:
        return "ORANGE"
    elif h < 35:
        return "YELLOW-ORANGE"
    elif h < 45:
        return "YELLOW"
    elif h < 55:
        return "YELLOW-GREEN"
    elif h < 70:
        return "GREEN"
    elif h < 85:
        return "CYAN-GREEN"
    elif h < 100:
        return "CYAN"
    elif h < 115:
        return "BLUE-CYAN"
    elif h < 130:
        return "BLUE"
    elif h < 145:
        return "BLUE-PURPLE"
    elif h < 160:
        return "PURPLE"
    elif h < 170:
        return "MAGENTA"
    else:
        return "BROWN / RED-BROWN TONE"

def get_bean_category(hsv_mean, other_features=None):
    h, s, v = hsv_mean

    if v > 200 and s < 50:
        return "LIGHT-COLORED"
    if v < 50 and s < 60:
        return "DEFECTIVE / VERY DARK"
    if v > 180 and s < 80 and h < 40:
        return "QUAKER"
    if v > 140 and s < 120 and h < 50:
        return "SEMI-QUAKER"
    if 60 < v < 180 and s > 60 and 15 < h < 60:
        return "NON-QUAKER / GOOD BEAN"
    return "OTHER DEFECT"

def main():
    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"format": "BGR888", "size": (640, 480)}
    )
    picam2.configure(config)
    picam2.start()

    print("Headless mode. Press Ctrl+C to stop.")
    try:
        while True:
            frame = picam2.capture_array()

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            h, w = hsv.shape[:2]
            roi_size = 50
            cx, cy = w // 2, h // 2
            roi = hsv[cy - roi_size // 2 : cy + roi_size // 2,
                      cx - roi_size // 2 : cx + roi_size // 2]

            mean_h = int(np.mean(roi[:, :, 0]))
            mean_s = int(np.mean(roi[:, :, 1]))
            mean_v = int(np.mean(roi[:, :, 2]))
            hsv_mean = (mean_h, mean_s, mean_v)

            general_color = get_general_color_name(hsv_mean)
            bean_category = get_bean_category(hsv_mean)

            # Print one line that updates in place
            print(
                f"H:{mean_h:3d} S:{mean_s:3d} V:{mean_v:3d} | "
                f"{general_color:25s} | {bean_category:25s}",
                end="\r",
                flush=True,
            )

            time.sleep(0.2)  # slow down prints a bit

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        picam2.stop()

if __name__ == "__main__":
    main()
