import cv2
import numpy as np
from picamera2 import Picamera2

def get_general_color_name(hsv_value):
    """General mapping from HSV hue + saturation/value to color names."""
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
    """Return bean category based on HSV mean (and optionally other features)."""
    h, s, v = hsv_mean

    # Specific bean-sorting categories thresholds (example values, tune later)
    if v > 200 and s < 50:
        return "LIGHT-COLORED"
    if v < 50 and s < 60:
        return "DEFECTIVE / VERY DARK"
    # Quaker: very pale beans
    if v > 180 and s < 80 and h < 40:
        return "QUAKER"
    # Semi-Quaker: intermediate pale
    if v > 140 and s < 120 and h < 50:
        return "SEMI-QUAKER"
    # Good bean / normal roasted
    if 60 < v < 180 and s > 60 and 15 < h < 60:
        return "NON-QUAKER / GOOD BEAN"
    # Other defects (crack, wormhole, etc)
    return "OTHER DEFECT"

def main():
    # Use PiCamera2 instead of VideoCapture(0)
    picam2 = Picamera2()

    # 640x480 is enough for your ROI + text overlay
    config = picam2.create_preview_configuration(
        main={"format": "BGR888", "size": (640, 480)}
    )
    picam2.configure(config)
    picam2.start()

    print("Press 'q' to quit.")
    while True:
        # Grab a frame as a NumPy array in BGR format (OpenCV-friendly)
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

        cv2.rectangle(
            frame,
            (cx - roi_size // 2, cy - roi_size // 2),
            (cx + roi_size // 2, cy + roi_size // 2),
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"General Color: {general_color}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Bean Category: {bean_category}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"H:{mean_h} S:{mean_s} V:{mean_v}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )

        cv2.imshow("PiCamera2 Bean-Color Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    picam2.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
