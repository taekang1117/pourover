# color_test_webcam.py

import cv2
import numpy as np

def get_color_name(hsv_value):
    """Simple mapping from Hue value to rough color category.
    You can expand this mapping for your bean-categories."""
    hue = hsv_value[0]
    if hue < 10 or hue > 170:
        return "RED/BROWN-TONE"
    elif hue < 25:
        return "ORANGE"
    elif hue < 40:
        return "YELLOW"
    elif hue < 80:
        return "GREEN"
    elif hue < 130:
        return "BLUE"
    elif hue < 160:
        return "PURPLE"
    else:
        return "UNKNOWN"

def main():
    cap = cv2.VideoCapture(0)  # 0 = default webcam, change if needed
    if not cap.isOpened():
        print("Cannot open camera")
        return

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Optionally resize for speed
        frame = cv2.resize(frame, (640, 480))

        # Convert from BGR to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Choose pixel in the center of frame (you can refine to ROI of object)
        height, width = hsv.shape[:2]
        cx, cy = width // 2, height // 2
        hsv_center = hsv[cy, cx]

        # Determine approximate color name
        color_name = get_color_name(hsv_center)

        # Display result on frame
        cv2.circle(frame, (cx, cy), 10, (0, 0, 0), 2)
        cv2.putText(frame, f"Color: {color_name}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

        cv2.imshow("Webcam Color Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
