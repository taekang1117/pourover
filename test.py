import cv2

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

if not cap.isOpened():
    print("Cannot open camera")
    raise SystemExit

for i in range(10):
    ret, frame = cap.read()
    print(f"Frame {i}: ret={ret}, shape={None if not ret else frame.shape}")

cap.release()
