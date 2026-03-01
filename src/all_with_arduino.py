import time
import cv2
import numpy as np
import pandas as pd
import pickle
from picamera2 import Picamera2
from rpi_ws281x import PixelStrip, Color
import os
import sys
import re  # added: for parsing 'Weight: xx g' lines

try:
    import serial
except ImportError:
    serial = None

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

def set_led_off():
    off = Color(0, 0, 0)
    for i in range(LED_COUNT):
        strip.setPixelColor(i, off)
    strip.show()

# =========================
# Configuration (DO NOT TOUCH per your request)
# =========================
FRAME_W, FRAME_H = 960, 540
ROI_X, ROI_Y, ROI_W, ROI_H = 280, 23, 431, 450

BLUR_K = 5
MORPH_K = 5
OPEN_ITERS = 2
CLOSE_ITERS = 2
MIN_AREA = 1000
MAX_AREA = 10000

MODEL_FILE = "bean_model.pkl"

# =========================
# 12V TMC2209 Slow Feeders (EN hard-grounded)
# =========================
M1_STEP = 17
M1_DIR  = 27

STEP_PULSE_US  = 20
STEP_GAP_US    = 1000
DIR_SETTLE_SEC = 0.05
DOSE_STEPS = 300

M1_DIR_NORMAL = True

# =========================
# Auto Feeder Control (slow, safe)
# =========================
AUTO_FEED_ENABLED = True
FEED_COOLDOWN_SEC = 1
FEED_TIMEOUT_SEC = 3.0
POST_FEED_SETTLE_SEC = 0.45
EMPTY_MASK_THRESH = 500
EMPTY_FRAMES = 10

# =========================
# Auto Flip Control
# =========================
AUTO_FLIP_ENABLED = True
FLIP_STABLE_FRAMES = 1
FLIP_COOLDOWN_SEC = 1.0
CLEAR_BEFORE_FLIP_THRESH = 600
ROCK_WINS_IF_BOTH = True

FLIP_DIR_FOR_BEAN = -1
FLIP_DIR_FOR_ROCK = +1

# =========================
# CV Gate / Anti-noise Sequencing
# =========================
DETECT_ONLY_AFTER_FEED = True
CV_GATE_AFTER_FEED_SEC = 0.35
DETECT_WINDOW_AFTER_FEED_SEC = 2.8
PRESENT_STABLE_FRAMES = 4

WAIT_CLEAR_AFTER_FLIP = True
CLEAR_AFTER_FLIP_THRESH = 450
CLEAR_STABLE_FRAMES = 6

# =========================
# NEW: WAIT_CLEAR timeout + nudge (prevents “stuck forever”)
# =========================
CLEAR_TIMEOUT_SEC = 2.0          # if not cleared after this, do a nudge / unlock
CLEAR_RETRY_MAX = 2              # number of nudges before unlocking
CLEAR_NUDGE_STEPS = 120          # small shake steps (not full 135 deg)
CLEAR_NUDGE_WAIT_SEC = 0.15      # pause between nudge out/back

# =========================
# Arduino UART (weight sensor + feeder signal)
# =========================
ARDUINO_SERIAL_PORT = "/dev/ttyACM0"   # 或 /dev/ttyUSB0，USB 连接时
ARDUINO_BAUD = 115200
ARDUINO_WEIGHT_TARGET_G = 20.0         # 与 LCWS_and_RA.cpp 中 WEIGHT_STOP_G 一致
# True：all.py 启动后再发 'g' 给 Arduino，Arduino 收到后才响应称重/FEED；False：一上电就响应
ARDUINO_START_AFTER_PI_READY = True

# =========================
# Feature columns used by train_model.py (16-feature model).
# =========================
FEATURE_COLS_16 = [
    "area","aspect_ratio","circularity","solidity","perimeter",
    "mean_hue","mean_saturation","gabor_mean","gabor_std",
    "hu1","hu2","hu3","hu4","hu5","hu6","hu7",
]
FEATURE_COLS_5 = ["area","aspect_ratio","circularity","solidity","perimeter"]

GABOR_KERNEL = cv2.getGaborKernel((9, 9), 3.0, np.pi / 4, 8.0, 0.5, 0, ktype=cv2.CV_32F)

# =========================
# ULN2003 Stepper Setup (28BYJ-48 5V)
# =========================
FLIP_IN1 = 5
FLIP_IN2 = 6
FLIP_IN3 = 13
FLIP_IN4 = 19

FLIP_STEP_DELAY = 0.0018
STEPS_135_DEG = 1000
# Try 1000. Originally it was 768
RETURN_WAIT_SEC = 0.5

# =========================
# 12V Feeder Driver Class
# =========================
class StepperTMC2209_EN_GND:
    """STEP/DIR stepper where EN is wired to GND (always enabled)."""
    def __init__(self, step_pin, dir_pin, dir_normal=True, name="M"):
        self.step_pin = step_pin
        self.dir_pin = dir_pin
        self.dir_normal = dir_normal
        self.name = name

    def set_dir(self, forward: bool):
        import RPi.GPIO as GPIO
        level = GPIO.HIGH if (forward == self.dir_normal) else GPIO.LOW
        GPIO.output(self.dir_pin, level)
        time.sleep(DIR_SETTLE_SEC)

    def step_n(self, steps: int, pulse_us=STEP_PULSE_US, gap_us=STEP_GAP_US):
        import RPi.GPIO as GPIO
        pulse_s = pulse_us / 1_000_000.0
        gap_s = gap_us / 1_000_000.0
        for _ in range(int(steps)):
            GPIO.output(self.step_pin, GPIO.HIGH)
            time.sleep(pulse_s)
            GPIO.output(self.step_pin, GPIO.LOW)
            time.sleep(gap_s)

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
    diff = cv2.subtract(bg_gray, g1)
    otsu_t, _ = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    min_diff_t = 12
    final_t = max(min_diff_t, int(otsu_t))
    _, mask = cv2.threshold(diff, final_t, 255, cv2.THRESH_BINARY)
    mask = morph_cleanup(mask)
    return mask

def is_valid_contour(cnt):
    area = cv2.contourArea(cnt)
    return MIN_AREA <= area <= MAX_AREA

def get_model_feature_cols(model):
    if hasattr(model, "feature_names_in_"):
        cols = [str(c) for c in model.feature_names_in_]
        if len(cols) == 0:
            raise ValueError("Model has empty feature_names_in_.")
        return cols
    n_features = int(getattr(model, "n_features_in_", 0))
    if n_features == len(FEATURE_COLS_16):
        return FEATURE_COLS_16
    if n_features == len(FEATURE_COLS_5):
        return FEATURE_COLS_5
    raise ValueError(
        f"Unsupported model feature count: {n_features}. "
        f"Expected {len(FEATURE_COLS_5)} or {len(FEATURE_COLS_16)}."
    )

def get_features_dict(cnt, roi_bgr):
    area = float(cv2.contourArea(cnt))
    perim = float(cv2.arcLength(cnt, True))
    if perim == 0:
        return None

    circularity = (4.0 * np.pi * area) / (perim * perim)

    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull))
    solidity = area / hull_area if hull_area > 0 else 0

    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio_invariant = float(max(w, h)) / (min(w, h) + 1e-9)

    obj_mask = np.zeros(roi_bgr.shape[:2], dtype=np.uint8)
    cv2.drawContours(obj_mask, [cnt], -1, 255, thickness=-1)

    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    mean_h, mean_s, _ = cv2.mean(hsv, mask=obj_mask)[:3]

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gabor_resp = cv2.filter2D(gray, cv2.CV_32F, GABOR_KERNEL)
    gabor_abs = np.abs(gabor_resp)
    gabor_mean, gabor_std = cv2.meanStdDev(gabor_abs, mask=obj_mask)
    gabor_mean = float(gabor_mean[0][0])
    gabor_std = float(gabor_std[0][0])

    hu = cv2.HuMoments(cv2.moments(cnt)).flatten()
    hu_log = [float(-np.sign(v) * np.log10(abs(v) + 1e-12)) for v in hu]

    return {
        "area": area,
        "aspect_ratio": aspect_ratio_invariant,
        "circularity": circularity,
        "solidity": solidity,
        "perimeter": perim,
        "mean_hue": float(mean_h),
        "mean_saturation": float(mean_s),
        "gabor_mean": gabor_mean,
        "gabor_std": gabor_std,
        "hu1": hu_log[0],
        "hu2": hu_log[1],
        "hu3": hu_log[2],
        "hu4": hu_log[3],
        "hu5": hu_log[4],
        "hu6": hu_log[5],
        "hu7": hu_log[6],
    }

class ULN2003Stepper:
    HALF_SEQ = [
        (1, 0, 0, 0),
        (1, 1, 0, 0),
        (0, 1, 0, 0),
        (0, 1, 1, 0),
        (0, 0, 1, 0),
        (0, 0, 1, 1),
        (0, 0, 0, 1),
        (1, 0, 0, 1),
    ]

    def __init__(self, in1, in2, in3, in4, step_delay=0.002):
        import RPi.GPIO as GPIO
        self.GPIO = GPIO
        self.pins = [in1, in2, in3, in4]
        self.step_delay = step_delay
        self._idx = 0

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        for p in self.pins:
            GPIO.setup(p, GPIO.OUT)
            GPIO.output(p, 0)

    def _write(self, a, b, c, d):
        self.GPIO.output(self.pins[0], a)
        self.GPIO.output(self.pins[1], b)
        self.GPIO.output(self.pins[2], c)
        self.GPIO.output(self.pins[3], d)

    def step(self, steps, direction=1):
        direction = 1 if direction >= 0 else -1
        for _ in range(abs(int(steps))):
            self._idx = (self._idx + direction) % len(self.HALF_SEQ)
            self._write(*self.HALF_SEQ[self._idx])
            time.sleep(self.step_delay)

    def release(self):
        self._write(0, 0, 0, 0)

    def cleanup(self):
        self.release()
        self.GPIO.cleanup()

def move_to(stepper, target_pos, current_pos):
    delta = int(target_pos - current_pos)
    if delta == 0:
        return current_pos
    stepper.step(abs(delta), direction=+1 if delta > 0 else -1)
    return target_pos

def run_flip(stepper, current_pos, direction):
    """
    Robust flip: always attempt to return to origin even if something goes wrong.
    """
    start_pos = current_pos
    target_pos = start_pos + (STEPS_135_DEG if direction > 0 else -STEPS_135_DEG)

    try:
        current_pos = move_to(stepper, target_pos, current_pos)
        time.sleep(RETURN_WAIT_SEC)
    finally:
        try:
            current_pos = move_to(stepper, start_pos, current_pos)
        except Exception as e:
            print(f"[FLIP] WARNING: return move failed: {e}")
        try:
            stepper.release()
        except:
            pass
        time.sleep(0.10)

    return current_pos

def nudge_flipper(stepper, stepper_pos, nudge_dir):
    """
    Small shake to help stuck objects clear. Returns updated stepper_pos.
    """
    if stepper is None:
        return stepper_pos
    start = stepper_pos
    target = start + (CLEAR_NUDGE_STEPS if nudge_dir > 0 else -CLEAR_NUDGE_STEPS)
    try:
        stepper_pos = move_to(stepper, target, stepper_pos)
        time.sleep(CLEAR_NUDGE_WAIT_SEC)
    finally:
        stepper_pos = move_to(stepper, start, stepper_pos)
        try:
            stepper.release()
        except:
            pass
    return stepper_pos

def recenter_flipper(stepper, stepper_pos):
    """
    Force the flipper back to home before resuming normal operation.
    """
    if stepper is None:
        return stepper_pos
    try:
        stepper_pos = move_to(stepper, 0, stepper_pos)
    except Exception as e:
        print(f"[FLIP] WARNING: recenter failed: {e}")
    try:
        stepper.release()
    except:
        pass
    return stepper_pos

def run_feeder_dose(feeder):
    print(f"[AUTO FEED] M1 dose (DOSE_STEPS={DOSE_STEPS})")
    feeder.set_dir(True)
    feeder.step_n(DOSE_STEPS)


def open_arduino_serial():
    """打开 Arduino 串口，失败返回 None。"""
    if serial is None:
        print("pyserial not installed. Arduino weight/feed signal disabled.")
        return None
    try:
        ser = serial.Serial(ARDUINO_SERIAL_PORT, ARDUINO_BAUD, timeout=0.01)
        print(f"Arduino serial opened: {ARDUINO_SERIAL_PORT} @ {ARDUINO_BAUD}")
        return ser
    except Exception as e:
        print(f"Arduino serial unavailable: {e}. Continuing without weight/feeder signal.")
        return None


def poll_arduino_serial(ser, state):
    """
    非阻塞读取串口，解析 Arduino → Pi 的消息并更新 state。

    目前支持（LCWS 相关）：
      - READY                 (Pi 发 'g' 后 Arduino 回复)
      - WEIGHT_RDY            (Arduino 已完成称重平均，下一条会是 WEIGHT_AVG)
      - WEIGHT_AVG,x.xx       (本次称重的平均重量，单位 g)
      - DONE / ARM_DONE       (机械臂动作结束，可选)

    state 推荐包含：
      last_weight_g (float|None)
      weight_rdy (bool)
      arduino_ready (bool)
      arm_done (bool)
    """
    if ser is None or (not getattr(ser, "is_open", False)):
        return

    try:
        while ser.in_waiting:
            line = ser.readline()
            if not line:
                break

            try:
                s = line.decode("utf-8", errors="ignore").strip()
            except Exception:
                continue

            if not s:
                continue

            # ---- handshake ----
            if s == "READY":
                state["arduino_ready"] = True
                continue

            # ---- weight protocol ----
            if s == "WEIGHT_RDY":
                state["weight_rdy"] = True
                continue

            if s.startswith("WEIGHT_AVG,"):
                try:
                    avg = float(s.split(",", 1)[1])
                    state["last_avg_weight_g"] = avg
                    state["last_weight_g"] = avg
                    state["last_weight_ts"] = time.time()
                    # 兼容：即使没有单独的 WEIGHT_RDY，也认为数据已就绪
                    state["weight_rdy"] = True
                except (ValueError, IndexError):
                    pass
                continue

            # ---- live weight (backward compatible) ----
            # Some firmwares stream: "Weight: 8.61 g"
            m = re.search(r"Weight:\s*([-+]?\d*\.?\d+)\s*g", s, flags=re.IGNORECASE)
            if m:
                try:
                    w = float(m.group(1))
                    state["last_live_weight_g"] = w
                    # If avg not provided yet, show live weight as fallback
                    if state.get("last_avg_weight_g") is None:
                        state["last_weight_g"] = w
                    state["last_weight_ts"] = time.time()
                    # Fallback: treat live weight as "ready" to avoid pipeline timeout on older firmware
                    state["weight_rdy"] = True
                except ValueError:
                    pass
                continue

            # ---- arm done (optional) ----
            if s in ("ARM_DONE", "DONE"):
                state["arm_done"] = True
                continue

            # ---- backward compatible messages (ignored) ----
            # FEED,0|1 旧协议：不再作为主控逻辑，仅保持兼容
            if s == "FEED,0":
                state["feeder_allowed_by_weight"] = False
            elif s == "FEED,1":
                state["feeder_allowed_by_weight"] = True

    except Exception as e:
        print(f"[ARDUINO] read error: {e}")


# =========================
# Main
# =========================
def main():
    global AUTO_FEED_ENABLED, AUTO_FLIP_ENABLED

    model = load_model()
    if model is None:
        sys.exit(1)
    try:
        model_feature_cols = get_model_feature_cols(model)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
    print(f"Using {len(model_feature_cols)} model features: {model_feature_cols}")

    # ---- GPIO init for feeders ----
    try:
        import RPi.GPIO as GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        for p in [M1_STEP, M1_DIR]:
            GPIO.setup(p, GPIO.OUT)
            GPIO.output(p, GPIO.LOW)

        feeder1 = StepperTMC2209_EN_GND(M1_STEP, M1_DIR, dir_normal=M1_DIR_NORMAL, name="M1")
        print("Feeder steppers initialized (EN hard-grounded).")
        print(f"Feeder timing: PULSE={STEP_PULSE_US}us GAP={STEP_GAP_US}us DOSE_STEPS={DOSE_STEPS}")
    except Exception as exc:
        feeder1 = None
        print(f"Feeder steppers unavailable: {exc}")
        print("Continuing without feeder controls.")

    roi_rect = clamp_roi(ROI_X, ROI_Y, ROI_W, ROI_H, FRAME_W, FRAME_H)
    rx, ry, rw, rh = roi_rect

    # ---- Arduino 串口（称重 + FEED 信号）----
    arduino_ser = open_arduino_serial()
    arduino_state = {
        "last_weight_g": None,
        "last_live_weight_g": None,
        "last_avg_weight_g": None,
        "last_weight_ts": 0.0,
        "weight_rdy": False,
        "await_weight": False,
        "weight_req_ts": 0.0,
        "target_reached": False,
        "arduino_ready": False,
        "arm_done": False,
        # backward compatible key (旧协议，不再作为主逻辑)
        "feeder_allowed_by_weight": True,
    }

    if arduino_ser is not None and ARDUINO_START_AFTER_PI_READY:
        try:
            time.sleep(2.0)  # Arduino 打开串口通常会 reset，等待它启动完成
            arduino_ser.write(b"g\n")
            arduino_ser.flush()
            print("[ARDUINO] Sent 'g' (start after Pi ready).")
            # 可选：等 READY 回复（短超时）
            deadline = time.time() + 1.0
            while time.time() < deadline:
                if arduino_ser.in_waiting:
                    line = arduino_ser.readline()
                    try:
                        s = line.decode("utf-8", errors="ignore").strip()
                        if s == "READY":
                            print("[ARDUINO] Arduino READY.")
                            break
                    except Exception:
                        pass
                time.sleep(0.02)
        except Exception as e:
            print(f"[ARDUINO] Send 'g' failed: {e}")

    picam2 = Picamera2()
    stepper = None
    stepper_pos = 0
    set_max_white()

    config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (FRAME_W, FRAME_H)})
    picam2.configure(config)
    picam2.start()

    time.sleep(1.5)
    try:
        picam2.set_controls({"AeEnable": False, "AwbEnable": False})
    except:
        pass

    # ===== Auto feeder runtime state =====
    last_feed_time = 0.0
    feed_started = False
    feed_stall_start = 0.0
    empty_streak = 0
    block_until = 0.0

    # ===== Auto flip runtime state =====
    last_flip_time = 0.0
    flip_armed = False
    decision_streak = 0
    last_decision = None

    # ===== gating state =====
    cv_gate_until = 0.0
    present_streak = 0

    waiting_clear = False
    clear_streak = 0

    # NEW: wait_clear timeout state
    clear_wait_start = 0.0
    clear_retry_count = 0
    last_flip_dir = +1  # remember last flip direction for nudges

    bg_gray = None
    try:
        try:
            stepper = ULN2003Stepper(
                FLIP_IN1, FLIP_IN2, FLIP_IN3, FLIP_IN4,
                step_delay=FLIP_STEP_DELAY
            )
            print("Flipper stepper initialized.")
        except Exception as exc:
            print(f"Flipper stepper unavailable: {exc}")
            print("Continuing without flipper controls.")

        print("Inference Mode (AUTO feeders + AUTO flip)")
        print("Keys:")
        print("  b: Capture Background")
        print("  1: Manual dose feeder")
        print("  a: Toggle AUTO feeder ON/OFF")
        print("  z: Toggle AUTO flip ON/OFF")
        print("  r/l: Manual flip right/left (debug)")
        print("  q: Quit")

        while True:
            poll_arduino_serial(arduino_ser, arduino_state)

            frame_rgb = picam2.capture_array()
            full_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            cv2.rectangle(full_bgr, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 2)
            roi_bgr = full_bgr[ry:ry + rh, rx:rx + rw]
            vis_roi = roi_bgr.copy()

            mask_area = 0
            beans_count = 0
            rocks_count = 0
            decision = None

            now = time.time()

            if bg_gray is None:
                cv2.putText(full_bgr, "Press 'b' for BACKGROUND", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                mask = get_object_mask(roi_bgr, bg_gray)
                mask_area = int(cv2.countNonZero(mask))
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # -------- CV Gate logic --------
                allow_decision = (now >= cv_gate_until)
                if DETECT_ONLY_AFTER_FEED:
                    allow_decision = allow_decision and ((now - last_feed_time) <= DETECT_WINDOW_AFTER_FEED_SEC)

                # determine "object present" with stable frames
                if now < cv_gate_until:
                    present_streak = 0
                else:
                    if mask_area > CLEAR_BEFORE_FLIP_THRESH:
                        present_streak += 1
                    else:
                        present_streak = 0

                object_present = (present_streak >= PRESENT_STABLE_FRAMES)

                # ---- run model prediction ----
                feature_list = []
                coords = []

                for cnt in contours:
                    if not is_valid_contour(cnt):
                        continue
                    feats = get_features_dict(cnt, roi_bgr)
                    if feats:
                        row = {name: feats.get(name, 0.0) for name in model_feature_cols}
                        feature_list.append(row)
                        coords.append(cv2.boundingRect(cnt))

                if feature_list:
                    feature_df = pd.DataFrame(feature_list, columns=model_feature_cols)
                    preds = model.predict(feature_df)

                    for i, label in enumerate(preds):
                        x, y, w, h = coords[i]
                        if label == 1:
                            beans_count += 1
                            color = (0, 255, 0)
                            text = "BEAN"
                        else:
                            rocks_count += 1
                            color = (0, 0, 255)
                            text = "ROCK"
                        cv2.rectangle(vis_roi, (x, y), (x+w, y+h), color, 2)
                        cv2.putText(vis_roi, text, (x, y - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # ---------- DECISION ----------
                if ROCK_WINS_IF_BOTH and rocks_count > 0:
                    decision = "ROCK"
                elif beans_count > 0:
                    decision = "BEAN"
                else:
                    decision = None

                # ===== Arduino WEIGHT handshake (pause motion until avg weight received) =====
                if arduino_state.get("await_weight", False):
                    # 超时保护：避免 Arduino 没回导致系统永久卡死
                    if (now - arduino_state.get("weight_req_ts", now)) > 3.0:
                        print("[WEIGHT] TIMEOUT waiting for Arduino. Unblocking pipeline.")
                        arduino_state["await_weight"] = False
                        arduino_state["weight_rdy"] = False

                    # 收到 WEIGHT_RDY/WEIGHT_AVG 后：读取并判断是否达到目标
                    if arduino_state.get("weight_rdy", False) and (arduino_state.get("last_weight_g") is not None):
                        avg_g = float(arduino_state["last_weight_g"])
                        print(f"[WEIGHT] Avg = {avg_g:.2f} g (target = {ARDUINO_WEIGHT_TARGET_G:.2f} g)")

                        # 本次称重完成，解除等待
                        arduino_state["await_weight"] = False
                        arduino_state["weight_rdy"] = False

                        # 达标：停止后续动作，并通知 Arduino 机械臂可以动
                        if avg_g >= ARDUINO_WEIGHT_TARGET_G:
                            arduino_state["target_reached"] = True
                            AUTO_FEED_ENABLED = False
                            AUTO_FLIP_ENABLED = False
                            if arduino_ser is not None:
                                try:
                                    arduino_ser.write(b"a\n")
                                    arduino_ser.flush()
                                    print("[ARDUINO] Sent 'a' (arm may move).")
                                except Exception as e:
                                    print(f"[ARDUINO] write 'a' failed: {e}")

                # 只要在等待称重结果或已达标，就禁止任何电机动作（feeder/flip/nudge）
                motion_blocked = bool(arduino_state.get("await_weight", False) or arduino_state.get("target_reached", False))

                # ===== AUTO FEED =====
                if feeder1 is not None and (not motion_blocked):
                    if now < block_until:
                        empty_streak = 0
                    else:
                        if mask_area < EMPTY_MASK_THRESH:
                            empty_streak += 1
                        else:
                            empty_streak = 0

                        if (AUTO_FEED_ENABLED and
                            empty_streak >= EMPTY_FRAMES and
                            (now - last_feed_time) >= FEED_COOLDOWN_SEC):
                            run_feeder_dose(feeder1)

                            last_feed_time = time.time()
                            feed_started = True
                            flip_armed = True

                            # after feed, gate CV for a bit (wait settle)
                            cv_gate_until = last_feed_time + CV_GATE_AFTER_FEED_SEC
                            present_streak = 0
                            decision_streak = 0
                            last_decision = None

                            empty_streak = 0
                            block_until = last_feed_time + POST_FEED_SETTLE_SEC

                # ===== WAIT_CLEAR (with timeout + nudge) =====
                if WAIT_CLEAR_AFTER_FLIP and waiting_clear:
                    if mask_area < CLEAR_AFTER_FLIP_THRESH:
                        clear_streak += 1
                    else:
                        clear_streak = 0

                    # cleared normally
                    if clear_streak >= CLEAR_STABLE_FRAMES:
                        waiting_clear = False
                        clear_streak = 0
                        decision_streak = 0
                        last_decision = None
                        clear_retry_count = 0

                    else:
                        # timeout -> nudge / unlock
                        if (not motion_blocked) and ((time.time() - clear_wait_start) >= CLEAR_TIMEOUT_SEC):
                            if (stepper is not None) and (clear_retry_count < CLEAR_RETRY_MAX):
                                clear_retry_count += 1
                                print(f"[CLEAR TIMEOUT] NUDGE {clear_retry_count}/{CLEAR_RETRY_MAX} (mask={mask_area})")

                                cv_gate_until = time.time() + 0.35  # ignore CV during shake
                                stepper_pos = nudge_flipper(stepper, stepper_pos, nudge_dir=last_flip_dir)

                                clear_wait_start = time.time()  # restart timer after nudge
                                clear_streak = 0               # require fresh clear streak
                            else:
                                # give up waiting forever
                                print(f"[CLEAR TIMEOUT] UNLOCK (mask={mask_area})")
                                stepper_pos = recenter_flipper(stepper, stepper_pos)
                                waiting_clear = False
                                clear_streak = 0
                                decision_streak = 0
                                last_decision = None
                                clear_retry_count = 0

                feeder_stalled = (
                    AUTO_FEED_ENABLED and
                    feed_started and
                    waiting_clear and
                    mask_area >= CLEAR_AFTER_FLIP_THRESH
                )

                if feeder_stalled:
                    if feed_stall_start == 0.0:
                        feed_stall_start = now
                else:
                    feed_stall_start = 0.0

                if (feeder1 is not None and (not motion_blocked) and
                    feed_stall_start > 0.0 and
                    (now - feed_stall_start) >= FEED_TIMEOUT_SEC):

                    print(f"[FEED TIMEOUT] WAIT_CLEAR stuck for {FEED_TIMEOUT_SEC:.1f}s. Forcing M1 dose and clearing stuck state.")
                    stepper_pos = recenter_flipper(stepper, stepper_pos)
                    waiting_clear = False
                    clear_streak = 0
                    clear_wait_start = 0.0
                    clear_retry_count = 0
                    present_streak = 0
                    decision_streak = 0
                    last_decision = None
                    empty_streak = 0
                    feed_stall_start = 0.0

                    run_feeder_dose(feeder1)

                    last_feed_time = time.time()
                    flip_armed = True
                    cv_gate_until = last_feed_time + CV_GATE_AFTER_FEED_SEC
                    block_until = last_feed_time + POST_FEED_SETTLE_SEC

                # ===== AUTO FLIP =====
                if AUTO_FLIP_ENABLED and (not motion_blocked) and (stepper is not None) and flip_armed and (not waiting_clear):
                    if allow_decision and object_present and (decision is not None):
                        if decision == last_decision:
                            decision_streak += 1
                        else:
                            last_decision = decision
                            decision_streak = 1
                    else:
                        last_decision = None
                        decision_streak = 0

                    if (decision_streak >= FLIP_STABLE_FRAMES and
                        (now - last_flip_time) >= FLIP_COOLDOWN_SEC):

                        flip_kind = last_decision  # "BEAN" or "ROCK"
                        flip_dir = FLIP_DIR_FOR_BEAN if flip_kind == "BEAN" else FLIP_DIR_FOR_ROCK
                        last_flip_dir = +1 if flip_dir > 0 else -1

                        print(f"[AUTO FLIP] decision={last_decision} dir={flip_dir}")
                        cv_gate_until = time.time() + 0.45
                        flip_armed = False

                        stepper_pos = run_flip(stepper, stepper_pos, direction=flip_dir)
                        last_flip_time = time.time()
                        decision_streak = 0
                        last_decision = None
                        present_streak = 0

                        # ===== NEW: request weight ONLY when flipping a BEAN =====
                        if (arduino_ser is not None) and (flip_kind == "BEAN"):
                            try:
                                arduino_state["await_weight"] = True
                                arduino_state["weight_rdy"] = False
                                arduino_state["weight_req_ts"] = time.time()

                                arduino_ser.write(b"r\n")
                                arduino_ser.flush()
                                print("[ARDUINO] Sent 'r' (avg weight). Pausing pipeline until WEIGHT_RDY/WEIGHT_AVG.")
                            except Exception as e:
                                arduino_state["await_weight"] = False
                                print(f"[ARDUINO] write 'r' failed: {e}")

                        if WAIT_CLEAR_AFTER_FLIP:
                            waiting_clear = True
                            clear_streak = 0
                            clear_wait_start = time.time()
                            clear_retry_count = 0

                        block_until = max(block_until, last_flip_time + 0.35)

                avg_w = arduino_state.get('last_avg_weight_g', None)
                live_w = arduino_state.get('last_live_weight_g', None)
                if avg_w is not None:
                    w_str = f"Wavg:{avg_w:.1f}g"
                elif live_w is not None:
                    w_str = f"W:{live_w:.1f}g"
                else:
                    w_str = "W:--"
                t_str = f"T:{ARDUINO_WEIGHT_TARGET_G:.1f}g"
                if arduino_state.get("target_reached", False):
                    phase_str = "TARGET:REACHED"
                elif arduino_state.get("await_weight", False):
                    phase_str = "WEIGH:WAIT"
                else:
                    phase_str = "WEIGH:IDLE"
                header = (
                    f"Beans:{beans_count} Rocks:{rocks_count} | mask:{mask_area} "
                    f"| {w_str}/{t_str} {phase_str} | empty:{empty_streak}/{EMPTY_FRAMES} AUTO_FEED:{'ON' if AUTO_FEED_ENABLED else 'OFF'} "
                    f"| decision:{decision or 'NONE'} streak:{decision_streak}/{FLIP_STABLE_FRAMES} AUTO_FLIP:{'ON' if AUTO_FLIP_ENABLED else 'OFF'} "
                    f"| gate:{max(0.0, cv_gate_until-now):.2f}s present:{present_streak}/{PRESENT_STABLE_FRAMES} "
                    f"| afterFeedWin:{'YES' if ((now-last_feed_time)<=DETECT_WINDOW_AFTER_FEED_SEC) else 'NO'} "
                    f"| waitClear:{'YES' if waiting_clear else 'NO'} clear:{clear_streak}/{CLEAR_STABLE_FRAMES} "
                    f"| clearTO:{max(0.0, CLEAR_TIMEOUT_SEC-(time.time()-clear_wait_start)):.2f}s retry:{clear_retry_count}/{CLEAR_RETRY_MAX}"
                )
                cv2.putText(full_bgr, header, (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)

                cv2.imshow("Mask", mask)

            cv2.imshow("Review", full_bgr)
            cv2.imshow("Inference", vis_roi)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                set_led_off()
                break

            elif key == ord('b'):
                print("Capturing background...")
                bg_gray = capture_background_gray(picam2, roi_rect)
                print("Background captured.")
                t = time.time()
                last_feed_time = t
                feed_started = False
                flip_armed = False
                feed_stall_start = 0.0
                last_flip_time = t
                empty_streak = 0
                decision_streak = 0
                last_decision = None
                present_streak = 0

                waiting_clear = False
                clear_streak = 0
                clear_wait_start = 0.0
                clear_retry_count = 0
                last_flip_dir = +1

                arduino_state["last_weight_g"] = None
                arduino_state["feeder_allowed_by_weight"] = True  # 新一批默认允许上料，等称重结果再更新

                cv_gate_until = t + POST_FEED_SETTLE_SEC
                block_until = t + POST_FEED_SETTLE_SEC

            elif key == ord('r'):
                if stepper is None:
                    print("Flipper not initialized.")
                    continue
                cv_gate_until = time.time() + 0.45
                last_flip_dir = +1
                stepper_pos = run_flip(stepper, stepper_pos, direction=+1)
                print("Manual flip right and returned.")
                if arduino_ser is not None:
                    try:
                        arduino_ser.write(b"r\n")
                        arduino_ser.flush()
                        print("[ARDUINO] Sent 'r' (avg weight request).")
                    except Exception as e:
                        print(f"[ARDUINO] write 's' failed: {e}")
                if WAIT_CLEAR_AFTER_FLIP:
                    waiting_clear = True
                    clear_streak = 0
                    clear_wait_start = time.time()
                    clear_retry_count = 0

            elif key == ord('l'):
                if stepper is None:
                    print("Flipper not initialized.")
                    continue
                cv_gate_until = time.time() + 0.45
                last_flip_dir = -1
                stepper_pos = run_flip(stepper, stepper_pos, direction=-1)
                print("Manual flip left and returned.")
                if arduino_ser is not None:
                    try:
                        arduino_ser.write(b"r\n")
                        arduino_ser.flush()
                        print("[ARDUINO] Sent 'r' (avg weight request).")
                    except Exception as e:
                        print(f"[ARDUINO] write 's' failed: {e}")
                if WAIT_CLEAR_AFTER_FLIP:
                    waiting_clear = True
                    clear_streak = 0
                    clear_wait_start = time.time()
                    clear_retry_count = 0

            elif key == ord('1'):
                if feeder1 is None:
                    print("Feeder 1 not initialized.")
                    continue
                print("[MANUAL FEED] M1 dose")
                run_feeder_dose(feeder1)
                last_feed_time = time.time()
                feed_started = True
                flip_armed = True
                feed_stall_start = 0.0
                empty_streak = 0
                block_until = last_feed_time + POST_FEED_SETTLE_SEC

                cv_gate_until = last_feed_time + CV_GATE_AFTER_FEED_SEC
                present_streak = 0
                decision_streak = 0
                last_decision = None

            elif key == ord('a'):
                AUTO_FEED_ENABLED = not AUTO_FEED_ENABLED
                print(f"[MODE] AUTO_FEED_ENABLED -> {AUTO_FEED_ENABLED}")

            elif key == ord('z'):
                AUTO_FLIP_ENABLED = not AUTO_FLIP_ENABLED
                print(f"[MODE] AUTO_FLIP_ENABLED -> {AUTO_FLIP_ENABLED}")

    finally:
        picam2.stop()
        cv2.destroyAllWindows()
        if stepper is not None:
            stepper.cleanup()
        if arduino_ser is not None and arduino_ser.is_open:
            try:
                arduino_ser.close()
            except Exception:
                pass

if __name__ == "__main__":
    main()
