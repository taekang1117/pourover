import asyncio
import base64
import json
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set

import cv2
import numpy as np
import pandas as pd
import pickle
from aiohttp import web

from picamera2 import Picamera2
from rpi_ws281x import PixelStrip, Color

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
AUTO_FEED_ENABLED_DEFAULT = True
FEED_COOLDOWN_SEC = 1
FEED_TIMEOUT_SEC = 3.0
POST_FEED_SETTLE_SEC = 0.45
EMPTY_MASK_THRESH = 500
EMPTY_FRAMES = 10

# =========================
# Auto Flip Control
# =========================
AUTO_FLIP_ENABLED_DEFAULT = True
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
# WAIT_CLEAR timeout + nudge
# =========================
CLEAR_TIMEOUT_SEC = 2.0
CLEAR_RETRY_MAX = 2
CLEAR_NUDGE_STEPS = 120
CLEAR_NUDGE_WAIT_SEC = 0.15

# =========================
# Arduino UART (weight sensor + feeder signal)
# =========================
ARDUINO_SERIAL_PORT = "/dev/ttyACM0"
ARDUINO_BAUD = 115200
ARDUINO_WEIGHT_TARGET_G = 20.0
# Weighing cycle is ~2s nominal (1s settle + 1s sampling); keep margin for load/serial jitter.
WEIGH_TIMEOUT_S = 6.0
ARDUINO_START_AFTER_PI_READY = True
AUTO_ARM_ON_DONE = True  # auto-send 'a' when target weight is reached

# =========================
# Web GUI server
# =========================
HTTP_HOST = "0.0.0.0"
HTTP_PORT = 8000
VIDEO_FPS = 10  # send to browser
JPEG_QUALITY = 75

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
RETURN_WAIT_SEC = 0.5


# -------------------------
# Log capture: tee stdout -> queue (no need to rewrite all print(...))
# -------------------------
class TeeStdout:
    def __init__(self, real, line_queue: "queue.Queue[str]"):
        self._real = real
        self._q = line_queue
        self._buf = ""

    def write(self, s: str):
        self._real.write(s)
        self._real.flush()
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line.rstrip("\r")
            if line.strip():
                try:
                    self._q.put_nowait(line)
                except Exception:
                    pass

    def flush(self):
        try:
            self._real.flush()
        except Exception:
            pass


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

    def step_n(self, steps: int, stop_event: threading.Event, pulse_us=STEP_PULSE_US, gap_us=STEP_GAP_US):
        import RPi.GPIO as GPIO
        pulse_s = pulse_us / 1_000_000.0
        gap_s = gap_us / 1_000_000.0
        for _ in range(int(steps)):
            if stop_event.is_set():
                return
            GPIO.output(self.step_pin, GPIO.HIGH)
            time.sleep(pulse_s)
            GPIO.output(self.step_pin, GPIO.LOW)
            time.sleep(gap_s)


# =========================
# CV Helpers
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


# =========================
# Flipper motor
# =========================
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

    def step(self, steps, direction, stop_event: threading.Event):
        direction = 1 if direction >= 0 else -1
        for _ in range(abs(int(steps))):
            if stop_event.is_set():
                return
            self._idx = (self._idx + direction) % len(self.HALF_SEQ)
            self._write(*self.HALF_SEQ[self._idx])
            time.sleep(self.step_delay)

    def release(self):
        self._write(0, 0, 0, 0)

    def cleanup(self):
        self.release()
        self.GPIO.cleanup()


def move_to(stepper: ULN2003Stepper, target_pos, current_pos, stop_event: threading.Event):
    delta = int(target_pos - current_pos)
    if delta == 0:
        return current_pos
    stepper.step(abs(delta), direction=+1 if delta > 0 else -1, stop_event=stop_event)
    return target_pos


def run_flip(stepper: ULN2003Stepper, current_pos, direction, stop_event: threading.Event):
    start_pos = current_pos
    target_pos = start_pos + (STEPS_135_DEG if direction > 0 else -STEPS_135_DEG)
    try:
        current_pos = move_to(stepper, target_pos, current_pos, stop_event)
        time.sleep(RETURN_WAIT_SEC)
    finally:
        try:
            current_pos = move_to(stepper, start_pos, current_pos, stop_event)
        except Exception as e:
            print(f"[FLIP] WARNING: return move failed: {e}")
        try:
            stepper.release()
        except Exception:
            pass
        time.sleep(0.10)
    return current_pos


def nudge_flipper(stepper: ULN2003Stepper, stepper_pos, nudge_dir, stop_event: threading.Event):
    start = stepper_pos
    target = start + (CLEAR_NUDGE_STEPS if nudge_dir > 0 else -CLEAR_NUDGE_STEPS)
    try:
        stepper_pos = move_to(stepper, target, stepper_pos, stop_event)
        time.sleep(CLEAR_NUDGE_WAIT_SEC)
    finally:
        stepper_pos = move_to(stepper, start, stepper_pos, stop_event)
        try:
            stepper.release()
        except Exception:
            pass
    return stepper_pos


def recenter_flipper(stepper: ULN2003Stepper, stepper_pos, stop_event: threading.Event):
    try:
        stepper_pos = move_to(stepper, 0, stepper_pos, stop_event)
    except Exception as e:
        print(f"[FLIP] WARNING: recenter failed: {e}")
    try:
        stepper.release()
    except Exception:
        pass
    return stepper_pos


# =========================
# Arduino helpers
# =========================
def open_arduino_serial():
    if serial is None:
        print("pyserial not installed. Arduino weight/feed signal disabled.")
        return None
    try:
        ser = serial.Serial(ARDUINO_SERIAL_PORT, ARDUINO_BAUD, timeout=0.2)
        print(f"Arduino serial opened: {ARDUINO_SERIAL_PORT} @ {ARDUINO_BAUD}")
        return ser
    except Exception as e:
        print(f"Arduino serial unavailable: {e}. Continuing without weight/feeder signal.")
        return None


def send_arduino(ser, payload: bytes):
    if ser is None:
        return
    try:
        ser.write(payload)
        ser.flush()
    except Exception as e:
        print(f"[ARDUINO] write failed: {e}")


def poll_arduino_serial(ser, state: Dict[str, Any]):
    """Read and parse Arduino serial output.

    Notes:
      - Avoid stale-data decisions: WEIGHT_AVG/FEED/WEIGHT_RDY are only treated as a 'completed weigh cycle'
        if state['weigh_inflight'] is True.
      - Clear Arduino-side error on successful signals (READY / WEIGHT_AVG / FEED).
    """
    if ser is None or (not ser.is_open):
        return
    try:
        while ser.in_waiting:
            line = ser.readline()
            if not line:
                break
            s = line.decode("utf-8", errors="ignore").strip()
            if not s:
                continue

            state["last_arduino_line"] = s

            # ---- handshake / readiness ----
            if s == "READY":
                state["arduino_ready"] = True
                state["last_error"] = None
                # optional: keep logs concise
                print("[ARDUINO<<] READY")
                continue

            if s == "NOT_READY":
                # Arduino indicates handshake missing (or it rebooted)
                state["arduino_ready"] = False
                state["last_error"] = {"code": "NOT_READY", "message": "Arduino not ready (missing 'g' handshake)"}
                # end any in-flight weighing so the pipeline can recover
                state["weigh_inflight"] = False
                state["weight_done"] = True
                # force re-handshake soon
                state["last_g_sent_ts"] = 0.0
                print("[ARDUINO<<] NOT_READY")
                continue

            # ---- tare feedback ----
            if s == "TARE_OK":
                state["last_tare"] = "ok"
                state["last_error"] = None
                print("[ARDUINO<<] TARE_OK")
                continue
            if s == "TARE_FAIL":
                state["last_tare"] = "fail"
                state["last_error"] = {"code": "TARE_FAIL", "message": "Arduino tare failed"}
                print("[ARDUINO<<] TARE_FAIL")
                continue

            # ---- arm feedback ----
            if s == "ARM_START":
                state["arm_running"] = True
                print("[ARDUINO<<] ARM_START")
                continue
            if s == "ARM_DONE":
                state["arm_running"] = False
                state["arm_done"] = True
                print("[ARDUINO<<] ARM_DONE")
                continue

            # ---- weighing protocol ----
            if s.startswith("WEIGHT_AVG,"):
                try:
                    w = float(s.split(",", 1)[1])
                except Exception:
                    w = None
                if w is not None:
                    # always update display weight
                    state["last_weight_g"] = w
                    state["weight_is_stale"] = False
                    # if we're in an active weigh cycle, bind this value to the cycle
                    if state.get("weigh_inflight"):
                        state["pending_weight_g"] = w
                        state["last_error"] = None
                continue

            if s == "FEED,0":
                state["feeder_allowed_by_weight"] = False
                state["weight_is_stale"] = False
                if state.get("weigh_inflight"):
                    state["pending_feed_allowed"] = False
                    state["last_error"] = None
                continue

            if s == "FEED,1":
                state["feeder_allowed_by_weight"] = True
                state["weight_is_stale"] = False
                if state.get("weigh_inflight"):
                    state["pending_feed_allowed"] = True
                    state["last_error"] = None
                continue

            if s == "WEIGHT_ERR":
                # mark error, but do NOT let stale last_weight_g/FEED drive decisions
                state["last_error"] = {"code": "WEIGHT_ERR", "message": "Arduino weight sampling failed"}
                state["weight_is_stale"] = True
                state["pending_weight_g"] = None
                state["pending_feed_allowed"] = None
                state["weigh_inflight"] = False
                state["weight_done"] = True
                print("[ARDUINO<<] WEIGHT_ERR")
                continue

            if s == "WEIGHT_RDY":
                # only treat as a completion signal if this was triggered by a current request
                if state.get("weigh_inflight"):
                    state["weigh_inflight"] = False
                    state["weight_done"] = True
                    state["weight_is_stale"] = False
                else:
                    # late completion from an old cycle; ignore for gating
                    state["late_weight_rdy"] = True
                continue

            # ignore other lines, but keep as last_arduino_line for debugging
    except Exception as e:
        print(f"[ARDUINO] read error: {e}")
        state["last_error"] = {"code": "SERIAL_READ", "message": str(e)}


# =========================
# Core app
# =========================
@dataclass
class AppState:
    phase: str = "initializing"  # initializing | idle | running | weighting | done
    error: Optional[Dict[str, str]] = None
    weight_g: Optional[float] = None
    target_g: float = ARDUINO_WEIGHT_TARGET_G
    feed_allowed: bool = True
    beans_sorted: int = 0
    rocks_sorted: int = 0
    decision: str = "NONE"


class SorterApp:
    def __init__(self, log_queue):
        self.log_queue = log_queue
        self._lock = threading.Lock()

        self.state = AppState()

        # latest video frame as JPEG bytes
        self._latest_jpeg: Optional[bytes] = None
        self._latest_jpeg_ts = 0.0

        # control flags
        self._run_enabled = False
        self._stop_event = threading.Event()

        # arduino
        self.arduino_ser = None
        self.arduino_state: Dict[str, Any] = {
            "arduino_ready": False,
            "last_weight_g": None,
            "feeder_allowed_by_weight": True,
            "weight_done": False,
            "weigh_inflight": False,
            "pending_weight_g": None,
            "pending_feed_allowed": None,
            "weight_is_stale": False,
            "late_weight_rdy": False,
            "arm_running": False,
            "arm_done": False,
            "last_tare": None,
            "last_error": None,
            "last_arduino_line": "",
            "last_g_sent_ts": 0.0,
        }

        # vision
        self.model = None
        self.model_feature_cols = None

        self.picam2 = None
        self.roi_rect = None
        self.bg_gray = None

        # motors
        self.feeder1 = None
        self.stepper = None
        self.stepper_pos = 0

        # runtime states (mostly taken from your original script)
        self.AUTO_FEED_ENABLED = AUTO_FEED_ENABLED_DEFAULT
        self.AUTO_FLIP_ENABLED = AUTO_FLIP_ENABLED_DEFAULT

        self.last_feed_time = 0.0
        self.feed_started = False
        self.feed_stall_start = 0.0
        self.empty_streak = 0
        self.block_until = 0.0

        self.last_flip_time = 0.0
        self.flip_armed = False
        self.decision_streak = 0
        self.last_decision = None

        self.cv_gate_until = 0.0
        self.present_streak = 0

        self.waiting_clear = False
        self.clear_streak = 0

        self.clear_wait_start = 0.0
        self.clear_retry_count = 0
        self.last_flip_dir = +1

        # weigh gating
        self.await_weight = False
        self.await_deadline = 0.0

    # ---- state helpers ----
    def _set_phase(self, phase: str):
        with self._lock:
            self.state.phase = phase

    def _set_error(self, code: str, message: str):
        with self._lock:
            self.state.error = {"code": code, "message": message}

    def _clear_error(self):
        with self._lock:
            self.state.error = None
            # also clear Arduino-side latched error, otherwise UI banner may re-appear
            if isinstance(self.arduino_state, dict):
                self.arduino_state["last_error"] = None

    def snapshot_state(self) -> Dict[str, Any]:
        with self._lock:
            s = self.state
            return {
                "phase": s.phase,
                "error": s.error,
                "weight_g": s.weight_g,
                "target_g": s.target_g,
                "feed_allowed": s.feed_allowed,
                "beans_sorted": s.beans_sorted,
                "rocks_sorted": s.rocks_sorted,
                "decision": s.decision,
                "arduino": {
                    "ready": bool(self.arduino_state.get("arduino_ready", False)),
                    "last_line": self.arduino_state.get("last_arduino_line", ""),
                },
            }

    def get_latest_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_jpeg

    def _update_latest_jpeg(self, bgr_frame: np.ndarray):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(JPEG_QUALITY)]
        ok, buf = cv2.imencode('.jpg', bgr_frame, encode_param)
        if not ok:
            return
        jpg = buf.tobytes()
        with self._lock:
            self._latest_jpeg = jpg
            self._latest_jpeg_ts = time.time()

    # ---- commands from GUI ----
    def cmd_start(self):
        print("[GUI] Start")
        self._clear_error()
        # best-effort: re-handshake in case Arduino rebooted
        self._ensure_arduino_handshake(force=True)
        self._run_enabled = True

        # if done previously, restart a new batch
        with self._lock:
            if self.state.phase == "done":
                self.state.beans_sorted = 0
                self.state.rocks_sorted = 0
                self.state.weight_g = None
                self.state.feed_allowed = True

        if self.bg_gray is None:
            self._set_phase("initializing")
        else:
            self._set_phase("running")

    def cmd_stop(self):
        print("[GUI] Stop")
        self._run_enabled = False
        self.await_weight = False
        # clear any in-flight weigh signals to avoid stale completions
        self.arduino_state["weight_done"] = False
        self.arduino_state["weigh_inflight"] = False
        self.arduino_state["pending_weight_g"] = None
        self.arduino_state["pending_feed_allowed"] = None
        self._set_phase("idle")
        # best-effort: release motors
        try:
            if self.stepper is not None:
                self.stepper.release()
        except Exception:
            pass

    def cmd_tare(self):
        print("[GUI] Tare")
        if self.arduino_ser is None:
            self._set_error("NO_ARDUINO", "Arduino serial not available")
            return
        # handshake if needed
        self._ensure_arduino_ready(timeout_s=1.0)
        send_arduino(self.arduino_ser, b"t\n")

    def cmd_arm(self):
        print("[GUI] Arm")
        if self.arduino_ser is None:
            self._set_error("NO_ARDUINO", "Arduino serial not available")
            return
        if not self._ensure_arduino_ready(timeout_s=1.0):
            self._set_error("ARDUINO_NOT_READY", "Arduino not ready for robotic arm trigger")
            return
        send_arduino(self.arduino_ser, b"a\n")

    def cmd_capture_bg(self):
        print("[GUI] Capture background")
        self.bg_gray = None
        self._set_phase("initializing")

    # ---- Arduino helpers ----
    def _ensure_arduino_handshake(self, force: bool = False):
        """Send 'g' handshake if Arduino is not ready (or if forced)."""
        if self.arduino_ser is None:
            return
        if (not force) and self.arduino_state.get("arduino_ready"):
            return
        now = time.time()
        last = float(self.arduino_state.get("last_g_sent_ts", 0.0) or 0.0)
        if force or (now - last) > 1.0:
            send_arduino(self.arduino_ser, b"g\n")
            self.arduino_state["last_g_sent_ts"] = now
            print("[ARDUINO] Sent 'g' (handshake).")

    def _ensure_arduino_ready(self, timeout_s: float = 1.0) -> bool:
        """Best-effort wait for READY. Returns True if Arduino is ready."""
        if self.arduino_ser is None:
            return False
        self._ensure_arduino_handshake(force=True)
        t0 = time.time()
        while (time.time() - t0) < timeout_s:
            poll_arduino_serial(self.arduino_ser, self.arduino_state)
            if self.arduino_state.get("arduino_ready"):
                return True
            time.sleep(0.05)
        return bool(self.arduino_state.get("arduino_ready"))

    # ---- hardware init ----
    def init_hardware(self):
        self._set_phase("initializing")

        # model
        self.model = load_model()
        if self.model is None:
            raise RuntimeError("Model not found")
        self.model_feature_cols = get_model_feature_cols(self.model)
        print(f"Using {len(self.model_feature_cols)} model features")

        # GPIO init feeders
        try:
            import RPi.GPIO as GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            for p in [M1_STEP, M1_DIR]:
                GPIO.setup(p, GPIO.OUT)
                GPIO.output(p, GPIO.LOW)
            self.feeder1 = StepperTMC2209_EN_GND(M1_STEP, M1_DIR, dir_normal=M1_DIR_NORMAL, name="M1")
            print("Feeder steppers initialized (EN hard-grounded).")
        except Exception as exc:
            self.feeder1 = None
            print(f"Feeder steppers unavailable: {exc}")

        # flipper
        try:
            self.stepper = ULN2003Stepper(FLIP_IN1, FLIP_IN2, FLIP_IN3, FLIP_IN4, step_delay=FLIP_STEP_DELAY)
            print("Flipper stepper initialized.")
        except Exception as exc:
            self.stepper = None
            print(f"Flipper stepper unavailable: {exc}")

        # arduino
        self.arduino_ser = open_arduino_serial()
        if self.arduino_ser is not None and ARDUINO_START_AFTER_PI_READY:
            # give Arduino time (serial open may reset it)
            time.sleep(2.0)
            send_arduino(self.arduino_ser, b"g\n")
            self.arduino_state["last_g_sent_ts"] = time.time()
            print("[ARDUINO] Sent 'g' (start after Pi ready).")

        # camera
        # IMPORTANT: turn on LEDs BEFORE starting the camera so auto-exposure can settle.
        # Otherwise exposure may be locked with LEDs off, and the scene becomes overexposed once LEDs turn on.
        set_max_white()
        time.sleep(0.2)

        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(main={"format": "RGB888", "size": (FRAME_W, FRAME_H)})
        self.picam2.configure(config)
        self.picam2.start()

        # Let AE/AWB settle briefly, then lock them for stable CV features
        time.sleep(1.5)
        try:
            self.picam2.set_controls({"AeEnable": False, "AwbEnable": False})
        except Exception:
            pass

        self.roi_rect = clamp_roi(ROI_X, ROI_Y, ROI_W, ROI_H, FRAME_W, FRAME_H)
        self._set_phase("idle")

    def shutdown(self):
        self._stop_event.set()
        try:
            set_led_off()
        except Exception:
            pass
        try:
            if self.picam2 is not None:
                self.picam2.stop()
        except Exception:
            pass
        try:
            if self.stepper is not None:
                self.stepper.cleanup()
        except Exception:
            pass
        try:
            if self.arduino_ser is not None and self.arduino_ser.is_open:
                self.arduino_ser.close()
        except Exception:
            pass

    # ---- main loop ----
    def run(self):
        try:
            self.init_hardware()
        except Exception as e:
            self._set_error("INIT_FAIL", str(e))
            print(f"[ERROR] init failed: {e}")
            return

        rx, ry, rw, rh = self.roi_rect

        # send at ~VIDEO_FPS
        next_video_ts = time.time()
        video_period = 1.0 / max(1, int(VIDEO_FPS))

        while not self._stop_event.is_set():
            # keep handshake alive (Arduino may reboot)
            self._ensure_arduino_handshake(force=False)
            # poll Arduino
            poll_arduino_serial(self.arduino_ser, self.arduino_state)

            # propagate Arduino state to UI
            with self._lock:
                self.state.weight_g = self.arduino_state.get("last_weight_g")
                self.state.feed_allowed = bool(self.arduino_state.get("feeder_allowed_by_weight", True))
                # surface Arduino-side error
                if self.arduino_state.get("last_error"):
                    self.state.error = self.arduino_state["last_error"]

            # capture
            try:
                frame_rgb = self.picam2.capture_array()
            except Exception as e:
                self._set_error("CAMERA", str(e))
                time.sleep(0.1)
                continue

            full_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # Always draw ROI rect
            cv2.rectangle(full_bgr, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 2)

            roi_bgr = full_bgr[ry:ry + rh, rx:rx + rw]
            vis_roi = roi_bgr.copy()

            now = time.time()

            # Auto-capture background when starting and bg not ready
            if self._run_enabled and self.bg_gray is None:
                self._set_phase("initializing")
                print("[BG] Capturing background...")
                try:
                    self.bg_gray = capture_background_gray(self.picam2, self.roi_rect)
                    print("[BG] Background captured.")
                except Exception as e:
                    self._set_error("BG", str(e))
                    time.sleep(0.1)
                    continue
                # reset some runtime state
                t = time.time()
                self.last_feed_time = t
                self.feed_started = False
                self.flip_armed = False
                self.feed_stall_start = 0.0
                self.last_flip_time = t
                self.empty_streak = 0
                self.decision_streak = 0
                self.last_decision = None
                self.present_streak = 0

                self.waiting_clear = False
                self.clear_streak = 0
                self.clear_wait_start = 0.0
                self.clear_retry_count = 0
                self.last_flip_dir = +1

                self.cv_gate_until = t + POST_FEED_SETTLE_SEC
                self.block_until = t + POST_FEED_SETTLE_SEC
                self._set_phase("running")

            # detect
            mask_area = 0
            beans_count = 0
            rocks_count = 0
            decision = None

            if self.bg_gray is not None:
                mask = get_object_mask(roi_bgr, self.bg_gray)
                mask_area = int(cv2.countNonZero(mask))
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                allow_decision = (now >= self.cv_gate_until)
                if DETECT_ONLY_AFTER_FEED:
                    allow_decision = allow_decision and ((now - self.last_feed_time) <= DETECT_WINDOW_AFTER_FEED_SEC)

                # determine "object present" with stable frames
                if now < self.cv_gate_until:
                    self.present_streak = 0
                else:
                    if mask_area > CLEAR_BEFORE_FLIP_THRESH:
                        self.present_streak += 1
                    else:
                        self.present_streak = 0
                object_present = (self.present_streak >= PRESENT_STABLE_FRAMES)

                feature_list = []
                coords = []
                for cnt in contours:
                    if not is_valid_contour(cnt):
                        continue
                    feats = get_features_dict(cnt, roi_bgr)
                    if feats:
                        row = {name: feats.get(name, 0.0) for name in self.model_feature_cols}
                        feature_list.append(row)
                        coords.append(cv2.boundingRect(cnt))

                if feature_list:
                    feature_df = pd.DataFrame(feature_list, columns=self.model_feature_cols)
                    preds = self.model.predict(feature_df)
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
                        cv2.putText(vis_roi, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # decision
                if ROCK_WINS_IF_BOTH and rocks_count > 0:
                    decision = "ROCK"
                elif beans_count > 0:
                    decision = "BEAN"
                else:
                    decision = None

                # reflect decision to UI state
                with self._lock:
                    self.state.decision = decision or "NONE"

                # Weigh wait logic
                if self.await_weight:
                    self._set_phase("weighting")
                    if self.arduino_state.get("weight_done"):
                        # consume completion
                        self.arduino_state["weight_done"] = False
                        self.await_weight = False
                
                        err = self.arduino_state.get("last_error")
                        w = self.arduino_state.get("pending_weight_g")
                        feed_allowed = self.arduino_state.get("pending_feed_allowed")
                        # clear pending fields so they cannot affect future cycles
                        self.arduino_state["pending_weight_g"] = None
                        self.arduino_state["pending_feed_allowed"] = None
                
                        if err:
                            print("[WEIGHT] Arduino replied error: %s." % (err.get('code'),))
                            self._set_error(err.get('code', 'WEIGHT_ERR'), err.get('message', 'Arduino error'))
                            self._set_phase("running" if self._run_enabled else "idle")
                        elif (w is None) or (feed_allowed is None):
                            # avoid stale decisions if protocol messages were missing
                            self._set_error("WEIGHT_INCOMPLETE", "Weigh cycle completed without WEIGHT_AVG/FEED")
                            self._set_phase("running" if self._run_enabled else "idle")
                        else:
                            if (w >= ARDUINO_WEIGHT_TARGET_G) or (not bool(feed_allowed)):
                                print("[WEIGHT] Target reached (w=%s)." % (w,))
                                self._run_enabled = False
                                self._set_phase("done")
                                if AUTO_ARM_ON_DONE and self.arduino_ser is not None:
                                    # ensure ready before arm trigger
                                    if self._ensure_arduino_ready(timeout_s=1.0):
                                        print("[ARM] Auto-triggering robotic arm with command 'a'.")
                                        send_arduino(self.arduino_ser, b"a\n")
                                    else:
                                        self._set_error("ARDUINO_NOT_READY", "Target reached, but Arduino was not ready for arm trigger")
                            else:
                                self._set_phase("running" if self._run_enabled else "idle")
                    elif now > self.await_deadline:
                        print("[WEIGHT] TIMEOUT waiting for Arduino. Unblocking pipeline.")
                        self.await_weight = False
                        # clear in-flight flags so late WEIGHT_RDY won't be consumed as a future completion
                        self.arduino_state["weigh_inflight"] = False
                        self.arduino_state["weight_done"] = False
                        self.arduino_state["pending_weight_g"] = None
                        self.arduino_state["pending_feed_allowed"] = None
                        self._set_error("WEIGHT_TIMEOUT", "Timeout waiting for WEIGHT_RDY")
                        self._set_phase("running" if self._run_enabled else "idle")


                # If stopped, we do not drive motors.
                if not self._run_enabled:
                    self._set_phase("idle" if self.state.phase != "done" else "done")
                else:
                    # pause pipeline while weighing
                    if self.await_weight:
                        pass
                    else:
                        # ===== AUTO FEED =====
                        if self.feeder1 is not None:
                            if now < self.block_until:
                                self.empty_streak = 0
                            else:
                                if mask_area < EMPTY_MASK_THRESH:
                                    self.empty_streak += 1
                                else:
                                    self.empty_streak = 0

                                if (self.AUTO_FEED_ENABLED and
                                    (self.arduino_ser is None or self.arduino_state["feeder_allowed_by_weight"]) and
                                    self.empty_streak >= EMPTY_FRAMES and
                                    (now - self.last_feed_time) >= FEED_COOLDOWN_SEC):

                                    print(f"[AUTO FEED] M1 dose (DOSE_STEPS={DOSE_STEPS})")
                                    self.feeder1.set_dir(True)
                                    self.feeder1.step_n(DOSE_STEPS, stop_event=self._stop_event)

                                    self.last_feed_time = time.time()
                                    self.feed_started = True
                                    self.flip_armed = True

                                    self.cv_gate_until = self.last_feed_time + CV_GATE_AFTER_FEED_SEC
                                    self.present_streak = 0
                                    self.decision_streak = 0
                                    self.last_decision = None

                                    self.empty_streak = 0
                                    self.block_until = self.last_feed_time + POST_FEED_SETTLE_SEC

                        # ===== WAIT_CLEAR (with timeout + nudge) =====
                        if WAIT_CLEAR_AFTER_FLIP and self.waiting_clear:
                            if mask_area < CLEAR_AFTER_FLIP_THRESH:
                                self.clear_streak += 1
                            else:
                                self.clear_streak = 0

                            if self.clear_streak >= CLEAR_STABLE_FRAMES:
                                self.waiting_clear = False
                                self.clear_streak = 0
                                self.decision_streak = 0
                                self.last_decision = None
                                self.clear_retry_count = 0
                            else:
                                if (time.time() - self.clear_wait_start) >= CLEAR_TIMEOUT_SEC:
                                    if (self.stepper is not None) and (self.clear_retry_count < CLEAR_RETRY_MAX):
                                        self.clear_retry_count += 1
                                        print(f"[CLEAR TIMEOUT] NUDGE {self.clear_retry_count}/{CLEAR_RETRY_MAX} (mask={mask_area})")
                                        self.cv_gate_until = time.time() + 0.35
                                        self.stepper_pos = nudge_flipper(self.stepper, self.stepper_pos, nudge_dir=self.last_flip_dir, stop_event=self._stop_event)
                                        self.clear_wait_start = time.time()
                                        self.clear_streak = 0
                                    else:
                                        print(f"[CLEAR TIMEOUT] UNLOCK (mask={mask_area})")
                                        if self.stepper is not None:
                                            self.stepper_pos = recenter_flipper(self.stepper, self.stepper_pos, stop_event=self._stop_event)
                                        self.waiting_clear = False
                                        self.clear_streak = 0
                                        self.decision_streak = 0
                                        self.last_decision = None
                                        self.clear_retry_count = 0

                        feeder_stalled = (
                            self.AUTO_FEED_ENABLED and
                            self.feed_started and
                            self.waiting_clear and
                            mask_area >= CLEAR_AFTER_FLIP_THRESH
                        )
                        if feeder_stalled:
                            if self.feed_stall_start == 0.0:
                                self.feed_stall_start = now
                        else:
                            self.feed_stall_start = 0.0

                        if (self.feeder1 is not None and
                            self.feed_stall_start > 0.0 and
                            (now - self.feed_stall_start) >= FEED_TIMEOUT_SEC):

                            print(f"[FEED TIMEOUT] WAIT_CLEAR stuck for {FEED_TIMEOUT_SEC:.1f}s. Forcing M1 dose and clearing stuck state.")
                            if self.stepper is not None:
                                self.stepper_pos = recenter_flipper(self.stepper, self.stepper_pos, stop_event=self._stop_event)
                            self.waiting_clear = False
                            self.clear_streak = 0
                            self.clear_wait_start = 0.0
                            self.clear_retry_count = 0
                            self.present_streak = 0
                            self.decision_streak = 0
                            self.last_decision = None
                            self.empty_streak = 0
                            self.feed_stall_start = 0.0

                            print(f"[AUTO FEED] M1 dose (DOSE_STEPS={DOSE_STEPS})")
                            self.feeder1.set_dir(True)
                            self.feeder1.step_n(DOSE_STEPS, stop_event=self._stop_event)

                            self.last_feed_time = time.time()
                            self.flip_armed = True
                            self.cv_gate_until = self.last_feed_time + CV_GATE_AFTER_FEED_SEC
                            self.block_until = self.last_feed_time + POST_FEED_SETTLE_SEC

                        # ===== AUTO FLIP =====
                        if self.AUTO_FLIP_ENABLED and (self.stepper is not None) and self.flip_armed and (not self.waiting_clear):
                            if allow_decision and object_present and (decision is not None):
                                if decision == self.last_decision:
                                    self.decision_streak += 1
                                else:
                                    self.last_decision = decision
                                    self.decision_streak = 1
                            else:
                                self.last_decision = None
                                self.decision_streak = 0

                            if (self.decision_streak >= FLIP_STABLE_FRAMES and
                                (now - self.last_flip_time) >= FLIP_COOLDOWN_SEC):

                                flip_dir = FLIP_DIR_FOR_BEAN if self.last_decision == "BEAN" else FLIP_DIR_FOR_ROCK
                                self.last_flip_dir = +1 if flip_dir > 0 else -1

                                print(f"[AUTO FLIP] decision={self.last_decision} dir={flip_dir}")
                                self.cv_gate_until = time.time() + 0.45
                                self.flip_armed = False

                                self.stepper_pos = run_flip(self.stepper, self.stepper_pos, direction=flip_dir, stop_event=self._stop_event)
                                self.last_flip_time = time.time()
                                self.decision_streak = 0
                                flip_label = self.last_decision
                                self.last_decision = None
                                self.present_streak = 0

                                # increment sorted counters (basket count)
                                with self._lock:
                                    if flip_label == "BEAN":
                                        self.state.beans_sorted += 1
                                    else:
                                        self.state.rocks_sorted += 1

                                # trigger weigh
                                if self.arduino_ser is not None:
                                    # ensure Arduino is ready; re-handshake if it rebooted
                                    if not self._ensure_arduino_ready(timeout_s=1.0):
                                        self._set_error('NOT_READY', "Arduino not ready (handshake 'g' missing)")
                                        print("[WEIGHT] Arduino NOT_READY; skip weigh request this cycle.")
                                    else:
                                        # start a new weigh cycle; clear pending fields to avoid stale decisions
                                        self.arduino_state['weigh_inflight'] = True
                                        self.arduino_state['weight_done'] = False
                                        self.arduino_state['pending_weight_g'] = None
                                        self.arduino_state['pending_feed_allowed'] = None
                                        self.arduino_state['late_weight_rdy'] = False
                                        send_arduino(self.arduino_ser, b"r\n")
                                        print("[ARDUINO] Sent 'r' (avg weight). Pausing pipeline until WEIGHT_RDY.")
                                        self.await_weight = True
                                        self.await_deadline = time.time() + WEIGH_TIMEOUT_S

                                if WAIT_CLEAR_AFTER_FLIP:
                                    self.waiting_clear = True
                                    self.clear_streak = 0
                                    self.clear_wait_start = time.time()
                                    self.clear_retry_count = 0

                                self.block_until = max(self.block_until, self.last_flip_time + 0.35)

            # merge vis_roi back into full frame for streaming
            full_bgr[ry:ry + rh, rx:rx + rw] = vis_roi

            # encode/send video at fixed FPS
            if now >= next_video_ts:
                self._update_latest_jpeg(full_bgr)
                next_video_ts = now + video_period

            # small sleep to yield
            time.sleep(0.001)


# =========================
# Web server (aiohttp)
# =========================
UI_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>LCWS + Sorting Dashboard</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 0; background:#0b0f14; color:#e6edf3; }
    .wrap { display:grid; grid-template-columns: 1.6fr 1fr; gap: 12px; padding: 12px; height: 100vh; box-sizing: border-box; }
    .card { background:#121923; border:1px solid #1f2a37; border-radius: 12px; padding: 12px; }
    #videoCard { display:flex; flex-direction:column; }
    #canvas { width: 100%; height: auto; background:#000; border-radius: 10px; }
    .row { display:flex; gap: 10px; }
    .row .card { flex:1; }
    .big { font-size: 34px; font-weight: 700; }
    .label { opacity: .75; font-size: 12px; }
    .phase { font-weight: 700; }
    .btns { display:flex; gap: 10px; flex-wrap: wrap; }
    button { padding: 10px 14px; border-radius: 10px; border: 1px solid #2b3b4f; background:#0f1720; color:#e6edf3; cursor:pointer; }
    button:hover { background:#132033; }
    #error { display:none; padding: 10px 12px; border-radius: 10px; border: 1px solid #7f1d1d; background:#2a0b0b; color:#fecaca; margin-bottom: 10px; }
    #log { height: 240px; overflow:auto; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas; font-size: 12px; white-space: pre-wrap; background:#0a0f16; border-radius: 10px; border:1px solid #1f2a37; padding: 10px; }
    ul { margin: 8px 0 0 18px; padding:0; }
    li { margin: 4px 0; }
    .muted { opacity:.75 }
    .statusline { display:flex; justify-content:space-between; gap:10px; font-size:12px; opacity:.85; margin-top:8px; }
    .pill { border:1px solid #2b3b4f; padding:2px 8px; border-radius:999px; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card" id="videoCard">
      <div id="error"></div>
      <canvas id="canvas" width="960" height="540"></canvas>
      <div class="statusline muted">
        <div>Video: Original frame with ROI + bounding boxes.</div>
        <div>
          <span class="pill" id="wsStat">WS: --</span>
          <span class="pill" id="wsVStat">VIDEO: --</span>
        </div>
      </div>
    </div>

    <div style="display:flex; flex-direction:column; gap:12px">
      <div class="row">
        <div class="card">
          <div class="label">Current Weight (avg)</div>
          <div class="big" id="weight">-- g</div>
          <div class="muted">Target: <span id="target">--</span> g</div>
        </div>
        <div class="card">
          <div class="label">Beans in Basket</div>
          <div class="big" id="beans">0</div>
          <div class="muted">Rocks: <span id="rocks">0</span></div>
        </div>
      </div>

      <div class="card">
        <div class="label">State</div>
        <div class="big phase" id="phase">initializing</div>
        <div class="muted">Decision: <span id="decision">NONE</span> · Arduino: <span id="ard">--</span></div>
      </div>

      <div class="card">
        <div class="label">Controls</div>
        <div class="btns" style="margin-top:8px">
          <button id="startBtn">Start</button>
          <button id="stopBtn">Stop</button>
          <button id="tareBtn">Tare</button>
          <button id="armBtn">Arm</button>
        </div>
        <div class="muted" style="margin-top:10px">Hints</div>
        <ul class="muted" id="hints"></ul>
      </div>

      <div class="card">
        <div class="label">Log</div>
        <div id="log"></div>
      </div>

    </div>
  </div>

<script>
(function(){
  function $(id){ return document.getElementById(id); }
  var logEl = $('log');
  var errEl = $('error');
  var canvas = $('canvas');
  var ctx = canvas.getContext('2d');
  var wsStat = $('wsStat');
  var wsVStat = $('wsVStat');

  function addLog(line){
    try{
      var maxLines = 500;
      logEl.textContent += line + '\\n';
      var lines = logEl.textContent.split('\\n');
      if(lines.length > maxLines){
        logEl.textContent = lines.slice(lines.length - maxLines).join('\\n');
      }
      logEl.scrollTop = logEl.scrollHeight;
    }catch(e){}
  }

  function showError(err){
    if(!err){
      errEl.style.display='none';
      errEl.textContent='';
      return;
    }
    errEl.style.display='block';
    errEl.textContent = '[' + (err.code || 'ERR') + '] ' + (err.message || '');
  }

  function setHints(hints){
    var ul = $('hints');
    ul.innerHTML='';
    if(!hints) return;
    for(var i=0;i<hints.length;i++){
      var li=document.createElement('li');
      li.textContent = hints[i];
      ul.appendChild(li);
    }
  }

  function wsUrl(path){
    var proto = (location.protocol === 'https:') ? 'wss://' : 'ws://';
    return proto + location.host + path;
  }

  // JSON WS (state + commands + logs)
  var ws = null;
  var wsReconnectMs = 1000;

  function connectWS(){
    try{
      wsStat.textContent = 'WS: connecting';
      ws = new WebSocket(wsUrl('/ws'));
    }catch(e){
      wsStat.textContent = 'WS: error';
      addLog('[WS] connect exception (reconnecting...)');
      setTimeout(connectWS, wsReconnectMs);
      wsReconnectMs = Math.min(10000, wsReconnectMs * 2);
      return;
    }

    ws.onopen = function(){
      wsStat.textContent='WS: connected';
      addLog('[WS] connected');
      wsReconnectMs = 1000;
    };
    ws.onclose = function(){
      wsStat.textContent='WS: disconnected';
      addLog('[WS] disconnected (reconnecting...)');
      setTimeout(connectWS, wsReconnectMs);
      wsReconnectMs = Math.min(10000, wsReconnectMs * 2);
    };
    ws.onerror = function(){
      wsStat.textContent='WS: error';
      addLog('[WS] error');
      // onclose will retry
    };
    ws.onmessage = handleWSMessage;
  }

  connectWS();


  function handleWSMessage(ev){
    var msg;
    try{ msg = JSON.parse(ev.data); }catch(e){ return; }
    if(msg.type === 'state'){
      var d = msg.data || {};
      $('phase').textContent = d.phase || 'initializing';
      $('target').textContent = (typeof d.target_g === 'number') ? String(d.target_g) : '--';
      $('beans').textContent = (typeof d.beans_sorted === 'number') ? String(d.beans_sorted) : '0';
      $('rocks').textContent = (typeof d.rocks_sorted === 'number') ? String(d.rocks_sorted) : '0';
      $('decision').textContent = d.decision || 'NONE';
      var ard = d.arduino || {};
      $('ard').textContent = ard.ready ? 'READY' : 'NOT_READY';
      if(d.weight_g !== null && typeof d.weight_g === 'number'){
        $('weight').textContent = d.weight_g.toFixed(2) + ' g';
      }else{
        $('weight').textContent = '-- g';
      }
      showError(d.error);
    }else if(msg.type === 'log'){
      addLog(msg.line);
    }else if(msg.type === 'controls'){
      setHints(msg.hints);
    }else if(msg.type === 'error'){
      showError(msg.error);
    }
  }
function sendCmd(cmd){
    if(!ws || ws.readyState !== 1){
      addLog('[WS] not connected; cannot send ' + cmd);
      return;
    }
    ws.send(JSON.stringify({type:'cmd', cmd: cmd}));
    addLog('[CMD] ' + cmd);
  }

  $('startBtn').onclick = function(){ sendCmd('start'); };
  $('stopBtn').onclick  = function(){ sendCmd('stop');  };
  $('tareBtn').onclick  = function(){ sendCmd('tare');  };
  $('armBtn').onclick   = function(){ sendCmd('arm');   };

  // Video WS (binary JPEG)
  var wsv = null;
  var wsvReconnectMs = 1000;

  function connectVideo(){
    try{
      wsVStat.textContent = 'VIDEO: connecting';
      wsv = new WebSocket(wsUrl('/ws/video'));
      wsv.binaryType = 'arraybuffer';
    }catch(e){
      wsVStat.textContent = 'VIDEO: error';
      addLog('[WS-VIDEO] connect exception (reconnecting...)');
      setTimeout(connectVideo, wsvReconnectMs);
      wsvReconnectMs = Math.min(10000, wsvReconnectMs * 2);
      return;
    }

    wsv.onopen = function(){
      wsVStat.textContent='VIDEO: connected';
      addLog('[WS-VIDEO] connected');
      wsvReconnectMs = 1000;
    };
    wsv.onclose = function(){
      wsVStat.textContent='VIDEO: disconnected';
      addLog('[WS-VIDEO] disconnected (reconnecting...)');
      setTimeout(connectVideo, wsvReconnectMs);
      wsvReconnectMs = Math.min(10000, wsvReconnectMs * 2);
    };
    wsv.onerror = function(){
      wsVStat.textContent='VIDEO: error';
      addLog('[WS-VIDEO] error');
      // onclose will retry
    };
    wsv.onmessage = handleVideoMessage;
  }

  connectVideo();


  // Fallback decode: Image + ObjectURL (more compatible than createImageBitmap)
  var img = new Image();
  var drawing = false;
  img.onload = function(){
    try{
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    }catch(e){}
    drawing = false;
  };

  function handleVideoMessage(ev){
    try{
      if(drawing) return; // drop frames if decode is behind
      drawing = true;
      var blob = new Blob([ev.data], {type:'image/jpeg'});
      var url = (window.URL || window.webkitURL).createObjectURL(blob);
      img.src = url;
      // revoke after load (best effort)
      setTimeout(function(){
        try{ (window.URL || window.webkitURL).revokeObjectURL(url); }catch(e){}
      }, 5000);
    }catch(e){
      drawing = false;
    }
  }
})();
</script>
</body>
</html>"""


async def index(_request):
    return web.Response(text=UI_HTML, content_type="text/html")


async def ws_handler(request):
    app: web.Application = request.app
    sorter: SorterApp = app["sorter"]

    ws = web.WebSocketResponse(heartbeat=None)
    await ws.prepare(request)

    app["ws_clients"].add(ws)

    # send controls + initial state
    hints = [
        "Start: begin auto sorting", 
        "Stop: stop all actions", 
        "Tare: tare the load cell (basket empty)", 
        "Arm: manually trigger the robotic arm once Arduino is ready",
        "State meanings: initializing / idle / running / weighting / done",
        "Video shows ROI + object boxes (BEAN/ROCK)",
    ]
    await ws.send_str(json.dumps({"type": "controls", "hints": hints}))

    async def state_loop():
        while not ws.closed:
            try:
                await ws.send_str(json.dumps({"type": "state", "data": sorter.snapshot_state()}))
            except Exception:
                break
            await asyncio.sleep(0.1)

    state_task = asyncio.create_task(state_loop())

    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                try:
                    payload = json.loads(msg.data)
                except Exception:
                    continue
                if payload.get("type") == "cmd":
                    cmd = str(payload.get("cmd", ""))
                    if cmd == "start":
                        sorter.cmd_start()
                    elif cmd == "stop":
                        sorter.cmd_stop()
                    elif cmd == "tare":
                        sorter.cmd_tare()
                    elif cmd == "arm":
                        sorter.cmd_arm()
                    elif cmd == "capture_bg":
                        sorter.cmd_capture_bg()
                    else:
                        pass
            elif msg.type == web.WSMsgType.ERROR:
                break
    finally:
        state_task.cancel()
        app["ws_clients"].discard(ws)

    return ws


async def ws_video_handler(request):
    app: web.Application = request.app
    sorter: SorterApp = app["sorter"]

    ws = web.WebSocketResponse(heartbeat=None)
    await ws.prepare(request)

    period = 1.0 / max(1, int(VIDEO_FPS))

    try:
        while not ws.closed:
            jpg = sorter.get_latest_jpeg()
            if jpg is not None:
                try:
                    await ws.send_bytes(jpg)
                except Exception:
                    # client disconnected or send failed
                    break

            await asyncio.sleep(period)

    finally:
        # aiohttp will close ws automatically; keep for symmetry
        pass

    return ws


async def log_broadcaster(app: web.Application):
    """Broadcast captured stdout lines to all ws clients."""
    q = app["log_queue"]
    while True:
        line = await asyncio.get_event_loop().run_in_executor(None, q.get)
        dead = []
        for ws in list(app["ws_clients"]):
            if ws.closed:
                dead.append(ws)
                continue
            try:
                await ws.send_str(json.dumps({"type": "log", "line": line}))
            except Exception:
                dead.append(ws)
        for ws in dead:
            app["ws_clients"].discard(ws)


def main():
    import queue

    log_q: "queue.Queue[str]" = queue.Queue(maxsize=2000)
    sys.stdout = TeeStdout(sys.__stdout__, log_q)

    sorter = SorterApp(log_q)

    t = threading.Thread(target=sorter.run, daemon=True)
    t.start()

    app = web.Application()
    app["sorter"] = sorter
    app["ws_clients"] = set()
    app["log_queue"] = log_q

    app.router.add_get("/", index)
    app.router.add_get("/ws", ws_handler)
    app.router.add_get("/ws/video", ws_video_handler)

    async def on_startup(app):
        app["log_task"] = asyncio.create_task(log_broadcaster(app))

    async def on_cleanup(app):
        try:
            app["log_task"].cancel()
        except Exception:
            pass

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    print(f"[WEB] Open in Chromium: http://localhost:{HTTP_PORT}")
    try:
        web.run_app(app, host=HTTP_HOST, port=HTTP_PORT)
    finally:
        sorter.shutdown()


if __name__ == "__main__":
    main()
