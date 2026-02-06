#!/usr/bin/env python3
# Run on Raspberry Pi:
#   sudo python3 flask_rf_server.py
#
# Why sudo? rpi_ws281x typically requires root on Raspberry Pi.

import time
import os
import sys
import json
import threading
import signal
from typing import Any, Dict, Optional, Tuple, List

import cv2
import numpy as np
import pickle
from flask import Flask, Response, jsonify, request, render_template_string

from picamera2 import Picamera2
from rpi_ws281x import PixelStrip, Color

# =========================
# LED (WS2812) Setup
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

# =========================
# Default Params (match your main_rf.py)
# =========================
DEFAULT_PARAMS = {
    "frame_w": 960,
    "frame_h": 540,
    "roi_x": 260,
    "roi_y": 90,
    "roi_w": 440,
    "roi_h": 360,
    "blur_k": 5,
    "morph_k": 5,
    "open_iters": 2,
    "close_iters": 2,
    "min_area": 800,
    "max_area": 40000,
    "bg_n_frames": 20,
}

MODEL_FILE = "bean_model.pkl"

# =========================
# Helpers (ported from main_rf.py, but parameterized)
# =========================
def clamp_roi(x: int, y: int, w: int, h: int, W: int, H: int) -> Tuple[int, int, int, int]:
    x = max(0, min(int(x), W - 1))
    y = max(0, min(int(y), H - 1))
    w = max(1, min(int(w), W - x))
    h = max(1, min(int(h), H - y))
    return x, y, w, h

def morph_cleanup(mask: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    ksz = int(params["morph_k"])
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=int(params["open_iters"]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=int(params["close_iters"]))
    return mask

def capture_background_gray(picam2: Picamera2, roi_rect: Tuple[int, int, int, int], params: Dict[str, Any]) -> np.ndarray:
    rx, ry, rw, rh = roi_rect
    n = int(params["bg_n_frames"])
    blur_k = int(params["blur_k"])
    if blur_k % 2 == 0:
        blur_k += 1  # must be odd for GaussianBlur

    acc: Optional[np.ndarray] = None
    for _ in range(n):
        frame_rgb = picam2.capture_array()
        full_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        roi = full_bgr[ry:ry + rh, rx:rx + rw]
        g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
        g = cv2.GaussianBlur(g, (blur_k, blur_k), 0)
        acc = g if acc is None else acc + g
    return (acc / float(n)).astype(np.uint8)

def get_object_mask(roi_bgr: np.ndarray, bg_gray: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    blur_k = int(params["blur_k"])
    if blur_k % 2 == 0:
        blur_k += 1
    g1 = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    g1 = cv2.GaussianBlur(g1, (blur_k, blur_k), 0)
    diff = cv2.absdiff(g1, bg_gray)
    _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = morph_cleanup(mask, params)
    return mask

def get_features_vector(cnt: np.ndarray) -> Optional[List[float]]:
    # MUST MATCH train_model.py order:
    # ['area', 'aspect_ratio', 'circularity', 'solidity', 'perimeter']
    area = float(cv2.contourArea(cnt))
    perim = float(cv2.arcLength(cnt, True))
    if perim == 0:
        return None
    circularity = (4.0 * np.pi * area) / (perim * perim)

    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull))
    solidity = area / hull_area if hull_area > 0 else 0.0

    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio_invariant = float(max(w, h)) / (min(w, h) + 1e-9)

    return [area, aspect_ratio_invariant, circularity, solidity, perim]

# =========================
# App State
# =========================
class AppState:
    def __init__(self):
        self.lock = threading.Lock()

        self.params: Dict[str, Any] = dict(DEFAULT_PARAMS)

        self.model = None
        self.model_loaded = False
        self.model_error: Optional[str] = None

        self.picam2: Optional[Picamera2] = None
        self.bg_gray: Optional[np.ndarray] = None

        self.latest_full_bgr: Optional[np.ndarray] = None
        self.latest_roi_bgr: Optional[np.ndarray] = None
        self.latest_mask: Optional[np.ndarray] = None

        self.latest_status: Dict[str, Any] = {
            "ok": False,
            "message": "starting",
            "bg_captured": False,
            "counts": {"bean": 0, "rock": 0},
            "objects": [],
            "fps": 0.0,
            "timestamp": None,
            "params": dict(self.params),
            "model_loaded": False,
        }

        self._stop = threading.Event()
        self._request_bg = threading.Event()

        self._last_t = time.time()
        self._fps = 0.0

def load_model_or_record_error(st: AppState):
    if not os.path.exists(MODEL_FILE):
        st.model_loaded = False
        st.model_error = f"{MODEL_FILE} not found. Run collect_data.py then train_model.py."
        return
    try:
        with open(MODEL_FILE, "rb") as f:
            st.model = pickle.load(f)
        st.model_loaded = True
        st.model_error = None
    except Exception as e:
        st.model_loaded = False
        st.model_error = f"Failed to load model: {e}"

def init_camera(st: AppState):
    p = st.params
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (int(p["frame_w"]), int(p["frame_h"]))}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1.5)
    try:
        picam2.set_controls({"AeEnable": False, "AwbEnable": False})
    except Exception:
        pass
    st.picam2 = picam2

def encode_jpeg(bgr_or_gray: np.ndarray, quality: int = 85) -> bytes:
    if bgr_or_gray.ndim == 2:
        img = bgr_or_gray
    else:
        img = bgr_or_gray
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return b""
    return buf.tobytes()

def inference_loop(st: AppState):
    # LED on
    try:
        set_max_white()
    except Exception:
        # LED not critical for server
        pass

    while not st._stop.is_set():
        with st.lock:
            picam2 = st.picam2
            p = dict(st.params)  # local snapshot
            model = st.model
            model_loaded = st.model_loaded
            bg_gray = st.bg_gray
            request_bg = st._request_bg.is_set()

        if picam2 is None:
            time.sleep(0.1)
            continue

        # Background capture request: do it inside this thread to avoid camera concurrency issues
        if request_bg:
            roi_rect = clamp_roi(p["roi_x"], p["roi_y"], p["roi_w"], p["roi_h"], p["frame_w"], p["frame_h"])
            try:
                new_bg = capture_background_gray(picam2, roi_rect, p)
                with st.lock:
                    st.bg_gray = new_bg
                    st._request_bg.clear()
            except Exception as e:
                with st.lock:
                    st._request_bg.clear()
                    st.latest_status.update({
                        "ok": False,
                        "message": f"capture_background failed: {e}",
                        "timestamp": time.time(),
                    })

        # Capture frame
        try:
            frame_rgb = picam2.capture_array()
            full_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        except Exception as e:
            with st.lock:
                st.latest_status.update({
                    "ok": False,
                    "message": f"camera capture failed: {e}",
                    "timestamp": time.time(),
                })
            time.sleep(0.05)
            continue

        # ROI
        rx, ry, rw, rh = clamp_roi(p["roi_x"], p["roi_y"], p["roi_w"], p["roi_h"], p["frame_w"], p["frame_h"])
        cv2.rectangle(full_bgr, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 2)
        roi_bgr = full_bgr[ry:ry + rh, rx:rx + rw]
        vis_roi = roi_bgr.copy()

        objects_out: List[Dict[str, Any]] = []
        beans_count = 0
        rocks_count = 0
        mask = None

        if bg_gray is None:
            cv2.putText(full_bgr, "BG not captured. Use /api/capture_background",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            # mask + contours + features
            try:
                mask = get_object_mask(roi_bgr, bg_gray, p)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                feature_list = []
                coords = []
                feats_per_obj = []

                for cnt in contours:
                    a = cv2.contourArea(cnt)
                    if a < float(p["min_area"]) or a > float(p["max_area"]):
                        continue

                    vec = get_features_vector(cnt)
                    if vec is None:
                        continue

                    x, y, w, h = cv2.boundingRect(cnt)
                    feature_list.append(vec)
                    coords.append((x, y, w, h))
                    feats_per_obj.append({
                        "area": vec[0],
                        "aspect_ratio": vec[1],
                        "circularity": vec[2],
                        "solidity": vec[3],
                        "perimeter": vec[4],
                    })

                preds = []
                probs = None

                if feature_list and model_loaded and model is not None:
                    preds = model.predict(feature_list)
                    # optional confidence
                    if hasattr(model, "predict_proba"):
                        try:
                            probs = model.predict_proba(feature_list)
                        except Exception:
                            probs = None

                for i in range(len(feature_list)):
                    x, y, w, h = coords[i]
                    label = int(preds[i]) if len(preds) == len(feature_list) else -1

                    if label == 1:
                        beans_count += 1
                        color = (0, 255, 0)
                        text = "BEAN"
                    elif label == 0:
                        rocks_count += 1
                        color = (0, 0, 255)
                        text = "ROCK"
                    else:
                        color = (255, 255, 0)
                        text = "UNKNOWN"

                    cv2.rectangle(vis_roi, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(vis_roi, text, (x, max(10, y - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    conf = None
                    if probs is not None and label in (0, 1):
                        # For binary classification, proba columns usually correspond to classes sorted.
                        # We assume class '1' is BEAN and '0' is ROCK as per training.
                        try:
                            # Find class index
                            classes = list(getattr(model, "classes_", [0, 1]))
                            idx = classes.index(label)
                            conf = float(probs[i][idx])
                        except Exception:
                            conf = None

                    objects_out.append({
                        "bbox": {"x": x, "y": y, "w": w, "h": h},
                        "label": text,
                        "label_id": label,
                        "confidence": conf,
                        "features": feats_per_obj[i],
                    })

                header = f"Beans: {beans_count} | Rocks: {rocks_count}"
                cv2.putText(full_bgr, header, (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            except Exception as e:
                cv2.putText(full_bgr, f"Inference error: {e}",
                            (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # FPS
        now = time.time()
        dt = now - st._last_t
        if dt > 0:
            st._fps = 0.9 * st._fps + 0.1 * (1.0 / dt)
        st._last_t = now

        # store latest
        with st.lock:
            st.latest_full_bgr = full_bgr
            st.latest_roi_bgr = vis_roi
            st.latest_mask = mask

            st.latest_status = {
                "ok": True,
                "message": "running",
                "bg_captured": (st.bg_gray is not None),
                "counts": {"bean": beans_count, "rock": rocks_count},
                "objects": objects_out,
                "fps": float(st._fps),
                "timestamp": now,
                "params": dict(st.params),
                "model_loaded": st.model_loaded,
                "model_error": st.model_error,
            }

        # control loop rate (10~20 FPS is usually enough for UI)
        time.sleep(0.03)

# =========================
# Flask App
# =========================
app = Flask(__name__)
STATE = AppState()

@app.get("/")
def home():
    # Minimal debug UI (optional): helps you immediately see what backend can show
    html = """
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8"/>
      <title>RF Backend Preview</title>
      <style>
        body { font-family: Arial; margin: 16px; }
        .row { display:flex; gap:16px; }
        .card { border:1px solid #ddd; border-radius:8px; padding:12px; }
        img { max-width: 100%; height: auto; border-radius: 8px; border:1px solid #eee; }
        pre { background:#f7f7f7; padding:12px; border-radius:8px; overflow:auto; max-height: 420px; }
        button { padding:10px 12px; border-radius:8px; border:1px solid #ddd; cursor:pointer; }
      </style>
    </head>
    <body>
      <h2>RF Backend Preview (Flask)</h2>
      <div class="card">
        <button onclick="captureBg()">Capture Background</button>
        <span id="bgState" style="margin-left:12px;"></span>
        <span id="fps" style="margin-left:12px;"></span>
      </div>

      <div class="row" style="margin-top:16px;">
        <div class="card" style="flex:2;">
          <h3>Frame</h3>
          <img id="frame" src="/api/frame.jpg" />
        </div>
        <div class="card" style="flex:2;">
          <h3>ROI</h3>
          <img id="roi" src="/api/roi.jpg" />
        </div>
        <div class="card" style="flex:1;">
          <h3>Mask</h3>
          <img id="mask" src="/api/mask.jpg" />
        </div>
      </div>

      <div class="card" style="margin-top:16px;">
        <h3>Status JSON</h3>
        <pre id="status"></pre>
      </div>

      <script>
        async function captureBg() {
          const res = await fetch("/api/capture_background", {method:"POST"});
          const j = await res.json();
          console.log(j);
        }

        async function tick() {
          // refresh images (cache-bust)
          const t = Date.now();
          document.getElementById("frame").src = "/api/frame.jpg?t=" + t;
          document.getElementById("roi").src   = "/api/roi.jpg?t=" + t;
          document.getElementById("mask").src  = "/api/mask.jpg?t=" + t;

          const res = await fetch("/api/status");
          const j = await res.json();
          document.getElementById("status").textContent = JSON.stringify(j, null, 2);
          document.getElementById("bgState").textContent = "BG: " + (j.bg_captured ? "captured" : "not captured");
          document.getElementById("fps").textContent = "FPS: " + (j.fps ? j.fps.toFixed(1) : "0.0");
        }

        setInterval(tick, 300);
        tick();
      </script>
    </body>
    </html>
    """
    return render_template_string(html)

@app.get("/api/status")
def api_status():
    with STATE.lock:
        return jsonify(STATE.latest_status)

@app.post("/api/capture_background")
def api_capture_background():
    with STATE.lock:
        STATE._request_bg.set()
    return jsonify({"ok": True, "message": "background capture requested"})

@app.get("/api/params")
def api_get_params():
    with STATE.lock:
        return jsonify({"ok": True, "params": dict(STATE.params), "bg_captured": (STATE.bg_gray is not None)})

@app.post("/api/params")
def api_set_params():
    """
    POST JSON like:
    {
      "roi_x": 260, "roi_y": 90, "roi_w": 440, "roi_h": 360,
      "min_area": 800, "max_area": 40000,
      "blur_k": 5, "morph_k": 5, "open_iters": 2, "close_iters": 2
    }
    """
    data = request.get_json(force=True, silent=True) or {}
    allowed = set(DEFAULT_PARAMS.keys()) - {"frame_w", "frame_h"}  # frame size changes need restart
    changed = []
    invalidate_bg = False

    with STATE.lock:
        for k, v in data.items():
            if k not in allowed:
                continue
            # basic validation
            if k in ("roi_x", "roi_y", "roi_w", "roi_h", "blur_k", "morph_k", "open_iters", "close_iters",
                     "min_area", "max_area", "bg_n_frames"):
                try:
                    v = int(v)
                except Exception:
                    continue
                if v <= 0 and k not in ("roi_x", "roi_y"):
                    continue
            STATE.params[k] = v
            changed.append(k)

            # background depends on ROI + blur
            if k in ("roi_x", "roi_y", "roi_w", "roi_h", "blur_k", "bg_n_frames"):
                invalidate_bg = True

        if invalidate_bg:
            STATE.bg_gray = None
            STATE.latest_mask = None

        STATE.latest_status["params"] = dict(STATE.params)

    return jsonify({"ok": True, "changed": changed, "bg_invalidated": invalidate_bg})

def _serve_latest_image(which: str) -> Response:
    with STATE.lock:
        if which == "frame":
            img = STATE.latest_full_bgr
        elif which == "roi":
            img = STATE.latest_roi_bgr
        elif which == "mask":
            img = STATE.latest_mask
        else:
            img = None

    if img is None:
        # return a tiny placeholder
        placeholder = np.zeros((120, 160, 3), dtype=np.uint8)
        cv2.putText(placeholder, "N/A", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return Response(encode_jpeg(placeholder), mimetype="image/jpeg")

    # mask is single-channel; keep as gray
    return Response(encode_jpeg(img), mimetype="image/jpeg")

@app.get("/api/frame.jpg")
def api_frame_jpg():
    return _serve_latest_image("frame")

@app.get("/api/roi.jpg")
def api_roi_jpg():
    return _serve_latest_image("roi")

@app.get("/api/mask.jpg")
def api_mask_jpg():
    return _serve_latest_image("mask")

# =========================
# Startup / Shutdown
# =========================
def shutdown_handler(sig, frame):
    STATE._stop.set()
    time.sleep(0.2)
    with STATE.lock:
        if STATE.picam2 is not None:
            try:
                STATE.picam2.stop()
            except Exception:
                pass
    sys.exit(0)

def main():
    # model
    load_model_or_record_error(STATE)

    # camera
    init_camera(STATE)

    # background thread
    th = threading.Thread(target=inference_loop, args=(STATE,), daemon=True)
    th.start()

    # signal handlers
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    # serve
    # If you want LAN access: host="0.0.0.0"
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)

if __name__ == "__main__":
    main()
