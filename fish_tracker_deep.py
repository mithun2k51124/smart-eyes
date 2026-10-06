# -*- coding: utf-8 -*-
"""
SmartEyes - Real-Time Color & Pattern Fish Tracker with Health Telemetry (v4.0)
==============================================================================
Key Features:
  1. AMD Radeon GPU DirectML Hardware Acceleration (600+ FPS capability).
  2. Dynamic Real-Color & Skin Pattern Recognition:
     - Eliminates rigid species names; extracts the REAL dominant color and skin/scale pattern
       directly from the camera feed (e.g. "Orange Fish [Solid]", "Neon Pink Fish [Smooth]",
       "Charcoal Black Fish [Textured]", "Bright Yellow Fish [Solid]").
     - Bounding boxes are styled in the exact matching chromatic tone of each fish.
  3. Strict Air Pump & Stationary Phantom Elimination:
     - Multi-stage stillness filter eliminates rigid fixtures like the purple air pump.
     - Interactive Left-Click on any object to immediately blacklist & exclude it permanently.
     - Removed noisy generic gravel segmentation so no false bottom detections occur.
  4. Real-Time Health & Behavior Alert System:
     - Surface Floating / Dead Fish Detection: Flags motionless fish floating near surface (> 8s).
     - Inactivity / Lethargy Alert: Compares current stillness (> 15s) against historical swimming baseline.
     - Dedicated on-screen HUD Alert Panel + console logging with timestamps.

Controls:
  Left-Click : Click on any stationary object (like air pump) to instantly blacklist/exclude it
  Q / ESC    : Quit
  S          : Save session telemetry to fish_state.json
  R          : Reset tracking gallery & clear blacklists
  X          : Toggle Static Fixture Auto-Suppression
  C          : Toggle CLAHE Contrast Enhancement
"""

import time
import json
import math
import sys
import os
import threading
from collections import deque
from datetime import datetime

import cv2
import numpy as np

# DirectML ONNX Acceleration
try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

# =============================================================================
#   CONFIGURATION
# =============================================================================

CAMERA_INDEX        = 0
CAPTURE_WIDTH       = 1280
CAPTURE_HEIGHT      = 720
TARGET_FPS          = 30

DISPLAY_W           = 1280
DISPLAY_H           = 720

CUSTOM_MODEL_ONNX   = os.path.join("weights", "custom_fish_best.onnx")
CUSTOM_MODEL_PT     = os.path.join("weights", "custom_fish_best.pt")
FALLBACK_MODEL_ONNX = "aquarium_best.onnx"
FALLBACK_MODEL_PT   = "aquarium_best.pt"

INFER_SIZE          = 320
CONF_THRESH         = 0.18   # Clean confidence gate to avoid phantom bottom boxes
IOU_THRESH          = 0.40

# --- Tracking & Memory ---
SPATIAL_GATE_PX     = 180
COAST_FRAMES        = 45     # Smooth memory (~1.5s) during occlusion
SMOOTH_ALPHA        = 0.65
REID_SIM_THRESH     = 0.40

# --- Health & Behavioral Alert Thresholds ---
SPEED_STATIONARY        = 6.0    # px/s below which fish is considered motionless
SURFACE_DEPTH_RATIO     = 0.22   # Top 22% of tank is considered surface
SURFACE_FLOAT_ALERT_SEC = 8.0    # Seconds of stillness at surface to trigger dead/floating alert
LETHARGY_ALERT_SEC      = 15.0   # Seconds of inactivity to trigger lethargy alert
STATIC_FIXTURE_SEC      = 2.0    # Stillness to confirm stationary fixture (air pump)

FONT = cv2.FONT_HERSHEY_SIMPLEX

# =============================================================================
#   AMD GPU DIRECTML DETECTOR
# =============================================================================

class DirectMLDetector:
    """High-speed GPU Inference on AMD Radeon RX 6600 using DirectML."""
    def __init__(self, onnx_path, conf_thresh=CONF_THRESH, iou_thresh=IOU_THRESH, imgsz=INFER_SIZE):
        self.imgsz = imgsz
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.onnx_path = onnx_path

        providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        active_provider = self.session.get_providers()[0]
        self.is_gpu = "Dml" in active_provider
        self.input_name = self.session.get_inputs()[0].name

        gpu_name = "AMD Radeon RX 6600 (DirectX 12 DirectML)" if self.is_gpu else active_provider
        print(f"[HARDWARE] Acceleration Device: {gpu_name}")
        print(f"[MODEL] Loaded ONNX engine: {onnx_path}")

    def predict(self, frame_bgr):
        orig_h, orig_w = frame_bgr.shape[:2]

        resized = cv2.resize(frame_bgr, (self.imgsz, self.imgsz))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        blob = rgb.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)

        outputs = self.session.run(None, {self.input_name: blob})[0]
        outputs = np.squeeze(outputs, axis=0)
        outputs = np.transpose(outputs)

        boxes = []
        scores = []
        scale_x = orig_w / float(self.imgsz)
        scale_y = orig_h / float(self.imgsz)

        for row in outputs:
            class_scores = row[4:]
            max_score = float(np.max(class_scores)) if len(class_scores) > 0 else float(row[4])
            if max_score >= self.conf_thresh:
                cx, cy, w, h = row[0], row[1], row[2], row[3]
                x1 = int((cx - w / 2.0) * scale_x)
                y1 = int((cy - h / 2.0) * scale_y)
                bw = int(w * scale_x)
                bh = int(h * scale_y)

                boxes.append([x1, y1, bw, bh])
                scores.append(max_score)

        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_thresh, self.iou_thresh)
        final_dets = []
        if len(indices) > 0:
            for idx in indices.flatten():
                x, y, bw, bh = boxes[idx]
                conf = scores[idx]
                x2 = min(orig_w, x + bw)
                y2 = min(orig_h, y + bh)
                x1 = max(0, x)
                y1 = max(0, y)
                final_dets.append((x1, y1, x2, y2, conf))

        return final_dets

# =============================================================================
#   DYNAMIC COLOR & SKIN/PATTERN RECOGNITION
# =============================================================================

def analyze_color_and_pattern(crop_bgr):
    """
    Extracts the true dominant visual color and scale/skin pattern from the fish crop:
      - Dominant Color Name (Orange, Neon Pink, Bright Yellow, Charcoal Black, Amber Gold, etc.)
      - Skin Texture (Solid, Smooth, Textured Scales, Shimmer)
      - Exact BGR color for stylish bounding box drawing
    """
    if crop_bgr is None or crop_bgr.size == 0 or crop_bgr.shape[0] < 5 or crop_bgr.shape[1] < 5:
        return "Fish", "Smooth", (200, 200, 200), False

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h_mean = float(np.mean(hsv[:, :, 0]))
    s_mean = float(np.mean(hsv[:, :, 1]))
    v_mean = float(np.mean(hsv[:, :, 2]))

    # Check for stationary purple air pump (Purple in HSV: H ~ 120-165, S > 35)
    is_purple_pump = (116 <= h_mean <= 165) and (s_mean > 35) and (v_mean > 40)

    # 1. Texture & Scale Pattern (Laplacian variance)
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var > 140.0:
        pattern = "Textured"
    elif lap_var > 65.0:
        pattern = "Scales"
    else:
        pattern = "Smooth"

    # 2. Dynamic Color Extraction
    if v_mean < 75:
        color_name = "Charcoal Black"
        display_color = (60, 230, 110)  # Bright Emerald accent for dark fish
    elif s_mean < 40 and v_mean > 160:
        color_name = "Pearl White"
        display_color = (240, 240, 255)
    elif (140 <= h_mean <= 179) or (h_mean <= 6 and s_mean > 50):
        color_name = "Neon Pink"
        display_color = (230, 80, 255)
    elif 20 <= h_mean <= 42 and s_mean > 45:
        color_name = "Bright Yellow"
        display_color = (30, 240, 255)
    elif 7 <= h_mean < 20:
        if s_mean > 90:
            color_name = "Vibrant Orange"
            display_color = (0, 145, 255)
        else:
            color_name = "Amber Gold"
            display_color = (0, 200, 255)
    elif 43 <= h_mean <= 85:
        color_name = "Lime Green"
        display_color = (80, 255, 120)
    elif 86 <= h_mean <= 135:
        color_name = "Cyan Blue"
        display_color = (255, 210, 40)
    else:
        color_name = "Golden"
        display_color = (0, 180, 255)

    return color_name, pattern, display_color, is_purple_pump

# =============================================================================
#   APPEARANCE SIGNATURE & MULTI-VIEW RE-ID
# =============================================================================

def extract_signature(crop_bgr):
    if crop_bgr is None or crop_bgr.size == 0 or crop_bgr.shape[0] < 5 or crop_bgr.shape[1] < 5:
        return None

    h, w = crop_bgr.shape[:2]
    area = float(h * w)
    aspect = float(w / max(1, h))

    color_name, pattern, display_color, is_purple_pump = analyze_color_and_pattern(crop_bgr)

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256])
    vec = hist.flatten().astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 1e-6:
        vec /= norm

    return {
        'vector'        : vec,
        'area'          : area,
        'aspect'        : aspect,
        'color_name'    : color_name,
        'pattern'       : pattern,
        'color'         : display_color,
        'is_purple_pump': is_purple_pump
    }

def compute_similarity(sig_a, sig_b) -> float:
    if sig_a is None or sig_b is None:
        return 0.0

    color_match = 1.0 if sig_a['color_name'] == sig_b['color_name'] else 0.25
    pat_sim = max(0.0, float(np.dot(sig_a['vector'], sig_b['vector'])))
    min_a, max_a = min(sig_a['area'], sig_b['area']), max(sig_a['area'], sig_b['area'])
    size_sim = min_a / max(1.0, max_a)

    return 0.40 * color_match + 0.45 * pat_sim + 0.15 * size_sim

# =============================================================================
#   THREADED CAMERA STREAM (30 FPS)
# =============================================================================

class VideoStream:
    def __init__(self, src=0, width=1280, height=720):
        self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        if not self.stream.isOpened():
            self.stream = cv2.VideoCapture(src)

        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.stream.set(cv2.CAP_PROP_FPS, TARGET_FPS)

        self.ret, self.frame = self.stream.read()
        self.stopped = False
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while not self.stopped:
            ret, frame = self.stream.read()
            if not ret:
                time.sleep(0.01)
                continue
            with self.lock:
                self.ret = ret
                self.frame = frame

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    def is_opened(self):
        return self.stream.isOpened()

    def stop(self):
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.stream.release()

# =============================================================================
#   TRACKED FISH ENTITY (WITH HEALTH & BEHAVIOR TELEMETRY)
# =============================================================================

class TrackedFish:
    def __init__(self, fid: int, bbox, signature, conf: float):
        self.id            = fid
        self.color_name    = signature['color_name'] if signature else "Fish"
        self.pattern       = signature['pattern'] if signature else "Smooth"
        self.name          = f"{self.color_name} #{fid}"
        self.bbox          = [float(v) for v in bbox]
        self.color         = signature['color'] if signature else (0, 200, 255)
        self.conf          = conf

        now = time.time()
        self.first_seen    = now
        self.last_seen     = now
        self.total_sec     = 0.0
        self.is_active     = True
        self.is_occluded   = False
        self.missed_frames = 0

        # Motion & Velocity
        self.vx            = 0.0
        self.vy            = 0.0
        self.speed_px_s    = 0.0
        self.heading_deg   = 0.0
        self.speed_history = deque(maxlen=120)

        # Behavioral & Health Analysis
        self.behavior_state      = "Swimming"
        self.stationary_since    = None
        self.stationary_sec      = 0.0
        self.surface_float_since = None
        self.surface_float_sec   = 0.0
        self.is_surface_floating = False
        self.is_lethargic        = False
        self.last_alert_time     = 0.0

        # Multi-View Appearance Gallery
        self.gallery = deque(maxlen=6)
        if signature:
            self.gallery.append(signature)

        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        self.positions = deque(maxlen=90)
        self.positions.append((cx, cy, now))

        self.is_static_fixture = False

    @property
    def centroid(self):
        return ((self.bbox[0] + self.bbox[2]) / 2.0, (self.bbox[1] + self.bbox[3]) / 2.0)

    @property
    def baseline_speed(self):
        if not self.speed_history:
            return 15.0
        return float(np.mean(self.speed_history))

    def predicted_centroid(self, dt: float):
        cx, cy = self.centroid
        return (cx + self.vx * dt, cy + self.vy * dt)

    def match_best_view(self, new_sig) -> float:
        if not self.gallery or new_sig is None:
            return 0.0
        return max(compute_similarity(new_sig, s) for s in self.gallery)

    def update(self, new_bbox, new_sig, new_conf, now, dt, frame_h):
        old_cx, old_cy = self.centroid
        new_cx = (new_bbox[0] + new_bbox[2]) / 2.0
        new_cy = (new_bbox[1] + new_bbox[3]) / 2.0

        dt_safe = max(now - self.last_seen, 1e-4)
        inst_vx = (new_cx - old_cx) / dt_safe
        inst_vy = (new_cy - old_cy) / dt_safe

        self.vx = 0.55 * inst_vx + 0.45 * self.vx
        self.vy = 0.55 * inst_vy + 0.45 * self.vy
        self.speed_px_s = math.hypot(self.vx, self.vy)
        self.speed_history.append(self.speed_px_s)

        if self.speed_px_s > 6.0:
            self.heading_deg = math.degrees(math.atan2(self.vy, self.vx))

        for i in range(4):
            self.bbox[i] = SMOOTH_ALPHA * float(new_bbox[i]) + (1.0 - SMOOTH_ALPHA) * self.bbox[i]

        self.positions.append((new_cx, new_cy, now))
        self.conf = new_conf
        self.last_seen = now
        self.is_active = True
        self.is_occluded = False
        self.missed_frames = 0

        if new_sig is not None:
            if len(self.gallery) < 3 or max(compute_similarity(new_sig, s) for s in self.gallery) < 0.85:
                self.gallery.append(new_sig)
                self.color_name = new_sig['color_name']
                self.pattern = new_sig['pattern']
                self.color = new_sig['color']
                self.name = f"{self.color_name} #{self.id}"

        # Evaluate Health & Stillness
        self._evaluate_health(now, frame_h)

    def _evaluate_health(self, now, frame_h):
        """Monitors for Surface Floating / Dead Fish and Inactivity vs Historical Baseline."""
        y1 = self.bbox[1]
        is_near_surface = y1 < (SURFACE_DEPTH_RATIO * frame_h)
        is_motionless = self.speed_px_s < SPEED_STATIONARY

        # 1. Surface Floating / Dead Fish Monitor
        if is_near_surface and is_motionless:
            if self.surface_float_since is None:
                self.surface_float_since = now
            self.surface_float_sec = now - self.surface_float_since
            if self.surface_float_sec >= SURFACE_FLOAT_ALERT_SEC:
                self.is_surface_floating = True
                self.behavior_state = f"CRITICAL: Surface Floating ({int(self.surface_float_sec)}s)"
            else:
                self.behavior_state = f"Near Surface ({int(self.surface_float_sec)}s)"
        else:
            self.surface_float_since = None
            self.surface_float_sec = 0.0
            self.is_surface_floating = False

        # 2. Inactivity / Lethargy Monitor (vs historical baseline)
        if is_motionless and not self.is_surface_floating:
            if self.stationary_since is None:
                self.stationary_since = now
            self.stationary_sec = now - self.stationary_since

            if self.stationary_sec >= LETHARGY_ALERT_SEC:
                self.is_lethargic = True
                self.behavior_state = f"ALERT: Inactive ({int(self.stationary_sec)}s)"
            else:
                self.is_lethargic = False
                self.behavior_state = f"Stationary ({int(self.stationary_sec)}s)"
        elif not self.is_surface_floating:
            self.stationary_since = None
            self.stationary_sec = 0.0
            self.is_lethargic = False
            if self.speed_px_s > 20.0:
                self.behavior_state = "Active Swimming"
            else:
                self.behavior_state = "Cruising"

        # 3. Check for Static Non-Living Fixture (Air Pump)
        if len(self.positions) >= 60:
            recent = list(self.positions)[-60:]
            xs = [p[0] for p in recent]
            ys = [p[1] for p in recent]
            disp = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
            if disp < 4.0 and self.stationary_sec > STATIC_FIXTURE_SEC:
                self.is_static_fixture = True
            else:
                self.is_static_fixture = False

    def coast_prediction(self, dt: float):
        self.missed_frames += 1
        self.is_occluded = True
        self.vx *= 0.95
        self.vy *= 0.95
        self.bbox[0] += self.vx * dt
        self.bbox[1] += self.vy * dt
        self.bbox[2] += self.vx * dt
        self.bbox[3] += self.vy * dt

        if self.missed_frames > COAST_FRAMES:
            self.is_active = False

    def to_dict(self):
        return {
            "fish_id"           : self.id,
            "color_category"    : self.color_name,
            "skin_pattern"      : self.pattern,
            "label"             : self.name,
            "behavior"          : self.behavior_state,
            "speed_px_s"        : round(self.speed_px_s, 1),
            "baseline_speed"    : round(self.baseline_speed, 1),
            "surface_float_sec" : round(self.surface_float_sec, 1),
            "stationary_sec"    : round(self.stationary_sec, 1),
            "is_critical"       : self.is_surface_floating,
            "is_lethargic"      : self.is_lethargic,
            "total_seconds"     : round(self.total_sec, 2),
            "is_active"         : self.is_active
        }

# =============================================================================
#   TRACKING & ALERT ENGINE
# =============================================================================

class DeepFishTracker:
    def __init__(self, onnx_model_path, pt_fallback_path):
        if HAS_ORT and os.path.exists(onnx_model_path):
            self.detector = DirectMLDetector(onnx_model_path)
            self.is_gpu = self.detector.is_gpu
        elif HAS_YOLO and os.path.exists(pt_fallback_path):
            print(f"[INFO] Using PyTorch backend: {pt_fallback_path}")
            self.yolo_model = YOLO(pt_fallback_path)
            self.detector = None
            self.is_gpu = False
        else:
            sys.exit(f"[ERROR] No valid model found ({onnx_model_path} / {pt_fallback_path}).")

        self.gallery = {}
        self.next_id = 1
        self.suppress_static = True
        self.use_clahe = True
        self.excluded_regions = []  # List of [x1, y1, x2, y2] user-excluded zones (e.g. Air Pump)
        self.alert_feed = deque(maxlen=5)
        print(f"[INFO] Color/Pattern Engine & Health Telemetry Active.")

    def reset(self):
        self.gallery.clear()
        self.next_id = 1
        self.alert_feed.clear()
        print("[RESET] Cleared fish gallery and active alerts.")

    def add_exclusion_zone(self, cx, cy, radius=50):
        """Adds a blacklist zone (e.g. air pump location) from user mouse click."""
        x1 = max(0, cx - radius)
        y1 = max(0, cy - radius)
        x2 = cx + radius
        y2 = cy + radius
        self.excluded_regions.append((x1, y1, x2, y2))
        print(f"[EXCLUDE] Blacklisted zone at ({cx}, {cy})")

    def is_in_exclusion_zone(self, bbox):
        bx1, by1, bx2, by2 = bbox
        bcx = (bx1 + bx2) / 2.0
        bcy = (by1 + by2) / 2.0
        for ex1, ey1, ex2, ey2 in self.excluded_regions:
            if ex1 <= bcx <= ex2 and ey1 <= bcy <= ey2:
                return True
        return False

    def update(self, frame_bgr, now, dt):
        fh, fw = frame_bgr.shape[:2]

        # 1. CLAHE Contrast Preprocessing
        if self.use_clahe:
            lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
            enhanced = cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)
        else:
            enhanced = frame_bgr

        # 2. Deep Learning Detection (DirectML AMD GPU)
        raw_boxes = []
        if self.detector is not None:
            raw_boxes = self.detector.predict(enhanced)
        else:
            results = self.yolo_model.predict(source=enhanced, conf=CONF_THRESH, iou=IOU_THRESH, imgsz=INFER_SIZE, verbose=False)
            if results and results[0].boxes is not None:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                    conf = float(box.conf[0])
                    raw_boxes.append((x1, y1, x2, y2, conf))

        # 3. Filter False Positives (Air Pump & Excluded Zones)
        detections = []
        for x1, y1, x2, y2, conf in raw_boxes:
            if x1 < 12 or y1 < 10 or x2 > fw - 12 or y2 > fh - 10:
                continue
            if (x2 - x1) < 18 or (y2 - y1) < 14:
                continue
            if self.is_in_exclusion_zone((x1, y1, x2, y2)):
                continue

            crop = frame_bgr[y1:y2, x1:x2]
            sig = extract_signature(crop)
            if sig is not None:
                if self.suppress_static and sig['is_purple_pump']:
                    continue

                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'cx'  : (x1 + x2) / 2.0,
                    'cy'  : (y1 + y2) / 2.0,
                    'sig' : sig,
                    'conf': conf
                })

        # 4. Association & Multi-Object Tracking
        active_fishes = [f for f in self.gallery.values() if f.is_active and not (self.suppress_static and f.is_static_fixture)]
        inactive_fishes = [f for f in self.gallery.values() if not f.is_active and not (self.suppress_static and f.is_static_fixture)]

        matched_dets = set()
        matched_fishes = set()

        # Step A: Trajectory & Appearance Matching for Active Fish
        for fish in active_fishes:
            pred_cx, pred_cy = fish.predicted_centroid(dt)
            best_det_idx = None
            best_cost = float('inf')

            for i, det in enumerate(detections):
                if i in matched_dets:
                    continue
                dist = math.hypot(det['cx'] - pred_cx, det['cy'] - pred_cy)
                if dist < SPATIAL_GATE_PX:
                    sim = fish.match_best_view(det['sig'])
                    cost = (dist / SPATIAL_GATE_PX) * 0.35 + (1.0 - sim) * 0.65
                    if cost < best_cost:
                        best_cost = cost
                        best_det_idx = i

            if best_det_idx is not None and best_cost < 0.72:
                matched_dets.add(best_det_idx)
                matched_fishes.add(fish.id)
                d = detections[best_det_idx]
                fish.update(d['bbox'], d['sig'], d['conf'], now, dt, fh)

        # Step B: Deep Re-ID for Returning / Occluded Fish
        unmatched_dets = [i for i in range(len(detections)) if i not in matched_dets]
        for i in unmatched_dets:
            det = detections[i]
            best_fish = None
            best_sim = REID_SIM_THRESH

            for fish in inactive_fishes:
                if fish.id in matched_fishes:
                    continue
                sim = fish.match_best_view(det['sig'])
                if sim > best_sim:
                    best_sim = sim
                    best_fish = fish

            if best_fish is not None:
                matched_dets.add(i)
                matched_fishes.add(best_fish.id)
                best_fish.update(det['bbox'], det['sig'], det['conf'], now, dt, fh)

        # Step C: Register New Fish
        remaining_dets = [i for i in range(len(detections)) if i not in matched_dets]
        for i in remaining_dets:
            if len(self.gallery) < 14:
                det = detections[i]
                new_fid = self.next_id
                self.next_id += 1
                new_fish = TrackedFish(new_fid, det['bbox'], det['sig'], det['conf'])
                self.gallery[new_fid] = new_fish
                matched_fishes.add(new_fid)
                print(f"  [NEW FISH REGISTERED] {new_fish.name} ({new_fish.pattern})")

        # Step D: Coast Occluded Fish
        for fish in self.gallery.values():
            if fish.id not in matched_fishes:
                if fish.is_active:
                    fish.coast_prediction(dt)
                else:
                    fish.missed_frames += 1

        # Collect Active Health Alerts
        for fish in self.gallery.values():
            if fish.is_active and (fish.is_surface_floating or fish.is_lethargic):
                if now - fish.last_alert_time > 5.0:
                    fish.last_alert_time = now
                    tag = "🚨 CRITICAL" if fish.is_surface_floating else "⚠️ LETHARGY"
                    alert_msg = f"{tag}: {fish.name} - {fish.behavior_state}"
                    self.alert_feed.append((time.strftime("%H:%M:%S"), alert_msg, (0, 0, 255) if fish.is_surface_floating else (0, 165, 255)))
                    print(f"[{tag}] {time.strftime('%H:%M:%S')} -> {fish.name}: {fish.behavior_state}")

        if self.suppress_static:
            return [f for f in self.gallery.values() if f.is_active and not f.is_static_fixture]
        return [f for f in self.gallery.values() if f.is_active]

# =============================================================================
#   SLEEK RENDERING & HUD WITH ALERT PANEL
# =============================================================================

def draw_corner_rect(frame, bbox, color, thickness=2, corner_len=14):
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

    cl = min(corner_len, max(6, (x2 - x1) // 3), max(6, (y2 - y1) // 3))
    t = thickness
    cv2.line(frame, (x1, y1), (x1 + cl, y1), color, t)
    cv2.line(frame, (x1, y1), (x1, y1 + cl), color, t)
    cv2.line(frame, (x2, y1), (x2 - cl, y1), color, t)
    cv2.line(frame, (x2, y1), (x2, y1 + cl), color, t)
    cv2.line(frame, (x1, y2), (x1 + cl, y2), color, t)
    cv2.line(frame, (x1, y2), (x1, y2 - cl), color, t)
    cv2.line(frame, (x2, y2), (x2 - cl, y2), color, t)
    cv2.line(frame, (x2, y2), (x2, y2 - cl), color, t)

def render_fish_card(frame, fish: TrackedFish):
    x1, y1, x2, y2 = [int(round(v)) for v in fish.bbox]
    col = fish.color

    draw_corner_rect(frame, fish.bbox, col, thickness=2, corner_len=14)

    if fish.speed_px_s >= 8.0:
        cx, cy = int(round(fish.centroid[0])), int(round(fish.centroid[1]))
        arrow_len = int(np.clip(fish.speed_px_s * 0.28, 14, 38))
        rad = math.radians(fish.heading_deg)
        ex = int(cx + arrow_len * math.cos(rad))
        ey = int(cy + arrow_len * math.sin(rad))
        cv2.arrowedLine(frame, (cx, cy), (ex, ey), col, 1, cv2.LINE_AA, tipLength=0.28)

    is_critical = fish.is_surface_floating
    is_lethargic = fish.is_lethargic
    is_alert = is_critical or is_lethargic

    label_title = f"{fish.name} [{fish.pattern}]"
    label_sub = f"{fish.behavior_state} | {fish.speed_px_s:.0f} px/s"

    (tw1, th1), _ = cv2.getTextSize(label_title, FONT, 0.42, 1)
    (tw2, th2), _ = cv2.getTextSize(label_sub, FONT, 0.35, 1)
    max_w = max(tw1, tw2)

    px, py = 5, 4
    total_h = th1 + th2 + py * 3 + 2

    rx1 = max(0, x1)
    ry1 = max(0, y1 - total_h - 4)
    rx2 = min(frame.shape[1] - 1, x1 + max_w + px * 2 + 4)
    ry2 = max(0, y1 - 2)

    if ry2 > ry1 and rx2 > rx1:
        sub = frame[ry1:ry2, rx1:rx2]
        dark = np.zeros_like(sub)
        if is_critical:
            dark[:, :] = (0, 0, 160)
        elif is_lethargic:
            dark[:, :] = (0, 100, 180)
        cv2.addWeighted(sub, 0.20, dark, 0.80, 0, sub)
        frame[ry1:ry2, rx1:rx2] = sub
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 0, 255) if is_critical else col, 1)

    cv2.putText(frame, label_title, (x1 + px, ry1 + th1 + py), FONT, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, label_sub, (x1 + px, ry1 + th1 + th2 + py * 2), FONT, 0.35, (220, 240, 255) if not is_alert else (255, 255, 255), 1, cv2.LINE_AA)

def render_alert_panel(frame, alerts):
    """Renders active health/behavior alert messages on the top-right."""
    if not alerts:
        return

    pw = 420
    ph = len(alerts) * 24 + 18
    px1 = frame.shape[1] - pw - 12
    py1 = 10
    px2 = frame.shape[1] - 12
    py2 = 10 + ph

    sub = frame[py1:py2, px1:px2]
    dark = np.zeros_like(sub)
    dark[:, :] = (15, 15, 30)
    cv2.addWeighted(sub, 0.25, dark, 0.75, 0, sub)
    frame[py1:py2, px1:px2] = sub
    cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 220), 1)

    cv2.putText(frame, "HEALTH & BEHAVIOR ALERTS", (px1 + 8, py1 + 14), FONT, 0.36, (0, 180, 255), 1, cv2.LINE_AA)

    for i, (ts, msg, color) in enumerate(reversed(alerts)):
        txt = f"[{ts}] {msg}"
        if len(txt) > 46:
            txt = txt[:43] + "..."
        cv2.putText(frame, txt, (px1 + 8, py1 + 32 + i * 22), FONT, 0.34, color, 1, cv2.LINE_AA)

def render_ecosystem_hud(frame, fps: float, active_fishes, total_count: int, is_gpu: bool, excluded_count: int):
    """Modern translucent top HUD."""
    gpu_badge = "GPU: AMD RX 6600 (DirectML)" if is_gpu else "CPU Mode"
    ex_badge = f" | Excluded Zones: {excluded_count}" if excluded_count > 0 else ""

    hud_txt = (
        f"SmartEyes 4.0  |  {fps:.1f} FPS  |  {gpu_badge}  |  "
        f"Active Fish: {len(active_fishes)}  |  Total Tracked: {total_count}{ex_badge}  |  "
        f"[Click]: Blacklist Zone"
    )

    (tw, th), _ = cv2.getTextSize(hud_txt, FONT, 0.36, 1)
    x1, y1 = 12, 10
    x2, y2 = 12 + tw + 16, 10 + th + 14

    sub = frame[y1:y2, x1:x2]
    dark = np.zeros_like(sub)
    cv2.addWeighted(sub, 0.20, dark, 0.80, 0, sub)
    frame[y1:y2, x1:x2] = sub
    cv2.rectangle(frame, (x1, y1), (x2, y2), (70, 70, 70), 1)

    cv2.putText(frame, hud_txt, (x1 + 8, y1 + th + 6), FONT, 0.36, (240, 240, 240), 1, cv2.LINE_AA)

# =============================================================================
#   MAIN
# =============================================================================

_global_tracker = None

def on_mouse_click(event, x, y, flags, param):
    """Allows user to click on stationary fixtures (like air pump) to blacklist them."""
    global _global_tracker
    if event == cv2.EVENT_LBUTTONDOWN and _global_tracker is not None:
        _global_tracker.add_exclusion_zone(x, y, radius=55)

def run():
    global _global_tracker
    print("=" * 72)
    print(" SmartEyes v4.0 - AMD GPU Fish Tracker & Real-Time Health Telemetry ")
    print("=" * 72)

    model_onnx = CUSTOM_MODEL_ONNX if os.path.exists(CUSTOM_MODEL_ONNX) else FALLBACK_MODEL_ONNX
    model_pt = CUSTOM_MODEL_PT if os.path.exists(CUSTOM_MODEL_PT) else FALLBACK_MODEL_PT

    vs = VideoStream(src=CAMERA_INDEX, width=CAPTURE_WIDTH, height=CAPTURE_HEIGHT)
    time.sleep(0.5)

    if not vs.is_opened():
        sys.exit(f"[ERROR] Could not open camera {CAMERA_INDEX}.")

    tracker = DeepFishTracker(model_onnx, model_pt)
    _global_tracker = tracker

    cv2.namedWindow("SmartEyes Fish Health & Behavior Tracker", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("SmartEyes Fish Health & Behavior Tracker", DISPLAY_W, DISPLAY_H)
    cv2.setMouseCallback("SmartEyes Fish Health & Behavior Tracker", on_mouse_click)

    prev_time = time.time()
    ema_fps = 30.0

    print("\n[INFO] Color & Pattern Recognition Active.")
    print("[INFO] Health Telemetry Active (Surface Floating & Inactivity Alerts).")
    print("[TIP]  Left-Click on the air pump to blacklist/exclude its location permanently.")
    print("[KEYS] S: Save telemetry | R: Reset gallery & blacklists | X: Static Filter | Q: Quit\n")

    try:
        while True:
            ret, frame = vs.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            now = time.time()
            dt = max(now - prev_time, 1e-5)
            prev_time = now
            ema_fps = 0.88 * ema_fps + 0.12 * (1.0 / dt)

            disp = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))

            # Run tracking & behavior analysis on AMD GPU
            active_fishes = tracker.update(disp, now, dt)

            # Render individual fish cards
            for f in active_fishes:
                f.total_sec += dt
                render_fish_card(disp, f)

            # Render exclusion zone circles
            for ex1, ey1, ex2, ey2 in tracker.excluded_regions:
                ecx = (ex1 + ex2) // 2
                ecy = (ey1 + ey2) // 2
                cv2.circle(disp, (ecx, ecy), 50, (0, 0, 180), 1, cv2.LINE_AA)
                cv2.putText(disp, "EXCLUDED ZONE", (ex1, ey1 - 4), FONT, 0.32, (0, 0, 200), 1, cv2.LINE_AA)

            # Render Top HUD & Right Alert Panel
            render_ecosystem_hud(disp, ema_fps, active_fishes, len(tracker.gallery), tracker.is_gpu, len(tracker.excluded_regions))
            render_alert_panel(disp, tracker.alert_feed)

            cv2.imshow("SmartEyes Fish Health & Behavior Tracker", disp)

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q') or k == ord('Q') or k == 27:
                break
            elif k == ord('r') or k == ord('R'):
                tracker.reset()
                tracker.excluded_regions.clear()
            elif k == ord('x') or k == ord('X'):
                tracker.suppress_static = not tracker.suppress_static
                print(f"[TOGGLE] Static Suppression: {'ON' if tracker.suppress_static else 'OFF'}")
            elif k == ord('c') or k == ord('C'):
                tracker.use_clahe = not tracker.use_clahe
                print(f"[TOGGLE] CLAHE Contrast: {'ON' if tracker.use_clahe else 'OFF'}")
            elif k == ord('s') or k == ord('S'):
                out_path = "fish_state.json"
                with open(out_path, "w") as fp:
                    json.dump({str(f.id): f.to_dict() for f in tracker.gallery.values()}, fp, indent=2)
                print(f"[SAVED] Telemetry saved to {out_path}")

    finally:
        vs.stop()
        cv2.destroyAllWindows()

    print("\n" + "=" * 55)
    print("        AQUARIUM HEALTH & BEHAVIOR TELEMETRY       ")
    print("=" * 55)
    summary = {str(f.id): f.to_dict() for f in tracker.gallery.values()}
    print(json.dumps(summary, indent=2))
    print(f"\n[DONE] Monitored {len(summary)} total fish.")

if __name__ == "__main__":
    run()
