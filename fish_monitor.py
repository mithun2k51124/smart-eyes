# -*- coding: utf-8 -*-
"""
SmartEyes Fish Monitor v7 - Rock-Solid Stationary Stone Rejection
==================================================================
Key Fixes & Innovations:
  1. Permanent Stationary Stone Blacklist:
     - Real fish in water are never frozen in the exact same pixel position.
     - Monitors every candidate centroid: if an object's net displacement is
       < 3.5 px over 15 frames (0.5s), it is marked as a STATIONARY STONE.
     - Once blacklisted, all detections at that rock location are permanently
       silenced and dropped!
  2. Bottom Gravel Bed White-Stone Optical Filter:
     - Detects low-saturation / high-brightness white pebbles at the tank bottom
       and rejects them instantly.
  3. Strict Motion Verification (Bug Fix):
     - Eliminated the bug that allowed stationary objects through after locking.
     - Candidate boxes must have active motion or confirmed velocity.
  4. Multi-Factor Identity & Pattern Recognition:
     - Distinguishes similar-looking fish by combining:
         * 128-dim 3D HSV color-pattern histogram
         * Relative body size / area
         * Body aspect ratio (elongation vs roundness)
         * Hue tone & saturation profile
  5. Minimalist UI & 30 FPS:
     - Clean glowing dot on each fish centroid.
     - 'Fish 1', 'Fish 2', etc. floating neatly above each dot.
     - Breadcrumb trail of where the fish swam.
     - Smooth threaded 30 FPS video feed.

Controls:
  R       = Recalibrate fish population
  1 to 9  = Manually lock the exact number of fish on the fly
  Space   = Finalize calibration immediately
  S       = Save tracking log to fish_state.json
  Q       = Quit
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

MODEL_PATH          = "aquarium_best.pt"
CALIB_DURATION      = 5.0    # seconds to observe motion and count population
DEFAULT_FALLBACK_N  = 6      # fallback if no fish moving during calibration

# --- Tracking Parameters ---
SPATIAL_GATE_DIST   = 160    # max px distance for frame-to-frame continuity
SMOOTH_ALPHA        = 0.70   # position smoothing (0.7 new + 0.3 old)
COAST_FRAMES        = 10     # keep dot at last position if temporarily occluded

# --- Stationary Stone Rejection Parameters ---
STATIC_MAX_DISP     = 3.5    # px: max net displacement to classify as stationary
STATIC_HOLD_FRAMES  = 14     # frames (~0.45s) of zero motion before blacklisting stone
STATIC_PURGE_DIST   = 35.0   # px radius: ignore all future detections around blacklisted stone

# --- Aesthetics ---
DOT_RADIUS          = 5
TRAIL_LENGTH        = 20
FONT                = cv2.FONT_HERSHEY_SIMPLEX

# Distinct palette for fish dots (BGR)
PALETTE = [
    (0,   215, 255),   # Fish 1: Amber / Gold
    (50,  235,  70),   # Fish 2: Neon Green
    (255, 100, 210),   # Fish 3: Magenta / Pink
    (255, 185,  40),   # Fish 4: Sky Blue
    (30,  130, 255),   # Fish 5: Orange
    (220, 240, 255),   # Fish 6: Pearl White
    (190,  70, 255),   # Fish 7: Violet
    (40,  255, 220),   # Fish 8: Cyan
    (250, 210, 100),   # Fish 9: Turquoise
]

def get_palette_color(index: int):
    return PALETTE[index % len(PALETTE)]

# =============================================================================
#   THREADED CAMERA STREAM (Rock-Solid 30 FPS)
# =============================================================================

class VideoStream:
    """Non-blocking background camera reader for steady 30 FPS."""
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
#   STATIONARY STONE BLACKLIST ENGINE
# =============================================================================

class StationaryStoneFilter:
    """
    Monitors all detection centroids over consecutive frames.
    If an object remains virtually motionless (< 3.5 px displacement over 14 frames),
    it is permanently blacklisted as a stone / decoration and suppressed!
    """
    def __init__(self):
        self.probes = []           # tracked candidate locations
        self.blacklist = []        # list of (x, y, radius) permanent stone locations

    def is_stone_location(self, cx: float, cy: float) -> bool:
        """Returns True if (cx, cy) is at or near a blacklisted stationary stone."""
        for bx, by, r in self.blacklist:
            if math.hypot(cx - bx, cy - by) < r:
                return True
        return False

    def filter_and_update(self, raw_candidates):
        """
        raw_candidates: list of dicts with 'cx', 'cy', 'bbox', 'sig'
        Returns: list of valid MOVING fish candidates (stationary stones purged).
        """
        valid_candidates = []
        matched_probes = set()

        for cand in raw_candidates:
            cx, cy = cand['cx'], cand['cy']

            # If already near a known stationary stone, instantly discard!
            if self.is_stone_location(cx, cy):
                continue

            # Associate with existing probe
            best_p = None
            best_d = 28.0
            for i, p in enumerate(self.probes):
                if i in matched_probes:
                    continue
                d = math.hypot(cx - p['cx'], cy - p['cy'])
                if d < best_d:
                    best_d = d
                    best_p = i

            if best_p is not None:
                matched_probes.add(best_p)
                p = self.probes[best_p]
                orig_x, orig_y = p['history'][0]
                disp = math.hypot(cx - orig_x, cy - orig_y)
                p['history'].append((cx, cy))
                p['cx'], p['cy'] = cx, cy

                # Check if it has failed to move over enough frames
                if len(p['history']) >= STATIC_HOLD_FRAMES and disp < STATIC_MAX_DISP:
                    # Stationary stone confirmed!
                    self.blacklist.append((cx, cy, STATIC_PURGE_DIST))
                    # print(f"  [STONE BLACKLISTED] Fixed stone at ({cx:.0f}, {cy:.0f}) - Net move only {disp:.1f}px")
                else:
                    valid_candidates.append(cand)
            else:
                # New probe
                h = deque(maxlen=STATIC_HOLD_FRAMES * 2)
                h.append((cx, cy))
                self.probes.append({'history': h, 'cx': cx, 'cy': cy})
                valid_candidates.append(cand)

        # Cleanup old probes that disappeared
        if len(self.probes) > 50:
            self.probes = self.probes[-30:]

        return valid_candidates

# =============================================================================
#   WHITE-STONE OPTICAL FILTER (Bottom Gravel Bed)
# =============================================================================

def is_white_stone_optical(crop_bgr, y2, total_h) -> bool:
    """
    White pebbles sit at the bottom gravel bed (y > 70% of tank height),
    with bright washed-out RGB and low color saturation.
    """
    if y2 < (0.70 * total_h):
        return False

    b_mean = float(np.mean(crop_bgr[:, :, 0]))
    g_mean = float(np.mean(crop_bgr[:, :, 1]))
    r_mean = float(np.mean(crop_bgr[:, :, 2]))

    # White stone signature: all RGB channels are bright and balanced
    is_bright_white = (b_mean > 110.0 and g_mean > 115.0 and r_mean > 115.0)

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    mean_s = float(np.mean(hsv[:, :, 1]))

    return is_bright_white and (mean_s < 38.0)

# =============================================================================
#   MULTI-FACTOR FISH SIGNATURE (Color Pattern + Shape + Size)
# =============================================================================

def extract_multi_factor_signature(crop_bgr):
    """
    Extracts a multi-factor feature representation to distinguish fish:
      1. 128-dim 3D HSV color histogram (distribution across body).
      2. Normalized area & bounding box aspect ratio.
      3. Mean Hue & Saturation.
    """
    if crop_bgr is None or crop_bgr.size == 0 or crop_bgr.shape[0] < 4 or crop_bgr.shape[1] < 4:
        return None

    h, w = crop_bgr.shape[:2]
    area = float(h * w)
    aspect = float(w / max(1, h))

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    # 8 Hue x 4 Saturation x 4 Value = 128 bins
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256])
    vec = hist.flatten().astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 1e-6:
        vec /= norm

    mean_h = float(np.mean(hsv[:, :, 0]))
    mean_s = float(np.mean(hsv[:, :, 1]))

    return {
        'pattern_vec': vec,
        'area'       : area,
        'aspect'     : aspect,
        'mean_h'     : mean_h,
        'mean_s'     : mean_s
    }

def compute_similarity(sig_a, sig_b) -> float:
    """
    Computes a comprehensive similarity score in [0.0, 1.0]
    combining pattern, size, aspect ratio, and hue.
    """
    if sig_a is None or sig_b is None:
        return 0.0

    # 1. Pattern cosine similarity
    pat_sim = max(0.0, float(np.dot(sig_a['pattern_vec'], sig_b['pattern_vec'])))

    # 2. Size similarity (smaller / larger)
    min_a, max_a = min(sig_a['area'], sig_b['area']), max(sig_a['area'], sig_b['area'])
    size_sim = min_a / max(1.0, max_a)

    # 3. Aspect ratio similarity
    min_asp, max_asp = min(sig_a['aspect'], sig_b['aspect']), max(sig_a['aspect'], sig_b['aspect'])
    asp_sim = min_asp / max(0.01, max_asp)

    # 4. Hue tone similarity
    hue_diff = min(abs(sig_a['mean_h'] - sig_b['mean_h']), 180 - abs(sig_a['mean_h'] - sig_b['mean_h']))
    hue_sim = max(0.0, 1.0 - (hue_diff / 60.0))

    return 0.45 * pat_sim + 0.25 * size_sim + 0.15 * asp_sim + 0.15 * hue_sim

# =============================================================================
#   TRACKED FISH ENTITY
# =============================================================================

class TrackedFish:
    def __init__(self, fid: int, centroid, signature):
        self.id            = fid
        self.name          = f"Fish {fid}"
        self.cx            = float(centroid[0])
        self.cy            = float(centroid[1])
        self.signature     = signature
        self.color         = get_palette_color(fid - 1)

        now = time.time()
        self.first_seen    = now
        self.last_seen     = now
        self.total_sec     = 0.0
        self.is_visible    = True
        self.missed_frames = 0

        self.trail         = deque(maxlen=TRAIL_LENGTH)
        self.trail.append((self.cx, self.cy))

    def update_position(self, new_cx, new_cy, new_signature, now):
        self.cx = SMOOTH_ALPHA * float(new_cx) + (1.0 - SMOOTH_ALPHA) * self.cx
        self.cy = SMOOTH_ALPHA * float(new_cy) + (1.0 - SMOOTH_ALPHA) * self.cy

        if new_signature is not None and self.signature is not None:
            self.signature['pattern_vec'] = (0.85 * self.signature['pattern_vec'] +
                                             0.15 * new_signature['pattern_vec'])
            norm = np.linalg.norm(self.signature['pattern_vec'])
            if norm > 1e-6:
                self.signature['pattern_vec'] /= norm

            self.signature['area'] = 0.85 * self.signature['area'] + 0.15 * new_signature['area']
            self.signature['aspect'] = 0.85 * self.signature['aspect'] + 0.15 * new_signature['aspect']
            self.signature['mean_h'] = 0.85 * self.signature['mean_h'] + 0.15 * new_signature['mean_h']

        self.last_seen     = now
        self.is_visible    = True
        self.missed_frames = 0
        self.trail.append((self.cx, self.cy))

    def mark_missed(self):
        self.missed_frames += 1
        if self.missed_frames > COAST_FRAMES:
            self.is_visible = False

    def to_dict(self):
        return {
            "fish_id"      : self.id,
            "label"        : self.name,
            "first_seen"   : datetime.fromtimestamp(self.first_seen).isoformat(),
            "last_seen"    : datetime.fromtimestamp(self.last_seen).isoformat(),
            "total_seconds": round(self.total_sec, 2),
            "is_visible"   : self.is_visible
        }

# =============================================================================
#   BIPARTITE MATCHING HELPER
# =============================================================================

def greedy_bipartite_match(cost_matrix, max_cost=float('inf')):
    cost = cost_matrix.copy()
    matches = []
    while True:
        min_val = np.min(cost)
        if min_val >= max_cost or np.isinf(min_val):
            break
        r, c = np.unravel_index(np.argmin(cost), cost.shape)
        matches.append((int(r), int(c)))
        cost[r, :] = np.inf
        cost[:, c] = np.inf
    return matches

# =============================================================================
#   FISH TRACKING & POPULATION ENGINE
# =============================================================================

class SmartEyesEngine:
    def __init__(self, model_path=MODEL_PATH):
        self.fishes = []                  # list of TrackedFish
        self.population_locked = False
        self.target_population = DEFAULT_FALLBACK_N

        # MOG2 background subtractor
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=32, detectShadows=False)

        # Permanent stone rejector
        self.stone_filter = StationaryStoneFilter()

        # Calibration state
        self.calibrating = True
        self.calib_start = time.time()
        self.calib_counts = deque(maxlen=200)

        # YOLO Detector
        self.detector = None
        if HAS_YOLO and os.path.exists(model_path):
            try:
                self.detector = YOLO(model_path)
                print(f"[INFO] Detector loaded: {model_path}")
            except Exception as e:
                print(f"[WARN] Detector load failed: {e}")

    def start_calibration(self):
        self.calibrating = True
        self.calib_start = time.time()
        self.calib_counts.clear()
        self.fishes.clear()
        self.population_locked = False
        print("[CALIB] Calibrating fish population (ignoring stationary stones)...")

    def lock_population(self, count: int):
        self.target_population = max(1, min(9, count))
        self.calibrating = False
        self.population_locked = True
        print(f"[LOCKED] Population locked to: {self.target_population} Fish")

    def _extract_candidates(self, frame_bgr):
        fh, fw = frame_bgr.shape[:2]

        # 1. Background subtraction for motion
        fg_mask = self.bg_sub.apply(frame_bgr)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_clean = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_clean = cv2.morphologyEx(fg_clean, cv2.MORPH_DILATE, kernel, iterations=2)

        # 2. YOLO detections
        yolo_boxes = []
        if self.detector is not None:
            try:
                results = self.detector.predict(
                    source=frame_bgr,
                    conf=0.15,
                    iou=0.40,
                    imgsz=320,
                    classes=[0] if "aquarium" in MODEL_PATH.lower() else None,
                    verbose=False
                )
                if results and results[0].boxes is not None:
                    for b in results[0].boxes:
                        x1, y1, x2, y2 = [int(v) for v in b.xyxy[0]]
                        if x1 > 25 and y1 > 20 and x2 < fw - 25 and y2 < fh - 20:
                            yolo_boxes.append((x1, y1, x2, y2))
            except Exception:
                pass

        # 3. Motion contours
        contours, _ = cv2.findContours(fg_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_boxes = []
        for c in contours:
            area = cv2.contourArea(c)
            if 300 < area < 40000:
                mx, my, mw, mh = cv2.boundingRect(c)
                if mx > 25 and my > 20 and (mx + mw) < fw - 25 and (my + mh) < fh - 20:
                    motion_boxes.append((mx, my, mx + mw, my + mh))

        # Merge boxes with motion confirmation
        merged_boxes = []
        for yb in yolo_boxes:
            x1, y1, x2, y2 = yb
            crop = frame_bgr[y1:y2, x1:x2]

            # REJECT 1: Bottom gravel white stone optical check
            if is_white_stone_optical(crop, y2, fh):
                continue

            # REJECT 2: Real motion check inside box (eliminates stationary stones)
            mask_crop = fg_clean[y1:y2, x1:x2]
            motion_density = np.mean(mask_crop > 0)
            if motion_density > 0.04:
                merged_boxes.append(yb)

        # Include moving fish detected by motion even if YOLO missed them
        for mb in motion_boxes:
            mcx = (mb[0] + mb[2]) / 2.0
            mcy = (mb[1] + mb[3]) / 2.0
            overlap = any(math.hypot(mcx - ((yb[0]+yb[2])/2.0), mcy - ((yb[1]+yb[3])/2.0)) < 40 for yb in merged_boxes)
            if not overlap:
                crop = frame_bgr[mb[1]:mb[3], mb[0]:mb[2]]
                if not is_white_stone_optical(crop, mb[3], fh):
                    merged_boxes.append(mb)

        # Build candidate structures
        raw_candidates = []
        for b in merged_boxes:
            x1, y1, x2, y2 = b
            crop = frame_bgr[y1:y2, x1:x2]
            sig = extract_multi_factor_signature(crop)
            if sig is not None:
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                raw_candidates.append({
                    'cx': cx,
                    'cy': cy,
                    'bbox': b,
                    'sig': sig
                })

        # REJECT 3: Stationary Stone Filter (Blacklists objects with zero net displacement)
        verified_candidates = self.stone_filter.filter_and_update(raw_candidates)

        return verified_candidates

    def update(self, frame_bgr, now):
        candidates = self._extract_candidates(frame_bgr)

        # -------------------------------------------------------------
        # CALIBRATION PHASE: Count Moving Fish Only
        # -------------------------------------------------------------
        if self.calibrating:
            self.calib_counts.append(len(candidates))
            elapsed = now - self.calib_start

            if len(candidates) > len(self.fishes):
                self.fishes = [
                    TrackedFish(i + 1, (c['cx'], c['cy']), c['sig'])
                    for i, c in enumerate(candidates)
                ]

            if elapsed >= CALIB_DURATION:
                valid_counts = [c for c in self.calib_counts if c > 0]
                if valid_counts:
                    determined_count = int(np.percentile(valid_counts, 75))
                    determined_count = max(1, min(9, determined_count))
                else:
                    determined_count = DEFAULT_FALLBACK_N

                self.lock_population(determined_count)
                self.fishes = self.fishes[:self.target_population]
                while len(self.fishes) < self.target_population:
                    idx = len(self.fishes) + 1
                    dummy_sig = self.fishes[0].signature if self.fishes else None
                    self.fishes.append(TrackedFish(idx, (DISPLAY_W / 2, DISPLAY_H / 2), dummy_sig))

            return self.fishes

        # -------------------------------------------------------------
        # TRACKING & RE-ID PHASE: Strictly Fixed N Fish
        # -------------------------------------------------------------
        if not self.fishes:
            return []

        if not candidates:
            for f in self.fishes:
                f.mark_missed()
            return self.fishes

        # Cost matrix: Rows = Detections, Columns = Registered Fish
        num_dets = len(candidates)
        num_fish = len(self.fishes)
        cost_matrix = np.full((num_dets, num_fish), float('inf'), dtype=np.float32)

        for i, cand in enumerate(candidates):
            for j, fish in enumerate(self.fishes):
                sim = compute_similarity(cand['sig'], fish.signature)
                dist = math.hypot(cand['cx'] - fish.cx, cand['cy'] - fish.cy)

                if fish.is_visible:
                    spatial_cost = min(1.0, dist / SPATIAL_GATE_DIST)
                    cost = 0.50 * spatial_cost + 0.50 * (1.0 - sim)
                else:
                    cost = 1.0 - sim

                cost_matrix[i, j] = cost

        matches = greedy_bipartite_match(cost_matrix, max_cost=0.75)
        matched_fish_indices = set()

        for det_idx, fish_idx in matches:
            matched_fish_indices.add(fish_idx)
            cand = candidates[det_idx]
            self.fishes[fish_idx].update_position(cand['cx'], cand['cy'], cand['sig'], now)

        for j, fish in enumerate(self.fishes):
            if j not in matched_fish_indices:
                fish.mark_missed()

        return self.fishes

# =============================================================================
#   MINIMALIST RENDERING
# =============================================================================

def render_fish(frame, fish: TrackedFish):
    ix = int(round(fish.cx))
    iy = int(round(fish.cy))
    col = fish.color

    # 1. Subtle breadcrumb trail
    if len(fish.trail) > 1:
        pts = list(fish.trail)
        for i in range(1, len(pts)):
            alpha = (i / len(pts))
            r = max(1, int(3 * alpha))
            tx, ty = int(pts[i][0]), int(pts[i][1])
            cv2.circle(frame, (tx, ty), r, col, -1, cv2.LINE_AA)

    # 2. Glowing center dot
    cv2.circle(frame, (ix, iy), DOT_RADIUS + 3, (15, 15, 15), -1, cv2.LINE_AA)
    cv2.circle(frame, (ix, iy), DOT_RADIUS, col, -1, cv2.LINE_AA)
    cv2.circle(frame, (ix, iy), 2, (255, 255, 255), -1, cv2.LINE_AA)

    # 3. Clean floating label: 'Fish 1', 'Fish 2'
    label = fish.name
    tx, ty = ix + 9, iy - 6
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.44, 1)

    x1, y1 = tx - 3, ty - th - 2
    x2, y2 = tx + tw + 3, ty + 3
    if 0 <= x1 and x2 < frame.shape[1] and 0 <= y1 and y2 < frame.shape[0]:
        sub = frame[y1:y2, x1:x2]
        dark = np.zeros_like(sub)
        cv2.addWeighted(sub, 0.35, dark, 0.65, 0, sub)
        frame[y1:y2, x1:x2] = sub

    cv2.putText(frame, label, (tx, ty), FONT, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

def render_calibration_hud(frame, elapsed, total_time, current_detected):
    h, w = frame.shape[:2]
    pw, ph = 520, 60
    x1, y1 = (w - pw) // 2, 20
    x2, y2 = x1 + pw, y1 + ph

    sub = frame[y1:y2, x1:x2]
    dark = np.zeros_like(sub)
    cv2.addWeighted(sub, 0.20, dark, 0.80, 0, sub)
    frame[y1:y2, x1:x2] = sub
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 215, 255), 2)

    progress = min(1.0, elapsed / max(1e-5, total_time))
    txt = f"CALIBRATING TANK POPULATION... [{progress * 100:.0f}%]"
    sub_txt = f"Filtering stationary stones  |  Moving fish seen: {current_detected}  |  [Space]: Lock Now"

    cv2.putText(frame, txt, (x1 + 14, y1 + 24), FONT, 0.46, (0, 215, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, sub_txt, (x1 + 14, y1 + 46), FONT, 0.36, (220, 220, 220), 1, cv2.LINE_AA)

    bx1 = x1 + 2
    bx2 = x1 + int((pw - 4) * progress)
    cv2.line(frame, (bx1, y2 - 2), (bx2, y2 - 2), (0, 215, 255), 3)

def render_top_banner(frame, fps: float, visible_count: int, total_pop: int):
    txt = f"SmartEyes  |  {fps:.1f} FPS  |  Tank Population: {total_pop} Fish ({visible_count} In View)  |  [R] Re-calibrate  [1-9] Override"
    (tw, th), _ = cv2.getTextSize(txt, FONT, 0.40, 1)

    x1, y1 = 12, 10
    x2, y2 = 12 + tw + 14, 10 + th + 12

    sub = frame[y1:y2, x1:x2]
    dark = np.zeros_like(sub)
    cv2.addWeighted(sub, 0.25, dark, 0.75, 0, sub)
    frame[y1:y2, x1:x2] = sub
    cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 60, 60), 1)

    cv2.putText(frame, txt, (x1 + 7, y1 + th + 5), FONT, 0.40, (230, 230, 230), 1, cv2.LINE_AA)

# =============================================================================
#   MAIN
# =============================================================================

def run():
    print("=" * 65)
    print("   SmartEyes Computer Vision Fish Monitor (v7 Anti-Stone Engine) ")
    print("=" * 65)

    vs = VideoStream(src=CAMERA_INDEX, width=CAPTURE_WIDTH, height=CAPTURE_HEIGHT)
    time.sleep(0.5)

    if not vs.is_opened():
        sys.exit(f"[ERROR] Could not open camera {CAMERA_INDEX}.")

    engine = SmartEyesEngine()
    engine.start_calibration()

    cv2.namedWindow("SmartEyes Fish Monitor", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("SmartEyes Fish Monitor", DISPLAY_W, DISPLAY_H)

    prev_time = time.time()
    ema_fps = 30.0

    print("\n[INFO] Starting camera feed at 30 FPS.")
    print("[KEYS] R: Re-calibrate | 1..9: Set exact fish count | Space: Lock Count | Q: Quit\n")

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

            fishes = engine.update(disp, now)

            visible_count = 0
            for f in fishes:
                if f.is_visible:
                    visible_count += 1
                    f.total_sec += dt
                    render_fish(disp, f)

            if engine.calibrating:
                elapsed = now - engine.calib_start
                render_calibration_hud(disp, elapsed, CALIB_DURATION, len(fishes))
            else:
                render_top_banner(disp, ema_fps, visible_count, engine.target_population)

            cv2.imshow("SmartEyes Fish Monitor", disp)

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q') or k == ord('Q'):
                break
            elif k == ord('r') or k == ord('R'):
                engine.start_calibration()
            elif k == 32 or k == 13:
                if engine.calibrating:
                    engine.lock_population(len(fishes) if fishes else DEFAULT_FALLBACK_N)
            elif ord('1') <= k <= ord('9'):
                engine.lock_population(k - ord('0'))

    finally:
        vs.stop()
        cv2.destroyAllWindows()

    print("\n" + "=" * 50)
    print("           MONITORING SESSION SUMMARY             ")
    print("=" * 50)
    summary = {str(f.id): f.to_dict() for f in engine.fishes}
    print(json.dumps(summary, indent=2))
    print(f"\n[DONE] Monitored {len(summary)} fish.")

if __name__ == "__main__":
    run()
