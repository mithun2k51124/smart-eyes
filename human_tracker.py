# -*- coding: utf-8 -*-
"""
SmartEyes - Human Perception, Multi-Person Re-ID, Clothing & Attendance System (v1.0)
=====================================================================================
Dedicated Subsystem for Human & Person Intelligence:
  - DOES NOT interfere with the SmartEyes Fish Tracking subsystem.
  - Multi-Person Full-Body Detection with AMD Radeon RX 6600 DirectML GPU Acceleration.
  - Deep Facial Recognition (YuNet Face Detector + SFace 128-D Embeddings).
  - Overhead & Top-Down Camera Friendly:
    * Extracts Upper Body Clothing color & texture (e.g. "Black Hoodie [Solid]").
    * Extracts Lower Body Clothing color (e.g. "Blue Denim", "Khaki Pants").
    * Detects Bagging & Backpacks carried by each person.
    * Uses Clothing & Bagging Re-ID signature when faces are hidden from overhead angles!
  - Automated Attendance System:
    * Automatic Check-In / Clock-In timestamping.
    * Real-time Duration Present tracking.
    * Exports attendance records to 'attendance_log.json' and 'attendance_log.csv'.

Controls:
  Q / ESC : Quit
  S       : Save / Export Attendance Ledger (JSON & CSV)
  R       : Reset Tracking Gallery
  A       : Print Attendance Summary to Console
"""

import time
import json
import csv
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

MODEL_PERSON_ONNX   = "yolov8n.onnx"
MODEL_PERSON_PT     = "yolov8n.pt"

MODEL_YUNET         = os.path.join("weights", "face_detection_yunet.onnx")
MODEL_SFACE         = os.path.join("weights", "face_recognition_sface.onnx")

ATTENDANCE_JSON     = "attendance_log.json"
ATTENDANCE_CSV      = "attendance_log.csv"

INFER_SIZE          = 416
CONF_THRESH_PERSON  = 0.35
CONF_THRESH_BAG     = 0.25
IOU_THRESH          = 0.45

# --- Re-ID & Spatial Continuity ---
SPATIAL_GATE_PX     = 220    # Max px search radius for human movement
COAST_FRAMES        = 50     # Keep identity alive for ~1.7s when momentarily occluded
SMOOTH_ALPHA        = 0.70
FACE_SIM_THRESH     = 0.42   # SFace cosine similarity threshold
CLOTH_SIM_THRESH    = 0.50   # Clothing appearance similarity threshold

FONT = cv2.FONT_HERSHEY_SIMPLEX

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
#   AMD GPU DIRECTML HUMAN & BAG DETECTOR
# =============================================================================

class DirectMLHumanDetector:
    """Runs YOLOv8 person & backpack detection accelerated on AMD Radeon RX 6600."""
    def __init__(self, onnx_path=MODEL_PERSON_ONNX, imgsz=INFER_SIZE):
        self.imgsz = imgsz
        providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        active_provider = self.session.get_providers()[0]
        self.is_gpu = "Dml" in active_provider
        self.input_name = self.session.get_inputs()[0].name

        device_name = "AMD Radeon RX 6600 (DirectX 12 DirectML)" if self.is_gpu else active_provider
        print(f"[HUMAN DETECTOR] Hardware Acceleration: {device_name}")

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

        scale_x = orig_w / float(self.imgsz)
        scale_y = orig_h / float(self.imgsz)

        person_boxes, person_scores = [], []
        bag_boxes, bag_scores = [], []

        for row in outputs:
            cx, cy, w, h = row[0], row[1], row[2], row[3]
            class_scores = row[4:]

            # Class 0 = Person
            person_score = float(class_scores[0])
            if person_score >= CONF_THRESH_PERSON:
                x1 = int((cx - w / 2.0) * scale_x)
                y1 = int((cy - h / 2.0) * scale_y)
                bw = int(w * scale_x)
                bh = int(h * scale_y)
                person_boxes.append([x1, y1, bw, bh])
                person_scores.append(person_score)

            # Bag classes: 24 (backpack), 26 (handbag), 28 (suitcase)
            bag_score = max(float(class_scores[24]), float(class_scores[26]), float(class_scores[28]))
            if bag_score >= CONF_THRESH_BAG:
                x1 = int((cx - w / 2.0) * scale_x)
                y1 = int((cy - h / 2.0) * scale_y)
                bw = int(w * scale_x)
                bh = int(h * scale_y)
                bag_boxes.append([x1, y1, bw, bh])
                bag_scores.append(bag_score)

        # NMS for persons
        final_persons = []
        if person_boxes:
            indices = cv2.dnn.NMSBoxes(person_boxes, person_scores, CONF_THRESH_PERSON, IOU_THRESH)
            if len(indices) > 0:
                for idx in indices.flatten():
                    x, y, bw, bh = person_boxes[idx]
                    conf = person_scores[idx]
                    x1, y1 = max(0, x), max(0, y)
                    x2, y2 = min(orig_w, x + bw), min(orig_h, y + bh)
                    if (x2 - x1) > 25 and (y2 - y1) > 50:
                        final_persons.append((x1, y1, x2, y2, conf))

        # NMS for bags
        final_bags = []
        if bag_boxes:
            b_indices = cv2.dnn.NMSBoxes(bag_boxes, bag_scores, CONF_THRESH_BAG, IOU_THRESH)
            if len(b_indices) > 0:
                for idx in b_indices.flatten():
                    x, y, bw, bh = bag_boxes[idx]
                    conf = bag_scores[idx]
                    x1, y1 = max(0, x), max(0, y)
                    x2, y2 = min(orig_w, x + bw), min(orig_h, y + bh)
                    final_bags.append((x1, y1, x2, y2, conf))

        return final_persons, final_bags

# =============================================================================
#   FACIAL RECOGNITION (YUNET + SFACE)
# =============================================================================

class FaceBiometrics:
    def __init__(self, yunet_path=MODEL_YUNET, sface_path=MODEL_SFACE):
        self.available = False
        if os.path.exists(yunet_path) and os.path.exists(sface_path):
            try:
                self.detector = cv2.FaceDetectorYN.create(yunet_path, "", (320, 320), 0.6, 0.3, 500)
                self.recognizer = cv2.FaceRecognizerSF.create(sface_path, "")
                self.available = True
                print("[FACE BIOMETRICS] YuNet + SFace initialized successfully.")
            except Exception as e:
                print(f"[FACE BIOMETRICS] Init warning: {e}")

    def extract_face_embedding(self, person_crop):
        """Detects face inside person crop and extracts 128-D SFace embedding."""
        if not self.available or person_crop is None or person_crop.size == 0:
            return None, None

        ch, cw = person_crop.shape[:2]
        if ch < 40 or cw < 30:
            return None, None

        # Focus face detector on upper half of person
        head_crop = person_crop[0:int(ch * 0.55), :]
        hh, hw = head_crop.shape[:2]
        if hh < 20 or hw < 20:
            return None, None

        self.detector.setInputSize((hw, hh))
        _, faces = self.detector.detect(head_crop)

        if faces is None or len(faces) == 0:
            return None, None

        # Select highest confidence face
        best_face = faces[0]
        aligned_face = self.recognizer.alignCrop(head_crop, best_face)
        feat = self.recognizer.feature(aligned_face)

        # Normalize feature vector
        norm = np.linalg.norm(feat)
        if norm > 1e-6:
            feat /= norm

        face_bbox = [int(best_face[0]), int(best_face[1]), int(best_face[2]), int(best_face[3])]
        return feat, face_bbox

    def compute_face_similarity(self, feat_a, feat_b) -> float:
        if feat_a is None or feat_b is None:
            return 0.0
        # Cosine similarity
        return float(np.dot(feat_a.flatten(), feat_b.flatten()))

# =============================================================================
#   CLOTHING, APPAREL & BAG RECOGNITION (TOP-DOWN & OVERHEAD FRIENDLY)
# =============================================================================

def classify_clothing_color(crop_bgr):
    """Identifies color tone of clothing segment."""
    if crop_bgr is None or crop_bgr.size == 0:
        return "Unknown", (180, 180, 180)

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h = float(np.mean(hsv[:, :, 0]))
    s = float(np.mean(hsv[:, :, 1]))
    v = float(np.mean(hsv[:, :, 2]))

    if v < 55:
        return "Black/Dark", (40, 40, 40)
    elif s < 30 and v > 175:
        return "White/Light", (240, 240, 240)
    elif s < 35:
        return "Grey", (150, 150, 150)
    elif (h <= 10 or h >= 170) and s > 50:
        return "Red/Maroon", (30, 30, 210)
    elif 11 <= h <= 25 and s > 55:
        return "Orange", (20, 130, 240)
    elif 26 <= h <= 36 and s > 50:
        return "Yellow", (30, 220, 240)
    elif 37 <= h <= 85 and s > 40:
        return "Green/Olive", (50, 190, 80)
    elif 90 <= h <= 130 and s > 40:
        return "Blue/Navy", (220, 100, 30)
    elif 131 <= h <= 165 and s > 40:
        return "Purple/Violet", (200, 50, 200)

    return "Neutral", (180, 180, 180)

def analyze_clothing_and_bag(person_crop, person_bbox, detected_bags):
    """
    Analyzes upper & lower body clothing, pattern, and bag carrying status.
    Provides robust top-down appearance descriptors when face is hidden.
    """
    ch, cw = person_crop.shape[:2]
    px1, py1, px2, py2 = person_bbox

    # 1. Upper Body (Torso / Top)
    upper_y1 = int(ch * 0.14)
    upper_y2 = int(ch * 0.54)
    upper_x1 = int(cw * 0.15)
    upper_x2 = int(cw * 0.85)
    upper_crop = person_crop[upper_y1:upper_y2, upper_x1:upper_x2]
    upper_color, upper_bgr = classify_clothing_color(upper_crop)

    # Upper texture/pattern
    upper_gray = cv2.cvtColor(upper_crop, cv2.COLOR_BGR2GRAY) if upper_crop.size > 0 else np.zeros((1, 1), dtype=np.uint8)
    lap_var = cv2.Laplacian(upper_gray, cv2.CV_64F).var() if upper_gray.size > 0 else 0.0
    pattern = "Textured/Patterned" if lap_var > 120.0 else "Solid"

    # 2. Lower Body (Pants / Skirt)
    lower_y1 = int(ch * 0.55)
    lower_y2 = int(ch * 0.94)
    lower_crop = person_crop[lower_y1:lower_y2, upper_x1:upper_x2]
    lower_color, lower_bgr = classify_clothing_color(lower_crop)

    # 3. Bag / Backpack Detection (spatial overlap)
    has_bag = False
    bag_label = "None"
    for bx1, by1, bx2, by2, bconf in detected_bags:
        bcx = (bx1 + bx2) / 2.0
        bcy = (by1 + by2) / 2.0
        if (px1 <= bcx <= px2) and (py1 <= bcy <= py2):
            has_bag = True
            bag_label = "Backpack"
            break

    # 4. Top-Down Appearance Re-ID Signature (3D HSV Histograms of upper + lower)
    hsv = cv2.cvtColor(person_crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256])
    vec = hist.flatten().astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 1e-6:
        vec /= norm

    return {
        "upper_color": upper_color,
        "upper_bgr"  : upper_bgr,
        "pattern"    : pattern,
        "lower_color": lower_color,
        "lower_bgr"  : lower_bgr,
        "has_bag"    : has_bag,
        "bag_label"  : bag_label,
        "signature"  : vec
    }

# =============================================================================
#   TRACKED PERSON ENTITY & ATTENDANCE RECORD
# =============================================================================

class TrackedPerson:
    def __init__(self, pid: int, bbox, clothing_info, face_embedding, conf: float):
        self.id               = pid
        self.name             = f"Person #{pid}"
        self.bbox             = [float(v) for v in bbox]
        self.conf             = conf
        self.clothing         = clothing_info

        # Face Biometrics
        self.face_embedding   = face_embedding
        self.face_count       = 1 if face_embedding is not None else 0
        self.face_gallery     = deque(maxlen=4)
        if face_embedding is not None:
            self.face_gallery.append(face_embedding)

        # Multi-view Clothing Gallery (for overhead tracking)
        self.clothing_gallery = deque(maxlen=6)
        if clothing_info and clothing_info["signature"] is not None:
            self.clothing_gallery.append(clothing_info["signature"])

        # Timestamps & Attendance
        now = time.time()
        self.first_seen       = now
        self.last_seen        = now
        self.clock_in_str     = datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S")
        self.total_seconds    = 0.0
        self.is_active        = True
        self.missed_frames    = 0

        # Motion & Centroid
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        self.positions        = deque(maxlen=60)
        self.positions.append((cx, cy, now))
        self.speed_px_s       = 0.0

    @property
    def centroid(self):
        return ((self.bbox[0] + self.bbox[2]) / 2.0, (self.bbox[1] + self.bbox[3]) / 2.0)

    def match_similarity(self, new_face, new_cloth_sig, biometrics: FaceBiometrics) -> float:
        """Fused multimodal biometric match: Face Similarity + Clothing Signature."""
        face_sim = 0.0
        if new_face is not None and self.face_gallery:
            face_sim = max(biometrics.compute_face_similarity(new_face, f) for f in self.face_gallery)

        cloth_sim = 0.0
        if new_cloth_sig is not None and self.clothing_gallery:
            cloth_sim = max(float(np.dot(new_cloth_sig, c)) for c in self.clothing_gallery)

        # If face is recognized, weight face heavily (75%)
        if face_sim > 0.40:
            return 0.75 * face_sim + 0.25 * cloth_sim
        # If face is hidden (overhead camera), rely on clothing signature
        return cloth_sim

    def update(self, new_bbox, new_clothing, new_face, new_conf, now, dt):
        old_cx, old_cy = self.centroid
        new_cx = (new_bbox[0] + new_bbox[2]) / 2.0
        new_cy = (new_bbox[1] + new_bbox[3]) / 2.0

        # Velocity
        dist = math.hypot(new_cx - old_cx, new_cy - old_cy)
        if dt > 0:
            self.speed_px_s = dist / dt

        # Bounding Box Smoothing
        for i in range(4):
            self.bbox[i] = SMOOTH_ALPHA * float(new_bbox[i]) + (1.0 - SMOOTH_ALPHA) * self.bbox[i]

        self.positions.append((new_cx, new_cy, now))
        self.conf = new_conf
        self.clothing = new_clothing
        self.last_seen = now
        self.is_active = True
        self.missed_frames = 0
        self.total_seconds += dt

        # Update Face Embedding
        if new_face is not None:
            self.face_embedding = new_face
            self.face_gallery.append(new_face)
            self.face_count += 1

        # Update Clothing Signature
        if new_clothing and new_clothing["signature"] is not None:
            self.clothing_gallery.append(new_clothing["signature"])

    def mark_missed(self):
        self.missed_frames += 1
        if self.missed_frames > COAST_FRAMES:
            self.is_active = False

    def to_attendance_dict(self):
        duration_m = int(self.total_seconds // 60)
        duration_s = int(self.total_seconds % 60)
        return {
            "person_id"        : self.id,
            "name"             : self.name,
            "clock_in"         : self.clock_in_str,
            "last_seen"        : datetime.fromtimestamp(self.last_seen).strftime("%H:%M:%S"),
            "duration_str"     : f"{duration_m}m {duration_s}s",
            "duration_seconds" : round(self.total_seconds, 1),
            "face_recognized"  : self.face_count > 0,
            "upper_clothing"   : f"{self.clothing['upper_color']} [{self.clothing['pattern']}]",
            "lower_clothing"   : self.clothing['lower_color'],
            "bag_status"       : self.clothing['bag_label'],
            "status"           : "Present" if self.is_active else "Left"
        }

# =============================================================================
#   HUMAN TRACKING & ATTENDANCE ENGINE
# =============================================================================

class HumanAttendanceTracker:
    def __init__(self):
        # 1. Detection Model (AMD GPU DirectML)
        if HAS_ORT and os.path.exists(MODEL_PERSON_ONNX):
            self.detector = DirectMLHumanDetector(MODEL_PERSON_ONNX)
        else:
            print("[INFO] Fallback to PyTorch YOLOv8n detector.")
            self.yolo = YOLO(MODEL_PERSON_PT)
            self.detector = None

        # 2. Facial Biometrics Engine
        self.biometrics = FaceBiometrics()

        # 3. People Gallery & Attendance Log
        self.gallery = {}         # pid -> TrackedPerson
        self.next_id = 1
        self.lock = threading.Lock()
        print(f"[HUMAN ENGINE] System ready for multi-person tracking & attendance.")

    def reset(self):
        with self.lock:
            self.gallery.clear()
            self.next_id = 1
        print("[RESET] Cleared human tracking gallery.")

    def update(self, frame_bgr, now, dt):
        fh, fw = frame_bgr.shape[:2]

        # 1. Detect Persons & Bags
        if self.detector is not None:
            raw_persons, detected_bags = self.detector.predict(frame_bgr)
        else:
            results = self.yolo.predict(source=frame_bgr, conf=CONF_THRESH_PERSON, classes=[0, 24, 26, 28], verbose=False)[0]
            raw_persons = []
            detected_bags = []
            if results.boxes is not None:
                for b in results.boxes:
                    cls_id = int(b.cls[0])
                    x1, y1, x2, y2 = [int(v) for v in b.xyxy[0]]
                    conf = float(b.conf[0])
                    if cls_id == 0:
                        raw_persons.append((x1, y1, x2, y2, conf))
                    elif cls_id in (24, 26, 28):
                        detected_bags.append((x1, y1, x2, y2, conf))

        # 2. Extract Multimodal Appearance & Face Biometrics per Person
        candidate_detections = []
        for x1, y1, x2, y2, conf in raw_persons:
            person_crop = frame_bgr[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue

            # Clothing & Bagging Analysis
            clothing_info = analyze_clothing_and_bag(person_crop, (x1, y1, x2, y2), detected_bags)

            # Face Biometrics
            face_feat, face_box = self.biometrics.extract_face_embedding(person_crop)

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            candidate_detections.append({
                "bbox"         : (x1, y1, x2, y2),
                "cx"           : cx, "cy": cy,
                "clothing"     : clothing_info,
                "face_feat"    : face_feat,
                "face_box"     : face_box,
                "conf"         : conf
            })

        # 3. Association & Tracking
        with self.lock:
            active_persons = [p for p in self.gallery.values() if p.is_active]
            inactive_persons = [p for p in self.gallery.values() if not p.is_active]

            matched_dets = set()
            matched_persons = set()

            # Step A: Spatial Continuity + Appearance Re-ID for Active Persons
            for person in active_persons:
                pcx, pcy = person.centroid
                best_det_idx = None
                best_cost = float('inf')

                for i, det in enumerate(candidate_detections):
                    if i in matched_dets:
                        continue
                    dist = math.hypot(det['cx'] - pcx, det['cy'] - pcy)
                    if dist < SPATIAL_GATE_PX:
                        sim = person.match_similarity(det['face_feat'], det['clothing']['signature'], self.biometrics)
                        cost = (dist / SPATIAL_GATE_PX) * 0.40 + (1.0 - sim) * 0.60
                        if cost < best_cost:
                            best_cost = cost
                            best_det_idx = i

                if best_det_idx is not None and best_cost < 0.70:
                    matched_dets.add(best_det_idx)
                    matched_persons.add(person.id)
                    d = candidate_detections[best_det_idx]
                    person.update(d['bbox'], d['clothing'], d['face_feat'], d['conf'], now, dt)

            # Step B: Re-ID for Returning Inactive Persons (Face or Clothing Signature)
            unmatched_dets = [i for i in range(len(candidate_detections)) if i not in matched_dets]
            for i in unmatched_dets:
                det = candidate_detections[i]
                best_person = None
                best_sim = 0.45

                for person in inactive_persons:
                    if person.id in matched_persons:
                        continue
                    sim = person.match_similarity(det['face_feat'], det['clothing']['signature'], self.biometrics)
                    if sim > best_sim:
                        best_sim = sim
                        best_person = person

                if best_person is not None:
                    matched_dets.add(i)
                    matched_persons.add(best_person.id)
                    best_person.update(det['bbox'], det['clothing'], det['face_feat'], det['conf'], now, dt)
                    print(f"  [RE-ID ATTENDANCE] {best_person.name} recognized & clocked! (Sim: {best_sim:.2f})")

            # Step C: Register New Person
            remaining_dets = [i for i in range(len(candidate_detections)) if i not in matched_dets]
            for i in remaining_dets:
                det = candidate_detections[i]
                new_pid = self.next_id
                self.next_id += 1
                new_person = TrackedPerson(new_pid, det['bbox'], det['clothing'], det['face_feat'], det['conf'])
                self.gallery[new_pid] = new_person
                matched_persons.add(new_pid)
                face_tag = " [Face ID ✓]" if det['face_feat'] is not None else " [Overhead View]"
                print(f"  [NEW PERSON REGISTERED] {new_person.name} | Upper: {det['clothing']['upper_color']} | Bag: {det['clothing']['bag_label']}{face_tag}")

            # Step D: Mark Missed
            for person in self.gallery.values():
                if person.id not in matched_persons:
                    person.mark_missed()

            return [p for p in self.gallery.values() if p.is_active]

    def export_attendance(self):
        """Saves current attendance logs to JSON and CSV."""
        with self.lock:
            records = [p.to_attendance_dict() for p in self.gallery.values()]

        # Save JSON
        with open(ATTENDANCE_JSON, "w") as fp:
            json.dump(records, fp, indent=2)

        # Save CSV
        if records:
            keys = records[0].keys()
            with open(ATTENDANCE_CSV, "w", newline="", encoding="utf-8") as fp:
                writer = csv.DictWriter(fp, fieldnames=keys)
                writer.writeheader()
                writer.writerows(records)

        print(f"[EXPORT] Attendance saved to {ATTENDANCE_JSON} and {ATTENDANCE_CSV} ({len(records)} persons recorded).")

# =============================================================================
#   MODERN RENDERING & ATTENDANCE HUD
# =============================================================================

def draw_corner_rect(frame, bbox, color, thickness=2, corner_len=18):
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

    cl = min(corner_len, max(8, (x2 - x1) // 4), max(8, (y2 - y1) // 4))
    t = thickness
    cv2.line(frame, (x1, y1), (x1 + cl, y1), color, t)
    cv2.line(frame, (x1, y1), (x1, y1 + cl), color, t)
    cv2.line(frame, (x2, y1), (x2 - cl, y1), color, t)
    cv2.line(frame, (x2, y1), (x2, y1 + cl), color, t)
    cv2.line(frame, (x1, y2), (x1 + cl, y2), color, t)
    cv2.line(frame, (x1, y2), (x1 - cl, y2), color, t)
    cv2.line(frame, (x2, y2), (x2 - cl, y2), color, t)
    cv2.line(frame, (x2, y2), (x2, y2 - cl), color, t)

def render_person_hud(frame, person: TrackedPerson):
    x1, y1, x2, y2 = [int(round(v)) for v in person.bbox]
    col = (0, 215, 255)  # Cyan Gold Accent

    draw_corner_rect(frame, person.bbox, col, thickness=2, corner_len=18)

    # Multi-line Badge: Identity, Face Status, Upper/Lower Clothing, Bag
    face_badge = " [Face ID]" if person.face_count > 0 else " [Overhead/Cloth Re-ID]"
    label_title = f"{person.name}{face_badge}"
    c_info = person.clothing
    bag_str = f" • {c_info['bag_label']}" if c_info['has_bag'] else ""
    label_sub = f"Top: {c_info['upper_color']} • Bottom: {c_info['lower_color']}{bag_str}"

    (tw1, th1), _ = cv2.getTextSize(label_title, FONT, 0.44, 1)
    (tw2, th2), _ = cv2.getTextSize(label_sub, FONT, 0.36, 1)
    max_w = max(tw1, tw2)

    px, py = 6, 4
    total_h = th1 + th2 + py * 3 + 2

    rx1 = max(0, x1)
    ry1 = max(0, y1 - total_h - 4)
    rx2 = min(frame.shape[1] - 1, x1 + max_w + px * 2 + 6)
    ry2 = max(0, y1 - 2)

    if ry2 > ry1 and rx2 > rx1:
        sub = frame[ry1:ry2, rx1:rx2]
        dark = np.zeros_like(sub)
        cv2.addWeighted(sub, 0.20, dark, 0.80, 0, sub)
        frame[ry1:ry2, rx1:rx2] = sub
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), col, 1)

    cv2.putText(frame, label_title, (x1 + px, ry1 + th1 + py), FONT, 0.44, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, label_sub, (x1 + px, ry1 + th1 + th2 + py * 2), FONT, 0.36, (200, 240, 255), 1, cv2.LINE_AA)

def render_attendance_banner(frame, fps: float, active_count: int, total_count: int, is_gpu: bool):
    """Clean modern top banner for attendance status."""
    gpu_badge = "GPU: AMD RX 6600 (DirectML)" if is_gpu else "CPU Mode"
    txt = (
        f"SmartEyes Attendance & Human Perception  |  {fps:.1f} FPS  |  {gpu_badge}  |  "
        f"Present Now: {active_count}  |  Total Clocked: {total_count}  |  [S] Save Attendance"
    )

    (tw, th), _ = cv2.getTextSize(txt, FONT, 0.36, 1)
    x1, y1 = 12, 10
    x2, y2 = 12 + tw + 16, 10 + th + 14

    sub = frame[y1:y2, x1:x2]
    dark = np.zeros_like(sub)
    cv2.addWeighted(sub, 0.20, dark, 0.80, 0, sub)
    frame[y1:y2, x1:x2] = sub
    cv2.rectangle(frame, (x1, y1), (x2, y2), (70, 70, 70), 1)

    cv2.putText(frame, txt, (x1 + 8, y1 + th + 6), FONT, 0.36, (240, 240, 240), 1, cv2.LINE_AA)

# =============================================================================
#   EMBEDDED WEB DASHBOARD SERVER (PORT 5001)
# =============================================================================

from flask import Flask, Response, render_template_string, jsonify, send_file
import io

web_app = Flask(__name__)
_global_jpeg_frame = None
_global_jpeg_lock = threading.Lock()
_active_tracker_ref = None

WEB_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SmartEyes | Human Perception & Attendance System</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #090d16;
      --card-bg: #111726;
      --card-border: rgba(255, 255, 255, 0.08);
      --text: #f0f4f8;
      --text-muted: #8b9bb4;
      --accent: #00d2ff;
      --accent-glow: rgba(0, 210, 255, 0.25);
      --success: #2ed573;
      --danger: #ff4757;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }

    body {
      background-color: var(--bg);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      padding: 18px 24px;
    }

    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-bottom: 16px;
      border-bottom: 1px solid var(--card-border);
      margin-bottom: 20px;
    }

    .brand { display: flex; align-items: center; gap: 12px; }

    .brand-icon {
      width: 32px; height: 32px;
      background: linear-gradient(135deg, #00d2ff, #7928ca);
      border-radius: 8px;
      display: flex; align-items: center; justify-content: center;
      font-size: 18px;
    }

    .brand h1 { font-size: 1.15rem; font-weight: 700; }

    .header-badges { display: flex; align-items: center; gap: 10px; }

    .pill {
      font-size: 0.75rem; font-weight: 600; padding: 4px 10px;
      border-radius: 20px; background: rgba(255, 255, 255, 0.05);
      border: 1px solid var(--card-border);
      display: flex; align-items: center; gap: 6px;
    }

    .pill-live {
      background: rgba(46, 213, 115, 0.12);
      border-color: rgba(46, 213, 115, 0.3);
      color: var(--success);
    }

    .pill-live::before {
      content: ""; width: 7px; height: 7px; background: var(--success);
      border-radius: 50%;
    }

    .dashboard-grid {
      display: grid;
      grid-template-columns: 1.6fr 1.2fr;
      gap: 20px;
      flex: 1;
    }

    @media (max-width: 1024px) {
      .dashboard-grid { grid-template-columns: 1fr; }
    }

    .card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 18px;
      display: flex;
      flex-direction: column;
    }

    .card-title {
      font-size: 0.88rem; font-weight: 600; color: var(--text-muted);
      text-transform: uppercase; letter-spacing: 0.05em;
      margin-bottom: 14px; display: flex; justify-content: space-between; align-items: center;
    }

    .feed-container {
      width: 100%; border-radius: 10px; overflow: hidden;
      background: #000; border: 1px solid var(--card-border);
      aspect-ratio: 16 / 9;
    }

    .feed-container img { width: 100%; height: 100%; object-fit: cover; display: block; }

    .table-container {
      flex: 1;
      overflow-y: auto;
      max-height: 520px;
      border-radius: 8px;
      border: 1px solid var(--card-border);
    }

    table {
      width: 100%; border-collapse: collapse; font-size: 0.8rem; text-align: left;
    }

    th {
      background: rgba(255, 255, 255, 0.04);
      padding: 10px 12px;
      font-weight: 600;
      color: var(--text-muted);
      border-bottom: 1px solid var(--card-border);
      position: sticky; top: 0; z-index: 1;
    }

    td {
      padding: 10px 12px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    }

    tr:hover { background: rgba(255, 255, 255, 0.02); }

    .status-badge {
      display: inline-block; padding: 2px 8px; border-radius: 12px;
      font-size: 0.72rem; font-weight: 600;
    }

    .status-present { background: rgba(46, 213, 115, 0.15); color: var(--success); }
    .status-left { background: rgba(255, 255, 255, 0.06); color: var(--text-muted); }

    .btn-action {
      background: rgba(0, 210, 255, 0.15); border: 1px solid rgba(0, 210, 255, 0.3);
      color: var(--accent); padding: 6px 12px; border-radius: 6px;
      font-size: 0.78rem; font-weight: 600; cursor: pointer; text-decoration: none;
      display: inline-flex; align-items: center; gap: 6px;
    }

    .btn-action:hover { background: rgba(0, 210, 255, 0.25); }

    .actions-bar {
      display: flex; justify-content: space-between; align-items: center;
      margin-top: 14px;
    }
  </style>
</head>
<body>

  <header>
    <div class="brand">
      <div class="brand-icon">👥</div>
      <div>
        <h1>SmartEyes Attendance & Human Perception</h1>
      </div>
    </div>
    <div class="header-badges">
      <div class="pill pill-live">LIVE FEED</div>
      <div class="pill" id="badge-gpu">AMD RX 6600 (DirectML)</div>
      <div class="pill" id="badge-present" style="color: var(--success);">0 Present Now</div>
    </div>
  </header>

  <div class="dashboard-grid">

    <!-- Column 1: Live Stream -->
    <div class="card">
      <div class="card-title">
        <span>Overhead & Room Human Tracking</span>
        <span id="fps-counter" style="color: var(--accent);">-- FPS</span>
      </div>
      <div class="feed-container">
        <img id="video-feed" src="/video_feed" alt="Live Human Feed">
      </div>
    </div>

    <!-- Column 2: Live Attendance Roll Call -->
    <div class="card">
      <div class="card-title">
        <span>Real-Time Attendance Roll Call</span>
        <span id="badge-total" class="pill" style="font-size: 0.7rem;">0 Total Clocked</span>
      </div>

      <div class="table-container">
        <table>
          <thead>
            <tr>
              <th>Person</th>
              <th>Status</th>
              <th>Clock In</th>
              <th>Duration</th>
              <th>Clothing Profile</th>
              <th>Bag</th>
            </tr>
          </thead>
          <tbody id="attendance-tbody">
            <tr>
              <td colspan="6" style="text-align: center; color: var(--text-muted); padding: 30px;">
                No persons detected yet. Waiting for camera feed...
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <div class="actions-bar">
        <a href="/api/export_csv" class="btn-action" download="attendance_log.csv">
          <span>📥</span> Download Attendance (CSV)
        </a>
        <button class="btn-action" onclick="resetGallery()" style="border-color: rgba(255,255,255,0.2); color: var(--text-muted);">
          <span>🔄</span> Reset Roll Call
        </button>
      </div>
    </div>

  </div>

  <script>
    async function updateStatus() {
      try {
        const res = await fetch('/api/status');
        const data = await res.json();

        document.getElementById('fps-counter').innerText = `${data.fps} FPS`;
        document.getElementById('badge-gpu').innerText = data.gpu;
        document.getElementById('badge-present').innerText = `${data.present_count} Present Now`;
        document.getElementById('badge-total').innerText = `${data.total_count} Total Clocked`;

        renderTable(data.records);
      } catch (err) {
        console.error("Status update error:", err);
      }
    }

    function renderTable(records) {
      const tbody = document.getElementById('attendance-tbody');
      if (!records || records.length === 0) {
        tbody.innerHTML = `<tr><td colspan="6" style="text-align: center; color: var(--text-muted); padding: 30px;">No persons detected yet.</td></tr>`;
        return;
      }

      tbody.innerHTML = records.map(r => `
        <tr>
          <td style="font-weight: 600;">
            ${r.name}
            ${r.face_recognized ? '<span style="color:#00d2ff; font-size:0.7rem; margin-left:4px;">[Face ✓]</span>' : '<span style="color:#8b9bb4; font-size:0.7rem; margin-left:4px;">[Cloth ID]</span>'}
          </td>
          <td>
            <span class="status-badge ${r.status === 'Present' ? 'status-present' : 'status-left'}">${r.status}</span>
          </td>
          <td>${r.clock_in.split(' ')[1] || r.clock_in}</td>
          <td>${r.duration_str}</td>
          <td style="color: #cbd5e1;">Top: ${r.upper_clothing} | Btm: ${r.lower_clothing}</td>
          <td>${r.bag_status !== 'None' ? '<span style="color:#ffa502;">🎒 ' + r.bag_status + '</span>' : 'None'}</td>
        </tr>
      `).join('');
    }

    async function resetGallery() {
      if (confirm("Reset attendance roll call for this session?")) {
        await fetch('/api/reset', { method: 'POST' });
        updateStatus();
      }
    }

    setInterval(updateStatus, 1500);
    updateStatus();
  </script>
</body>
</html>
"""

@web_app.route('/')
def web_index():
    return render_template_string(WEB_HTML_TEMPLATE)

def generate_web_frames():
    while True:
        with _global_jpeg_lock:
            frame_bytes = _global_jpeg_frame
        if frame_bytes is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.033)

@web_app.route('/video_feed')
def web_video_feed():
    return Response(generate_web_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@web_app.route('/api/status')
def web_api_status():
    global _active_tracker_ref
    if _active_tracker_ref is None:
        return jsonify({"error": "Tracker not active"}), 503
    with _active_tracker_ref.lock:
        records = [p.to_attendance_dict() for p in _active_tracker_ref.gallery.values()]
        active_count = sum(1 for r in records if r["status"] == "Present")
        total_count = len(records)
    is_gpu = getattr(_active_tracker_ref.detector, "is_gpu", False) if _active_tracker_ref.detector else False
    return jsonify({
        "fps": 30.0,
        "present_count": active_count,
        "total_count": total_count,
        "records": records,
        "gpu": "AMD Radeon RX 6600 (DirectML)" if is_gpu else "CPU"
    })

@web_app.route('/api/reset', methods=['POST'])
def web_api_reset():
    global _active_tracker_ref
    if _active_tracker_ref is not None:
        _active_tracker_ref.reset()
        return jsonify({"success": True})
    return jsonify({"error": "Tracker not active"}), 503

@web_app.route('/api/export_csv')
def web_api_export_csv():
    global _active_tracker_ref
    if _active_tracker_ref is None:
        return "Tracker not active", 503
    with _active_tracker_ref.lock:
        records = [p.to_attendance_dict() for p in _active_tracker_ref.gallery.values()]
    output = io.StringIO()
    if records:
        writer = csv.DictWriter(output, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    mem = io.BytesIO()
    mem.write(output.getvalue().encode('utf-8'))
    mem.seek(0)
    return send_file(mem, mimetype='text/csv', as_attachment=True, download_name='attendance_log.csv')

def start_embedded_web_server(port=5001):
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    web_app.run(host='0.0.0.0', port=port, debug=False, threaded=True, use_reloader=False)

# =============================================================================
#   MAIN
# =============================================================================

def run():
    global _active_tracker_ref, _global_jpeg_frame
    print("=" * 72)
    print(" SmartEyes - Multi-Person Re-ID, Clothing & Attendance System (v1.0) ")
    print("=" * 72)

    vs = VideoStream(src=CAMERA_INDEX, width=CAPTURE_WIDTH, height=CAPTURE_HEIGHT)
    time.sleep(0.5)

    if not vs.is_opened():
        sys.exit(f"[ERROR] Could not open camera {CAMERA_INDEX}.")

    tracker = HumanAttendanceTracker()
    _active_tracker_ref = tracker

    # Start Embedded Web Server on Port 5001 in Background Thread
    web_thread = threading.Thread(target=start_embedded_web_server, args=(5001,), daemon=True)
    web_thread.start()

    cv2.namedWindow("SmartEyes Human & Attendance Tracker", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("SmartEyes Human & Attendance Tracker", DISPLAY_W, DISPLAY_H)

    prev_time = time.time()
    ema_fps = 30.0

    print("\n[INFO] Human Tracking & Attendance Active.")
    print("[INFO] Features: Multi-Person Tracking, Facial Recognition, Clothing & Bag Re-ID.")
    print("[WEB]  Live Attendance Web Dashboard: http://localhost:5001 (or http://127.0.0.1:5001)")
    print("[KEYS] S: Save attendance to CSV/JSON | R: Reset | Q: Quit\n")

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

            # Run human tracking, clothing analysis, and attendance
            active_persons = tracker.update(disp, now, dt)

            # Render person cards
            for p in active_persons:
                render_person_hud(disp, p)

            is_gpu = getattr(tracker.detector, "is_gpu", False) if tracker.detector else False
            render_attendance_banner(disp, ema_fps, len(active_persons), len(tracker.gallery), is_gpu)

            # Update Web Stream Buffer
            ret_enc, jpeg = cv2.imencode('.jpg', disp, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret_enc:
                with _global_jpeg_lock:
                    _global_jpeg_frame = jpeg.tobytes()

            cv2.imshow("SmartEyes Human & Attendance Tracker", disp)

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q') or k == ord('Q') or k == 27:
                break
            elif k == ord('s') or k == ord('S'):
                tracker.export_attendance()
            elif k == ord('r') or k == ord('R'):
                tracker.reset()
            elif k == ord('a') or k == ord('A'):
                print("\n" + "=" * 50)
                print("            LIVE ATTENDANCE ROLL CALL             ")
                print("=" * 50)
                for p in tracker.gallery.values():
                    att = p.to_attendance_dict()
                    print(f"[{att['status']}] {att['name']} | In: {att['clock_in']} | Duration: {att['duration_str']} | Top: {att['upper_clothing']} | Bag: {att['bag_status']}")
                print("=" * 50)

    finally:
        tracker.export_attendance()
        vs.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run()
