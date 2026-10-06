# -*- coding: utf-8 -*-
"""
SmartEyes - Minimalist Web Dashboard with Live Stream, Feeder & Daily Timer (v1.0)
==================================================================================
Key Features:
  1. Live Low-Latency MJPEG Video Stream (AMD GPU DirectML Accelerated).
  2. Automated Feeder Section:
     - Manual "Feed Fish Now" trigger (ready for ESP32 / ESP8266 integration).
     - Daily Feeding Timer system (add/remove recurring daily feeding times).
     - Live countdown timer to next feeding time.
     - Persistent storage in 'feeder_schedule.json'.
  3. Real-Time Health & Behavior Alert Section:
     - Surface Floating / Dead Fish Warnings.
     - Inactivity / Lethargy warnings.
  4. Minimalist, modern glassmorphic interface with responsive layout.
"""

import time
import json
import math
import sys
import os
import threading
from collections import deque
from datetime import datetime, timedelta

import cv2
import numpy as np
from flask import Flask, Response, render_template_string, jsonify, request

# Import existing high-performance engine components
import fish_tracker_deep as ftd

app = Flask(__name__)

# =============================================================================
#   FEEDER SCHEDULE & STATE PERSISTENCE
# =============================================================================

SCHEDULE_FILE = "feeder_schedule.json"

DEFAULT_SCHEDULE = {
    "timers": [
        {"id": 1, "time": "09:00", "label": "Morning Feed", "enabled": True},
        {"id": 2, "time": "14:00", "label": "Afternoon Feed", "enabled": True},
        {"id": 3, "time": "20:00", "label": "Evening Feed", "enabled": True}
    ],
    "last_fed": None,
    "feed_history": []
}

def load_schedule():
    if os.path.exists(SCHEDULE_FILE):
        try:
            with open(SCHEDULE_FILE, "r") as fp:
                return json.load(fp)
        except Exception:
            pass
    return DEFAULT_SCHEDULE.copy()

def save_schedule(data):
    try:
        with open(SCHEDULE_FILE, "w") as fp:
            json.dump(data, fp, indent=2)
    except Exception as e:
        print(f"[ERROR] Could not save schedule: {e}")

schedule_data = load_schedule()
schedule_lock = threading.Lock()

# =============================================================================
#   GLOBAL VIDEO & TRACKING WORKER
# =============================================================================

class WebStreamWorker:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_jpeg = None
        self.active_fishes = []
        self.active_alerts = deque(maxlen=20)
        self.fps = 30.0
        self.running = True
        self.last_auto_feed_minute = None

        # Load models
        model_onnx = ftd.CUSTOM_MODEL_ONNX if os.path.exists(ftd.CUSTOM_MODEL_ONNX) else ftd.FALLBACK_MODEL_ONNX
        model_pt = ftd.CUSTOM_MODEL_PT if os.path.exists(ftd.CUSTOM_MODEL_PT) else ftd.FALLBACK_MODEL_PT
        
        print(f"[WEB STREAM] Initializing Deep Tracker on AMD GPU...")
        self.tracker = ftd.DeepFishTracker(model_onnx, model_pt)
        ftd._global_tracker = self.tracker

        # Start Camera Stream
        self.vs = ftd.VideoStream(src=ftd.CAMERA_INDEX, width=ftd.CAPTURE_WIDTH, height=ftd.CAPTURE_HEIGHT)
        time.sleep(0.5)

        # Worker Thread
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()

        # Schedule Checker Thread
        self.schedule_thread = threading.Thread(target=self._schedule_loop, daemon=True)
        self.schedule_thread.start()

    def _process_loop(self):
        prev_time = time.time()
        while self.running:
            ret, frame = self.vs.read()
            now = time.time()
            dt = max(now - prev_time, 1e-5)
            prev_time = now
            self.fps = 0.88 * self.fps + 0.12 * (1.0 / dt)

            if not ret or frame is None:
                # Fallback blank frame if camera disconnected
                disp = np.zeros((ftd.DISPLAY_H, ftd.DISPLAY_W, 3), dtype=np.uint8)
                cv2.putText(disp, "Connecting to Aquarium Camera...", (380, 360), ftd.FONT, 0.8, (180, 180, 180), 2)
            else:
                disp = cv2.resize(frame, (ftd.DISPLAY_W, ftd.DISPLAY_H))

                # Run tracking with DirectML AMD GPU
                active = self.tracker.update(disp, now, dt)
                with self.lock:
                    self.active_fishes = active

                # Render fish bounding cards
                for f in active:
                    f.total_sec += dt
                    ftd.render_fish_card(disp, f)

                # Render blacklisted exclusion zones
                for ex1, ey1, ex2, ey2 in self.tracker.excluded_regions:
                    ecx = (ex1 + ex2) // 2
                    ecy = (ey1 + ey2) // 2
                    cv2.circle(disp, (ecx, ecy), 50, (0, 0, 180), 1, cv2.LINE_AA)
                    cv2.putText(disp, "EXCLUDED", (ex1, ey1 - 4), ftd.FONT, 0.32, (0, 0, 200), 1, cv2.LINE_AA)

                # Collect alerts
                with self.lock:
                    for ts, msg, col in list(self.tracker.alert_feed):
                        alert_item = {"timestamp": ts, "message": msg, "severity": "critical" if "CRITICAL" in msg else "warning"}
                        if not any(a["timestamp"] == ts and a["message"] == msg for a in self.active_alerts):
                            self.active_alerts.append(alert_item)

            # Encode frame to JPEG
            ret_enc, jpeg = cv2.imencode('.jpg', disp, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret_enc:
                with self.lock:
                    self.latest_jpeg = jpeg.tobytes()

            time.sleep(0.015)  # Steady ~30-40 FPS loop

    def _schedule_loop(self):
        """Monitors daily feeding timers and executes scheduled feeds."""
        while self.running:
            now_dt = datetime.now()
            current_hm = now_dt.strftime("%H:%M")

            if current_hm != self.last_auto_feed_minute:
                with schedule_lock:
                    for timer in schedule_data.get("timers", []):
                        if timer.get("enabled", True) and timer.get("time") == current_hm:
                            self.trigger_feed(source=f"Scheduled ({timer.get('label', 'Daily Timer')})")
                            self.last_auto_feed_minute = current_hm
                            break

            time.sleep(2.0)

    def trigger_feed(self, source="Manual Button"):
        """Triggers the feeder event (ready for ESP32/ESP8266 integration)."""
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with schedule_lock:
            schedule_data["last_fed"] = now_str
            schedule_data["feed_history"].insert(0, {"timestamp": now_str, "source": source})
            if len(schedule_data["feed_history"]) > 30:
                schedule_data["feed_history"].pop()
            save_schedule(schedule_data)

        # Log an alert in feed
        with self.lock:
            ts = datetime.now().strftime("%H:%M:%S")
            self.active_alerts.append({
                "timestamp": ts,
                "message": f"🍽️ Fish Feeder Activated via {source}!",
                "severity": "info"
            })
        print(f"[FEEDER] >>> Feeder triggered at {now_str} via {source} (ESP Ready) <<<")

    def get_frame(self):
        with self.lock:
            return self.latest_jpeg

    def get_status(self):
        with self.lock:
            fishes = [f.to_dict() for f in self.active_fishes]
            alerts = list(self.active_alerts)
            fps = round(self.fps, 1)

        with schedule_lock:
            last_fed = schedule_data.get("last_fed")
            timers = schedule_data.get("timers", [])

        # Calculate time to next scheduled feed
        next_feed_str = "None scheduled"
        next_feed_seconds = None
        now_dt = datetime.now()
        enabled_timers = [t for t in timers if t.get("enabled", True)]

        if enabled_timers:
            candidates = []
            for t in enabled_timers:
                try:
                    th, tm = [int(v) for v in t["time"].split(":")]
                    target_dt = now_dt.replace(hour=th, minute=tm, second=0, microsecond=0)
                    if target_dt <= now_dt:
                        target_dt += timedelta(days=1)
                    delta_sec = (target_dt - now_dt).total_seconds()
                    candidates.append((delta_sec, t["time"], t.get("label", "Timer")))
                except Exception:
                    pass

            if candidates:
                candidates.sort(key=lambda x: x[0])
                shortest_sec, nxt_time, nxt_label = candidates[0]
                hours = int(shortest_sec // 3600)
                mins = int((shortest_sec % 3600) // 60)
                next_feed_str = f"{nxt_time} ({hours}h {mins}m)"
                next_feed_seconds = int(shortest_sec)

        return {
            "fps": fps,
            "active_count": len(fishes),
            "fishes": fishes,
            "alerts": alerts[-10:],
            "last_fed": last_fed or "Not yet today",
            "next_feed": next_feed_str,
            "next_feed_seconds": next_feed_seconds,
            "timers": timers,
            "gpu": "AMD Radeon RX 6600 (DirectML)" if getattr(self.tracker, "is_gpu", False) else "CPU"
        }

worker = None

# =============================================================================
#   HTML TEMPLATE (MINIMALIST, MODERN, SLEEK)
# =============================================================================

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SmartEyes | Aquarium Live Feed & Feeder</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #090d16;
      --card-bg: #111726;
      --card-border: rgba(255, 255, 255, 0.08);
      --card-border-focus: rgba(0, 195, 255, 0.3);
      --text: #f0f4f8;
      --text-muted: #8b9bb4;
      --accent: #00d2ff;
      --accent-glow: rgba(0, 210, 255, 0.25);
      --danger: #ff4757;
      --warning: #ffa502;
      --success: #2ed573;
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    body {
      background-color: var(--bg);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      padding: 18px 24px;
    }

    /* Top Bar */
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-bottom: 16px;
      border-bottom: 1px solid var(--card-border);
      margin-bottom: 20px;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .brand-icon {
      width: 32px;
      height: 32px;
      background: linear-gradient(135deg, var(--accent), #0072ff);
      border-radius: 8px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 18px;
      box-shadow: 0 0 16px var(--accent-glow);
    }

    .brand h1 {
      font-size: 1.15rem;
      font-weight: 700;
      letter-spacing: -0.02em;
    }

    .header-badges {
      display: flex;
      align-items: center;
      gap: 10px;
    }

    .pill {
      font-size: 0.75rem;
      font-weight: 600;
      padding: 4px 10px;
      border-radius: 20px;
      display: flex;
      align-items: center;
      gap: 6px;
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid var(--card-border);
    }

    .pill-live {
      background: rgba(46, 213, 115, 0.12);
      border-color: rgba(46, 213, 115, 0.3);
      color: var(--success);
    }

    .pill-live::before {
      content: "";
      width: 7px;
      height: 7px;
      background: var(--success);
      border-radius: 50%;
      animation: pulse 1.6s infinite;
    }

    @keyframes pulse {
      0% { opacity: 0.4; transform: scale(0.9); }
      50% { opacity: 1; transform: scale(1.15); }
      100% { opacity: 0.4; transform: scale(0.9); }
    }

    /* Main Grid Layout */
    .dashboard-grid {
      display: grid;
      grid-template-columns: 1.8fr 1fr;
      gap: 20px;
      flex: 1;
    }

    @media (max-width: 960px) {
      .dashboard-grid {
        grid-template-columns: 1fr;
      }
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
      font-size: 0.88rem;
      font-weight: 600;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 14px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    /* Video Feed Container */
    .feed-container {
      position: relative;
      width: 100%;
      border-radius: 10px;
      overflow: hidden;
      background: #000;
      border: 1px solid var(--card-border);
      aspect-ratio: 16 / 9;
    }

    .feed-container img {
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
      cursor: crosshair;
    }

    .feed-overlay-hint {
      position: absolute;
      bottom: 8px;
      right: 10px;
      background: rgba(0, 0, 0, 0.65);
      backdrop-filter: blur(4px);
      padding: 4px 8px;
      border-radius: 6px;
      font-size: 0.68rem;
      color: #cbd5e1;
      pointer-events: none;
    }

    /* Feeder Controls */
    .feeder-actions {
      display: flex;
      flex-direction: column;
      gap: 14px;
    }

    .btn-feed {
      background: linear-gradient(135deg, #00d2ff, #0066ff);
      color: #fff;
      border: none;
      padding: 14px 18px;
      border-radius: 10px;
      font-size: 0.95rem;
      font-weight: 600;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
      transition: all 0.2s ease;
      box-shadow: 0 4px 18px var(--accent-glow);
    }

    .btn-feed:hover {
      transform: translateY(-2px);
      box-shadow: 0 6px 22px rgba(0, 210, 255, 0.4);
    }

    .btn-feed:active {
      transform: translateY(0);
    }

    .feeder-meta {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }

    .meta-box {
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 10px 12px;
    }

    .meta-label {
      font-size: 0.72rem;
      color: var(--text-muted);
      margin-bottom: 4px;
    }

    .meta-value {
      font-size: 0.95rem;
      font-weight: 600;
    }

    /* Timers List */
    .timer-section {
      margin-top: 14px;
      border-top: 1px solid var(--card-border);
      padding-top: 14px;
    }

    .timer-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 10px;
      font-size: 0.78rem;
      font-weight: 600;
      color: var(--text-muted);
    }

    .timer-list {
      display: flex;
      flex-direction: column;
      gap: 8px;
      max-height: 150px;
      overflow-y: auto;
    }

    .timer-item {
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: rgba(255, 255, 255, 0.02);
      border: 1px solid var(--card-border);
      padding: 8px 12px;
      border-radius: 8px;
    }

    .timer-time {
      font-size: 0.9rem;
      font-weight: 600;
      color: var(--accent);
    }

    .timer-label {
      font-size: 0.75rem;
      color: var(--text-muted);
      margin-left: 8px;
    }

    .btn-remove-timer {
      background: none;
      border: none;
      color: var(--text-muted);
      cursor: pointer;
      font-size: 1rem;
      padding: 0 4px;
      transition: color 0.15s;
    }

    .btn-remove-timer:hover {
      color: var(--danger);
    }

    .add-timer-form {
      display: flex;
      gap: 8px;
      margin-top: 10px;
    }

    .input-time, .input-text {
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid var(--card-border);
      border-radius: 6px;
      padding: 6px 10px;
      color: var(--text);
      font-size: 0.8rem;
    }

    .input-time:focus, .input-text:focus {
      outline: none;
      border-color: var(--accent);
    }

    .btn-add {
      background: rgba(0, 210, 255, 0.15);
      border: 1px solid rgba(0, 210, 255, 0.3);
      color: var(--accent);
      border-radius: 6px;
      padding: 6px 12px;
      font-size: 0.8rem;
      font-weight: 600;
      cursor: pointer;
      white-space: nowrap;
    }

    .btn-add:hover {
      background: rgba(0, 210, 255, 0.25);
    }

    /* Alerts Section */
    .alerts-container {
      display: flex;
      flex-direction: column;
      gap: 8px;
      max-height: 220px;
      overflow-y: auto;
    }

    .alert-card {
      padding: 10px 12px;
      border-radius: 8px;
      font-size: 0.82rem;
      display: flex;
      gap: 10px;
      align-items: flex-start;
      border-left: 3px solid transparent;
      background: rgba(255, 255, 255, 0.02);
    }

    .alert-critical {
      border-left-color: var(--danger);
      background: rgba(255, 71, 87, 0.08);
      color: #ff9ca3;
    }

    .alert-warning {
      border-left-color: var(--warning);
      background: rgba(255, 165, 2, 0.08);
      color: #ffe082;
    }

    .alert-info {
      border-left-color: var(--accent);
      background: rgba(0, 210, 255, 0.08);
      color: #a0f0ff;
    }

    .alert-time {
      font-size: 0.72rem;
      opacity: 0.7;
      white-space: nowrap;
    }

    .empty-state {
      text-align: center;
      padding: 24px 10px;
      color: var(--text-muted);
      font-size: 0.82rem;
    }

    .empty-icon {
      font-size: 1.6rem;
      margin-bottom: 6px;
      opacity: 0.6;
    }
  </style>
</head>
<body>

  <!-- Top Header -->
  <header>
    <div class="brand">
      <div class="brand-icon">🐟</div>
      <div>
        <h1>SmartEyes Tank Monitor</h1>
      </div>
    </div>
    <div class="header-badges">
      <div class="pill pill-live">LIVE FEED</div>
      <div class="pill" id="badge-gpu">AMD RX 6600 (DirectML)</div>
      <div class="pill" id="badge-count">0 Fish Active</div>
    </div>
  </header>

  <!-- Main Dashboard Grid -->
  <div class="dashboard-grid">

    <!-- Column 1: Live Video Feed -->
    <div class="card">
      <div class="card-title">
        <span>Aquarium Camera Feed</span>
        <span id="fps-counter" style="font-size: 0.75rem; color: var(--accent);">-- FPS</span>
      </div>
      <div class="feed-container">
        <img id="video-feed" src="/video_feed" alt="Live Fish Camera Stream" onclick="onStreamClick(event)">
        <div class="feed-overlay-hint">Click on air pump/decor to blacklist zone</div>
      </div>
    </div>

    <!-- Column 2: Feeder & Alert Controls -->
    <div style="display: flex; flex-direction: column; gap: 20px;">
      
      <!-- Feeder & Timer Card -->
      <div class="card">
        <div class="card-title">
          <span>Automatic Fish Feeder</span>
          <span style="font-size: 0.72rem; color: var(--text-muted);">ESP Ready</span>
        </div>
        
        <div class="feeder-actions">
          <button class="btn-feed" id="btn-feed" onclick="triggerFeed()">
            <span>🍽️</span>
            <span>Feed Fish Now</span>
          </button>

          <div class="feeder-meta">
            <div class="meta-box">
              <div class="meta-label">LAST FED</div>
              <div class="meta-value" id="last-fed-text">Loading...</div>
            </div>
            <div class="meta-box">
              <div class="meta-label">NEXT SCHEDULED</div>
              <div class="meta-value" id="next-feed-text" style="color: var(--accent);">Loading...</div>
            </div>
          </div>

          <!-- Daily Timers Management -->
          <div class="timer-section">
            <div class="timer-header">
              <span>DAILY FEEDING TIMERS</span>
            </div>
            
            <div class="timer-list" id="timer-list">
              <!-- Rendered via JavaScript -->
            </div>

            <form class="add-timer-form" onsubmit="addTimer(event)">
              <input type="time" class="input-time" id="new-timer-time" required>
              <input type="text" class="input-text" id="new-timer-label" placeholder="Label (e.g. Lunch)" style="flex: 1;">
              <button type="submit" class="btn-add">+ Add</button>
            </form>
          </div>
        </div>
      </div>

      <!-- Health & Behavior Alerts Card -->
      <div class="card" style="flex: 1;">
        <div class="card-title">
          <span>Health & Inactivity Alerts</span>
          <span id="alert-count-badge" class="pill" style="font-size: 0.68rem;">0 Alerts</span>
        </div>
        
        <div class="alerts-container" id="alerts-container">
          <div class="empty-state">
            <div class="empty-icon">🛡️</div>
            <div>All fish are active & healthy. No alerts.</div>
          </div>
        </div>
      </div>

    </div>
  </div>

  <script>
    // Fetch and sync telemetry state
    async function updateStatus() {
      try {
        const res = await fetch('/api/status');
        const data = await res.json();

        // Update Header Badges
        document.getElementById('fps-counter').innerText = `${data.fps} FPS`;
        document.getElementById('badge-count').innerText = `${data.active_count} Fish Active`;
        if (data.gpu) {
          document.getElementById('badge-gpu').innerText = data.gpu;
        }

        // Update Feeder Info
        document.getElementById('last-fed-text').innerText = data.last_fed;
        document.getElementById('next-feed-text').innerText = data.next_feed;

        // Render Timers
        renderTimers(data.timers);

        // Render Alerts
        renderAlerts(data.alerts);
      } catch (err) {
        console.error("Failed to update status:", err);
      }
    }

    function renderTimers(timers) {
      const list = document.getElementById('timer-list');
      if (!timers || timers.length === 0) {
        list.innerHTML = `<div style="font-size:0.75rem; color: var(--text-muted); padding: 4px;">No daily timers configured.</div>`;
        return;
      }

      list.innerHTML = timers.map(t => `
        <div class="timer-item">
          <div>
            <span class="timer-time">${t.time}</span>
            <span class="timer-label">${t.label || 'Daily'}</span>
          </div>
          <button class="btn-remove-timer" onclick="deleteTimer(${t.id})" title="Delete timer">✕</button>
        </div>
      `).join('');
    }

    function renderAlerts(alerts) {
      const container = document.getElementById('alerts-container');
      const badge = document.getElementById('alert-count-badge');

      if (!alerts || alerts.length === 0) {
        badge.innerText = "0 Alerts";
        badge.style.color = "var(--text-muted)";
        container.innerHTML = `
          <div class="empty-state">
            <div class="empty-icon">🛡️</div>
            <div>All fish are active & healthy. No alerts.</div>
          </div>
        `;
        return;
      }

      badge.innerText = `${alerts.length} Alert${alerts.length > 1 ? 's' : ''}`;
      badge.style.color = "var(--danger)";

      container.innerHTML = alerts.slice().reverse().map(a => {
        const cls = a.severity === 'critical' ? 'alert-critical' : (a.severity === 'warning' ? 'alert-warning' : 'alert-info');
        return `
          <div class="alert-card ${cls}">
            <div class="alert-time">[${a.timestamp}]</div>
            <div style="flex:1;">${a.message}</div>
          </div>
        `;
      }).join('');
    }

    // Manual Feed Trigger
    async function triggerFeed() {
      const btn = document.getElementById('btn-feed');
      btn.disabled = true;
      btn.style.opacity = '0.7';
      btn.innerHTML = `<span>⏳</span><span>Feeding...</span>`;

      try {
        const res = await fetch('/api/feed', { method: 'POST' });
        const data = await res.json();
        btn.innerHTML = `<span>✅</span><span>Fed Successfully!</span>`;
        setTimeout(() => {
          btn.disabled = false;
          btn.style.opacity = '1';
          btn.innerHTML = `<span>🍽️</span><span>Feed Fish Now</span>`;
        }, 1800);
        updateStatus();
      } catch (e) {
        btn.disabled = false;
        btn.style.opacity = '1';
        btn.innerHTML = `<span>🍽️</span><span>Feed Fish Now</span>`;
      }
    }

    // Add Timer
    async function addTimer(ev) {
      ev.preventDefault();
      const timeVal = document.getElementById('new-timer-time').value;
      const labelVal = document.getElementById('new-timer-label').value || 'Daily Feed';
      if (!timeVal) return;

      await fetch('/api/schedule', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'add', time: timeVal, label: labelVal })
      });

      document.getElementById('new-timer-time').value = '';
      document.getElementById('new-timer-label').value = '';
      updateStatus();
    }

    // Delete Timer
    async function deleteTimer(id) {
      await fetch('/api/schedule', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'delete', id: id })
      });
      updateStatus();
    }

    // Click on video feed to blacklist air pump/zone
    async function onStreamClick(ev) {
      const img = ev.target;
      const rect = img.getBoundingClientRect();
      const clickX = ev.clientX - rect.left;
      const clickY = ev.clientY - rect.top;

      // Scale to 1280x720 video coordinates
      const origX = Math.round((clickX / rect.width) * 1280);
      const origY = Math.round((clickY / rect.height) * 720);

      await fetch('/api/exclude', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ x: origX, y: origY })
      });
    }

    // Start auto polling every 1.5 seconds
    setInterval(updateStatus, 1500);
    updateStatus();
  </script>
</body>
</html>
"""

# =============================================================================
#   FLASK ROUTES & API
# =============================================================================

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

def generate_frames():
    while True:
        if worker is not None:
            jpeg = worker.get_frame()
            if jpeg is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')
        time.sleep(0.03)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def api_status():
    if worker is not None:
        return jsonify(worker.get_status())
    return jsonify({"error": "Worker not ready"}), 503

@app.route('/api/feed', methods=['POST'])
def api_feed():
    if worker is not None:
        worker.trigger_feed(source="Web Dashboard")
        return jsonify({"success": True, "message": "Feeder triggered"})
    return jsonify({"error": "Worker not ready"}), 503

@app.route('/api/schedule', methods=['GET', 'POST'])
def api_schedule():
    with schedule_lock:
        if request.method == 'GET':
            return jsonify(schedule_data.get("timers", []))

        body = request.get_json() or {}
        action = body.get('action')
        timers = schedule_data.get("timers", [])

        if action == 'add':
            time_val = body.get('time')
            label_val = body.get('label', 'Daily Feed')
            if time_val:
                new_id = int(time.time() * 1000) % 100000
                timers.append({"id": new_id, "time": time_val, "label": label_val, "enabled": True})
                schedule_data["timers"] = timers
                save_schedule(schedule_data)
                return jsonify({"success": True, "timers": timers})

        elif action == 'delete':
            timer_id = body.get('id')
            schedule_data["timers"] = [t for t in timers if t.get('id') != timer_id]
            save_schedule(schedule_data)
            return jsonify({"success": True, "timers": schedule_data["timers"]})

        return jsonify({"error": "Invalid action"}), 400

@app.route('/api/exclude', methods=['POST'])
def api_exclude():
    if worker is not None:
        body = request.get_json() or {}
        x = body.get('x', 0)
        y = body.get('y', 0)
        worker.tracker.add_exclusion_zone(x, y, radius=55)
        return jsonify({"success": True, "x": x, "y": y})
    return jsonify({"error": "Worker not ready"}), 503

# =============================================================================
#   MAIN RUNNER
# =============================================================================

def start_server(port=5000):
    global worker
    print("=" * 70)
    print(" SmartEyes Aquarium Web Dashboard - Starting Web Server ")
    print("=" * 70)
    worker = WebStreamWorker()
    print(f"\n[WEB] Server running at: http://127.0.0.1:{port}")
    print(f"[WEB] Open http://localhost:{port} in your browser to view the live feed.")
    print("[WEB] Press Ctrl+C in terminal to stop.\n")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)

if __name__ == '__main__':
    start_server(port=5000)
