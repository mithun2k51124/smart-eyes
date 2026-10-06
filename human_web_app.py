# -*- coding: utf-8 -*-
"""
SmartEyes - Human Perception, Multi-Person Re-ID & Attendance Web Dashboard (v1.0)
==================================================================================
Key Features:
  - DOES NOT interfere with the SmartEyes Fish Tracking subsystem.
  - Live Browser MJPEG Stream of Human Tracking with AMD GPU DirectML.
  - Real-Time Attendance Roll Call Table (Auto Clock-In, Duration, Status).
  - Overhead & Top-Down Camera Friendly: Shows Upper/Lower Clothing & Bagging Profile.
  - Download Attendance as CSV button.
  - Runs on port 5001 (separate from fish dashboard on port 5000).
"""

import time
import json
import csv
import io
import math
import sys
import os
import threading
from collections import deque
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, Response, render_template_string, jsonify, request, send_file

import human_tracker as ht

app = Flask(__name__)

# =============================================================================
#   BACKGROUND STREAM WORKER
# =============================================================================

class HumanWebWorker:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_jpeg = None
        self.fps = 30.0
        self.running = True

        print("[HUMAN WEB] Starting Human Attendance Tracker...")
        self.tracker = ht.HumanAttendanceTracker()
        self.vs = ht.VideoStream(src=ht.CAMERA_INDEX, width=ht.CAPTURE_WIDTH, height=ht.CAPTURE_HEIGHT)
        time.sleep(0.5)

        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()

    def _process_loop(self):
        prev_time = time.time()
        while self.running:
            ret, frame = self.vs.read()
            now = time.time()
            dt = max(now - prev_time, 1e-5)
            prev_time = now
            self.fps = 0.88 * self.fps + 0.12 * (1.0 / dt)

            if not ret or frame is None:
                disp = np.zeros((ht.DISPLAY_H, ht.DISPLAY_W, 3), dtype=np.uint8)
                cv2.putText(disp, "Connecting to Human Tracking Camera...", (350, 360), ht.FONT, 0.8, (180, 180, 180), 2)
            else:
                disp = cv2.resize(frame, (ht.DISPLAY_W, ht.DISPLAY_H))

                # Run Human Tracking & Attendance on AMD GPU
                active_persons = self.tracker.update(disp, now, dt)

                # Render person HUD
                for p in active_persons:
                    ht.render_person_hud(disp, p)

                is_gpu = getattr(self.tracker.detector, "is_gpu", False) if self.tracker.detector else False
                ht.render_attendance_banner(disp, self.fps, len(active_persons), len(self.tracker.gallery), is_gpu)

            # Encode frame to JPEG
            ret_enc, jpeg = cv2.imencode('.jpg', disp, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret_enc:
                with self.lock:
                    self.latest_jpeg = jpeg.tobytes()

            time.sleep(0.015)

    def get_frame(self):
        with self.lock:
            return self.latest_jpeg

    def get_status(self):
        with self.tracker.lock:
            records = [p.to_attendance_dict() for p in self.tracker.gallery.values()]
            active_count = sum(1 for r in records if r["status"] == "Present")
            total_count = len(records)

        is_gpu = getattr(self.tracker.detector, "is_gpu", False) if self.tracker.detector else False
        return {
            "fps": round(self.fps, 1),
            "present_count": active_count,
            "total_count": total_count,
            "records": records,
            "gpu": "AMD Radeon RX 6600 (DirectML)" if is_gpu else "CPU"
        }

worker = None

# =============================================================================
#   MINIMALIST HTML TEMPLATE
# =============================================================================

HTML_TEMPLATE = """<!DOCTYPE html>
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

    /* Attendance Table */
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

# =============================================================================
#   ROUTES
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

@app.route('/api/reset', methods=['POST'])
def api_reset():
    if worker is not None:
        worker.tracker.reset()
        return jsonify({"success": True})
    return jsonify({"error": "Worker not ready"}), 503

@app.route('/api/export_csv')
def export_csv():
    if worker is None:
        return "Worker not ready", 503

    with worker.tracker.lock:
        records = [p.to_attendance_dict() for p in worker.tracker.gallery.values()]

    output = io.StringIO()
    if records:
        writer = csv.DictWriter(output, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    mem = io.BytesIO()
    mem.write(output.getvalue().encode('utf-8'))
    mem.seek(0)
    return send_file(mem, mimetype='text/csv', as_attachment=True, download_name='attendance_log.csv')

def start_server(port=5001):
    global worker
    import webbrowser
    print("=" * 72)
    print(" SmartEyes Human Attendance Web Dashboard - Starting Server ")
    print("=" * 72)
    worker = HumanWebWorker()
    print(f"\n[WEB] Human Attendance Dashboard: http://127.0.0.1:{port}")
    print(f"[WEB] Fish Monitor Dashboard remains at: http://127.0.0.1:5000")
    print("[WEB] Opening dashboard in browser...")
    try:
        webbrowser.open(f"http://localhost:{port}")
    except Exception:
        pass
    print("[WEB] Press Ctrl+C in terminal to stop.\n")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)

if __name__ == '__main__':
    start_server(port=5001)
