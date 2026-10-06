# -*- coding: utf-8 -*-
"""
SmartEyes - Tank Dataset Collector
===================================
Captures images from your webcam to create a custom deep learning training dataset
specifically for your fish tank.

How to use:
  1. Point the camera at your fish tank.
  2. Run this script: python dataset_collector.py
  3. Press [Space] to capture a frame (or let auto-capture snap a frame every 2 seconds).
  4. Capture around 30 to 60 images of your fish swimming in different positions.
  5. Press [Q] when done!

Images are saved to: dataset/raw/
"""

import time
import os
import cv2

SAVE_DIR = os.path.join("dataset", "raw")
os.makedirs(SAVE_DIR, exist_ok=True)

CAMERA_INDEX = 0
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720

def run():
    print("=" * 60)
    print("   SmartEyes - Fish Dataset Collector for Deep Learning   ")
    print("=" * 60)
    print(f"[INFO] Saving images to: {os.path.abspath(SAVE_DIR)}")
    print("[KEYS] [Space]: Capture 1 frame | [A]: Toggle Auto-capture (every 2s) | [Q]: Finish\n")

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)

    if not cap.isOpened():
        print(f"[ERROR] Could not open camera {CAMERA_INDEX}")
        return

    # Count existing images
    existing = [f for f in os.listdir(SAVE_DIR) if f.endswith(('.jpg', '.png'))]
    img_count = len(existing)
    print(f"[INFO] Found {img_count} existing images in dataset/raw/")

    auto_capture = False
    last_auto_time = time.time()
    cv2.namedWindow("SmartEyes Dataset Collector", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("SmartEyes Dataset Collector", 1280, 720)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            now = time.time()
            disp = frame.copy()

            # Auto-capture check
            if auto_capture and (now - last_auto_time >= 2.0):
                img_count += 1
                filename = os.path.join(SAVE_DIR, f"fish_{img_count:04d}.jpg")
                cv2.imwrite(filename, frame)
                last_auto_time = now
                print(f"  [SAVED] {filename} (Total: {img_count})")

            # UI Overlay
            h, w = disp.shape[:2]
            status_auto = "ON (every 2s)" if auto_capture else "OFF"
            info = f"Captured: {img_count} images  |  Auto-capture [A]: {status_auto}  |  [Space]: Snap  [Q]: Done"
            
            # Top banner
            cv2.rectangle(disp, (10, 10), (w - 10, 50), (15, 15, 15), -1)
            cv2.rectangle(disp, (10, 10), (w - 10, 50), (0, 215, 255), 1)
            cv2.putText(disp, info, (22, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow("SmartEyes Dataset Collector", disp)
            k = cv2.waitKey(1) & 0xFF

            if k == ord('q') or k == ord('Q'):
                break
            elif k == 32:  # Space to capture single frame
                img_count += 1
                filename = os.path.join(SAVE_DIR, f"fish_{img_count:04d}.jpg")
                cv2.imwrite(filename, frame)
                print(f"  [SAVED] {filename} (Total: {img_count})")
            elif k == ord('a') or k == ord('A'):
                auto_capture = not auto_capture
                last_auto_time = now
                print(f"[AUTO-CAPTURE] {'ENABLED (capturing every 2s)' if auto_capture else 'DISABLED'}")

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"\n[DONE] Collected {img_count} images in '{SAVE_DIR}'.")
    print("Next step: Run 'python auto_annotate.py' to generate YOLO labels automatically!")

if __name__ == "__main__":
    run()
