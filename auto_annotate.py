# -*- coding: utf-8 -*-
"""
SmartEyes - Automated Dataset Annotator & Formatter
===================================================
Automatically converts captured fish tank photos into a standard YOLOv8
training dataset with train/val splits and data.yaml configuration.

What this does:
  1. Reads all raw photos from dataset/raw/
  2. Uses our base detector to auto-generate bounding box annotations (.txt)
  3. Filters out bottom gravel/stones so only real fish are labeled.
  4. Automatically splits into 80% Training and 20% Validation sets.
  5. Generates dataset/data.yaml ready for instant training!

How to use:
  Run: python auto_annotate.py
"""

import os
import shutil
import random
import cv2
import numpy as np
from ultralytics import YOLO

RAW_DIR     = os.path.join("dataset", "raw")
DATASET_DIR = "dataset"
TRAIN_IMG   = os.path.join(DATASET_DIR, "train", "images")
TRAIN_LBL   = os.path.join(DATASET_DIR, "train", "labels")
VAL_IMG     = os.path.join(DATASET_DIR, "val", "images")
VAL_LBL     = os.path.join(DATASET_DIR, "val", "labels")

MODEL_PATH  = "aquarium_best.pt"

def is_white_stone(crop_bgr, y2, total_h) -> bool:
    if y2 < (0.70 * total_h):
        return False
    b_mean = float(np.mean(crop_bgr[:, :, 0]))
    g_mean = float(np.mean(crop_bgr[:, :, 1]))
    r_mean = float(np.mean(crop_bgr[:, :, 2]))
    is_bright_white = (b_mean > 110.0 and g_mean > 115.0 and r_mean > 115.0)
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    mean_s = float(np.mean(hsv[:, :, 1]))
    return is_bright_white and (mean_s < 38.0)

def is_purple_pump(crop_bgr) -> bool:
    if crop_bgr is None or crop_bgr.size == 0:
        return False
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    mean_h = float(np.mean(hsv[:, :, 0]))
    mean_s = float(np.mean(hsv[:, :, 1]))
    # Purple in OpenCV HSV: H around 125-165, S > 45
    return (125 <= mean_h <= 165) and (mean_s > 45)

def run():
    print("=" * 60)
    print("   SmartEyes - Automatic YOLO Dataset Generator   ")
    print("=" * 60)

    # Also include any user photos from brain directory if raw is empty
    sample_user_img = r"C:\Users\mithun\.gemini\antigravity-ide\brain\62fd2964-d7a9-4cb4-9415-17822f9530ba\.user_uploaded\media_1791221670094.jpg"
    os.makedirs(RAW_DIR, exist_ok=True)
    if os.path.exists(sample_user_img) and not os.path.exists(os.path.join(RAW_DIR, "sample_tank.jpg")):
        shutil.copy(sample_user_img, os.path.join(RAW_DIR, "sample_tank.jpg"))
        print("[INFO] Added user fish tank photo to raw dataset.")

    images = [f for f in os.listdir(RAW_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not images:
        print(f"[ERROR] No images found in '{RAW_DIR}'.")
        print("Please run 'python dataset_collector.py' first to take photos of your tank!")
        return

    print(f"[INFO] Found {len(images)} raw images to annotate.")
    print(f"[INFO] Loading teacher model: {MODEL_PATH} ...")
    model = YOLO(MODEL_PATH)

    # Clean & recreate target directories
    for p in [TRAIN_IMG, TRAIN_LBL, VAL_IMG, VAL_LBL]:
        os.makedirs(p, exist_ok=True)

    random.seed(42)
    random.shuffle(images)

    if len(images) == 1:
        train_files = set(images)
        val_files = set(images)
    else:
        split_idx = max(1, int(len(images) * 0.80))
        if split_idx >= len(images):
            split_idx = len(images) - 1
        train_files = set(images[:split_idx])
        val_files = set(images[split_idx:])

    total_fish_labeled = 0

    for idx, fname in enumerate(images):
        src_path = os.path.join(RAW_DIR, fname)
        img = cv2.imread(src_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        target_splits = []
        if fname in train_files:
            target_splits.append((TRAIN_IMG, TRAIN_LBL))
        if fname in val_files:
            target_splits.append((VAL_IMG, VAL_LBL))

        for target_img_dir, target_lbl_dir in target_splits:
            dst_img_path = os.path.join(target_img_dir, fname)
            shutil.copy(src_path, dst_img_path)

        # Detect fish with CLAHE contrast enhancement for dark fish
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        l_enh = clahe.apply(l)
        enhanced_img = cv2.cvtColor(cv2.merge((l_enh, a, b)), cv2.COLOR_LAB2BGR)

        results = model.predict(source=enhanced_img, conf=0.12, classes=[0], imgsz=416, verbose=False)[0]

        label_lines = []
        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                crop = img[y1:y2, x1:x2]

                # Reject stationary white stones & purple air pump
                if is_white_stone(crop, y2, h) or is_purple_pump(crop):
                    continue

                # Convert to normalized YOLO format: class x_center y_center width height
                bx = ((x1 + x2) / 2.0) / w
                by = ((y1 + y2) / 2.0) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h

                label_lines.append(f"0 {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f}")
                total_fish_labeled += 1

        # Write label file to all applicable splits
        base_name = os.path.splitext(fname)[0]
        for _, target_lbl_dir in target_splits:
            lbl_path = os.path.join(target_lbl_dir, f"{base_name}.txt")
            with open(lbl_path, "w") as fp:
                fp.write("\n".join(label_lines) + "\n")

    # Generate data.yaml
    yaml_content = f"""# SmartEyes Aquarium Fish Dataset
path: {os.path.abspath(DATASET_DIR).replace('\\', '/')}
train: train/images
val: val/images

# Classes
names:
  0: fish
"""
    yaml_path = os.path.join(DATASET_DIR, "data.yaml")
    with open(yaml_path, "w") as fp:
        fp.write(yaml_content)

    print("\n" + "=" * 50)
    print("      DATASET GENERATION SUMMARY      ")
    print("=" * 50)
    print(f"Total Images: {len(images)}")
    print(f"  - Train Set: {len(train_files)} images ({TRAIN_IMG})")
    print(f"  - Val Set  : {len(val_files)} images ({VAL_IMG})")
    print(f"Total Fish Labels: {total_fish_labeled}")
    print(f"Config File: {os.path.abspath(yaml_path)}")
    print("\nNext step: Run 'python train.py' to start training your custom Deep Learning model!")

if __name__ == "__main__":
    run()
