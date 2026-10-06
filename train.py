# -*- coding: utf-8 -*-
"""
SmartEyes - Custom Deep Learning Model Trainer
==============================================
Trains a YOLOv8 deep neural network specifically tuned to detect fish
in your aquarium while ignoring stones, gravel, and background clutter.

What this script does:
  1. Loads pretrained YOLOv8 backbone (Transfer Learning).
  2. Fine-tunes the network layers on your fish dataset (dataset/data.yaml).
  3. Automatically selects GPU (CUDA) if available, or multi-threaded CPU.
  4. Saves the best model weights to: weights/custom_fish_best.pt
  5. Generates loss and accuracy curves (runs/detect/fish_model/results.png).

How to run:
  python train.py
"""

import os
import sys
import shutil
import torch
from ultralytics import YOLO

DATA_YAML = os.path.join("dataset", "data.yaml")
BASE_MODEL = "yolov8n.pt"  # nano model: fast, light, accurate
PROJECT_DIR = "runs/detect"
RUN_NAME = "fish_model"
OUTPUT_DIR = "weights"

# Training Hyperparameters
EPOCHS = 25              # 20-30 epochs is ideal for fine-tuning
BATCH_SIZE = 8           # fits nicely in CPU or GPU memory
IMG_SIZE = 320           # 320px gives 40+ FPS performance on CPU

def run():
    print("=" * 65)
    print("      SmartEyes - YOLOv8 Deep Learning Model Training         ")
    print("=" * 65)

    if not os.path.exists(DATA_YAML):
        print(f"[ERROR] '{DATA_YAML}' not found!")
        print("Please run:")
        print("  1. python dataset_collector.py (snap 30-50 photos of your tank)")
        print("  2. python auto_annotate.py     (creates labels & data.yaml)")
        print("Then run python train.py again.")
        return

    # Check hardware
    has_cuda = torch.cuda.is_available()
    device = "0" if has_cuda else "cpu"
    print(f"[HARDWARE] Training Device: {'NVIDIA GPU (CUDA)' if has_cuda else 'CPU (Multi-threaded)'}")
    print(f"[DATASET]  Config: {os.path.abspath(DATA_YAML)}")
    print(f"[SETTINGS] Epochs: {EPOCHS} | Batch Size: {BATCH_SIZE} | Image Size: {IMG_SIZE}px\n")

    print(f"[INFO] Initializing base deep learning backbone: {BASE_MODEL} ...")
    model = YOLO(BASE_MODEL)

    print("\n[INFO] Starting neural network training...")
    print("Ultralytics will show loss and mAP progression for each epoch.\n")

    # Start training
    results = model.train(
        data=os.path.abspath(DATA_YAML),
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        device=device,
        project=PROJECT_DIR,
        name=RUN_NAME,
        exist_ok=True,
        verbose=True,
        plots=True
    )

    # Copy best weights to weights/
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    best_pt = os.path.join(results.save_dir, "weights", "best.pt")
    dest_pt = os.path.join(OUTPUT_DIR, "custom_fish_best.pt")

    if os.path.exists(best_pt):
        shutil.copy(best_pt, dest_pt)
        print("\n" + "=" * 50)
        print("          TRAINING COMPLETED SUCCESSFULLY!        ")
        print("=" * 50)
        print(f"[MODEL SAVED] Best model weights: {os.path.abspath(dest_pt)}")
        print(f"[PLOTS] Loss & accuracy curves: {os.path.abspath(os.path.join(PROJECT_DIR, RUN_NAME, 'results.png'))}")
        print("\nNext steps:")
        print("  1. Run 'python evaluate.py' to inspect your model's accuracy grade (mAP).")
        print("  2. Run 'python fish_tracker_deep.py' to launch deep tracking!")
    else:
        print(f"[WARN] Could not find {best_pt}")

if __name__ == "__main__":
    run()
