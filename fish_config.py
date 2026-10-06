"""
=============================================================================
  fish_config.py  --  SmartEyes Tuning Parameters
=============================================================================
  Edit this file to tune the pipeline for your specific tank and camera.
  All values here OVERRIDE the defaults hardcoded in fish_monitor.py.
  Delete lines you want to keep at default values.
=============================================================================
"""

# ---------------------------------------------------------------------------
# VIDEO SOURCE
# ---------------------------------------------------------------------------
# 0        -> default webcam
# 1, 2 ... -> second webcam etc.
# "video.mp4" -> local video file (relative or absolute path)
SOURCE       = 0

# Display resolution (resize before inference -- does NOT affect capture res)
FRAME_WIDTH  = 1280
FRAME_HEIGHT = 720

# ---------------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------------
# "yolov8n.pt"        Nano -- fastest, smallest, auto-downloads from Ultralytics
# "yolov8s.pt"        Small -- better accuracy, ~2x slower than nano
# "yolov8m.pt"        Medium -- significantly better, needs decent GPU
# "fish_yolov8n.pt"   Custom aquarium weights (see README for download link)
MODEL_PATH   = "yolov8n.pt"

# Detection confidence gate (0.0-1.0).
# LOWER -> catches more fish but more false positives.
# HIGHER -> fewer false positives, may miss shy/dark fish.
CONF_THRESHOLD  = 0.35

# NMS IoU gate (0.0-1.0). Higher = allow more overlapping boxes.
IOU_THRESHOLD   = 0.45

# COCO class IDs to accept. None = accept all detected classes.
# Set to [0] if using custom fish-only weights (fish is class 0).
ALLOWED_CLASSES = None

# ---------------------------------------------------------------------------
# TRACKING
# ---------------------------------------------------------------------------
# Centroid history deque length (frames). Larger = smoother lethargy window
# but slightly more memory. Keep at ~30 for 30fps sources.
HISTORY_LEN = 30

# ---------------------------------------------------------------------------
# LETHARGY DETECTION
# ---------------------------------------------------------------------------
# A fish is Lethargic when it has moved fewer than LETHARGY_DISP_PX pixels
# (total cumulative path length) within the last LETHARGY_WINDOW_SEC seconds.
#
# HOW TO TUNE:
#   1. Measure your tank width in cm and in pixels from the camera image.
#      Example: tank is 60cm wide, appears as 900px -> scale = 60/900 = 0.067 cm/px
#   2. Decide what "stationary" means in real units. E.g., < 2cm in 5 seconds.
#   3. LETHARGY_DISP_PX = 2cm / 0.067 cm/px = ~30px
#   4. Raise LETHARGY_WINDOW_SEC if your fish are naturally slow swimmers.
#
LETHARGY_DISP_PX    = 15    # pixels -- default tuned for 1280x720 view of ~50cm tank
LETHARGY_WINDOW_SEC = 5     # seconds

# ---------------------------------------------------------------------------
# ERRATIC DETECTION
# ---------------------------------------------------------------------------
# A fish is Erratic when its speed OR acceleration exceeds these thresholds.
#
# HOW TO TUNE:
#   1. Record a normal swimming session and watch the "px/s" readout.
#   2. Note the typical maximum speed during calm swimming.
#   3. Set ERRATIC_SPEED_PX_S = 1.5x to 2x that baseline.
#   4. Similarly observe acceleration during direction changes.
#
# Example baseline for goldfish in a 50cm tank viewed at 1280px wide:
#   Calm cruising: ~40-60 px/s
#   Active play:   ~80-100 px/s
#   Stress darting:~150+ px/s  <-- set threshold here
#
ERRATIC_SPEED_PX_S  = 120   # px/s
ERRATIC_ACCEL_PX_S2 = 200   # px/s^2

# Number of consecutive frames a candidate state must persist before it is
# accepted. Prevents single-frame noise from flipping the label.
STATE_HOLD_FRAMES = 6

# ---------------------------------------------------------------------------
# FEEDER TRIGGER RULES
# ---------------------------------------------------------------------------
# Trigger feeder after a fish has been continuously Lethargic for this many seconds
FEED_ON_LETHARGY_SEC = 30   # seconds

# Don't re-trigger the feeder for the same fish within this cooldown window
FEED_COOLDOWN_SEC    = 120  # seconds (2 minutes)

# ---------------------------------------------------------------------------
# DISPLAY
# ---------------------------------------------------------------------------
SHOW_TRAILS = True   # draw motion trail behind each fish
TRAIL_LEN   = 20     # how many frames to show in the trail
