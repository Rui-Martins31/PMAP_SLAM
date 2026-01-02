# config.py - SLAM Tuning Parameters
# Adjust these values to tune the SLAM algorithm performance

import numpy as np

# === LIDAR Settings ===
LIDAR_ANGLE_MIN = -60  # degrees
LIDAR_ANGLE_MAX = 60   # degrees
LIDAR_MAX_RANGE = 10   # meters

# === ICP Settings ===
ICP_MAX_ITERATIONS = 300
ICP_TOLERANCE = 0.00001
ICP_MAX_CORRESPONDENCE_DIST = 0.15 #0.15  # meters - reject point pairs beyond this distance
ICP_NUM_CANDIDATES = 5  # number of previous scans to try matching against

# === Scan Registration ===
SCAN_MIN_TRANSLATION = 0.05  # meters - minimum movement before registering new scan
SCAN_MIN_ROTATION = 0.001     # radians - minimum rotation before registering new scan
SCAN_QUALITY_THRESHOLD = 0.10  # meters - max mean ICP error to accept registration
SCAN_MIN_VALID_POINTS = 40    # minimum number of inlier points for valid ICP

# === Loop Closure ===
LOOP_CLOSURE_MIN_SCANS = 30   # skip this many recent scans when looking for loops
LOOP_CLOSURE_MAHAL_THRESHOLD = 0.9  # Mahalanobis distance threshold for candidates
LOOP_CLOSURE_ICP_THRESHOLD = 0.035   # max mean ICP error for accepting loop closure

# === Information Matrix Weights ===
ODOM_INFO_WEIGHT = 1.0        # weight for odometry edges
ICP_INFO_WEIGHT = 50.0       # weight multiplier for ICP scan matching edges
LOOP_INFO_WEIGHT = 0.05        # weight multiplier for loop closure edges

# === Visualization ===
VIS_UPDATE_INTERVAL = 1      # update plot every N readings

# === Debug/Snapshot Settings ===
SAVE_SNAPSHOTS = True        # save each frame as image for debugging
SNAPSHOT_DIR = "snapshots"   # directory to save snapshots
DEBUG_LOGGING = True         # print diagnostic info about ICP/loop closure
