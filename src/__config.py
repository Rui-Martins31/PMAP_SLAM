import os

# Paths
PATH_DATASET: str = "data/data.txt"
PATH_OUTPUT: str  = "output/"
PATH_ODOMETRY: str = "output/odometry/"

os.makedirs(PATH_OUTPUT, exist_ok=True)
os.makedirs(PATH_ODOMETRY, exist_ok=True)

# Lidar
LIDAR_NUM_BEAMS: int  = 121
LIDAR_ANGLE_MIN: int  = -60
LIDAR_ANGLE_MAX: int  = 60
LIDAR_ANGLE_STEP: int = 1

# Debug
DEBUG: bool = False
NUM_EXAMPLE: int=300
TASK: int=[2]

# Corner Detection
CORNER_DETECT_ANGLE_THRESHOLD: float        = 45.0  # [degrees]
CORNER_DETECT_DISTANCE_THRESHOLD: float     = 1   # [m]
CORNER_DETECT_CLUSTER_JUMP_DISTANCE: float  = 0.2   # [m]