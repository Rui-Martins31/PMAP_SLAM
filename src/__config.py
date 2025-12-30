import os

# Paths
PATH_DATASET: str  = "data/data.txt"
PATH_OUTPUT: str   = "output/"
PATH_ODOMETRY: str = "output/odometry/"
PATH_SLAM: str     = "output/slam"

os.makedirs(PATH_OUTPUT, exist_ok=True)
os.makedirs(PATH_ODOMETRY, exist_ok=True)
os.makedirs(PATH_SLAM, exist_ok=True)


# Lidar
LIDAR_NUM_BEAMS: int  = 121
LIDAR_ANGLE_MIN: int  = -60
LIDAR_ANGLE_MAX: int  = 60
LIDAR_ANGLE_STEP: int = 1

# Debug
DEBUG: bool = False

# Corner Detection for odometry
CORNER_DETECT_ANGLE_THRESHOLD: float        = 30.0  # [degrees]
CORNER_DETECT_DISTANCE_THRESHOLD: float     = 0.001 # [m]

CORNER_DETECT_USE_CLUSTERING: bool          = True
CORNER_DETECT_CLUSTER_JUMP_DISTANCE: float  = 0.2   # [m]
CORNER_DETECT_CLUSTER_LINE_ERROR: float     = 0.01  # [m^2]

CORNER_DETECT_CONFIDENCE_THRESHOLD: float   = 0.5

# EKF SLAM Parameters
EKF_STATE_SIZE: int = 3      # Robot state: x, y, theta
EKF_LM_SIZE: int    = 2      # Landmark state: x, y

# Process noise
EKF_SIGMA_DELTA_S: float     = 0.15  # [m]
EKF_SIGMA_DELTA_THETA: float = 0.08  # [rad]

# Measurement noise
EKF_SIGMA_RANGE: float   = 0.08      # [m]
EKF_SIGMA_BEARING: float = 0.04      # [rad]

# Data association
EKF_MAHALANOBIS_THRESHOLD: float = 4.0  #2.5

# Initial covariance
EKF_INITIAL_P: float = 0.01

# Graph SLAM Parameters
GRAPH_SLAM_ODOM_NOISE_XY: float    = 1.8       # [m] 0.5
GRAPH_SLAM_ODOM_NOISE_THETA: float = 1.5      # [rad] 0.08 
GRAPH_SLAM_OBS_NOISE_DIST: float   = 2.5         # [m] 0.7
GRAPH_SLAM_OBS_NOISE_ANGLE: float  = 2      # [rad] 0.08
GRAPH_SLAM_MAX_ITERATIONS: int     = 1
GRAPH_SLAM_CORNER_THRESHOLD: float = 1.5        # [m] 1.5