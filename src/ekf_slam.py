"""
EKF SLAM Module
Adapted from PythonRobotics EKF SLAM for use with odometry increments.

Reference: https://github.com/AtsushiSakai/PythonRobotics/blob/master/SLAM/EKFSLAM/ekf_slam.py
"""

import math
import numpy as np

import src.__config as GLOBALS

# State dimensions
STATE_SIZE = GLOBALS.EKF_STATE_SIZE  # [x, y, yaw]
LM_SIZE = GLOBALS.EKF_LM_SIZE        # [x, y] per landmark

# Noise covariances (from config)
# Process noise (motion uncertainty)
Cx = np.diag([
    GLOBALS.EKF_MOTION_NOISE_X,
    GLOBALS.EKF_MOTION_NOISE_Y,
    np.deg2rad(GLOBALS.EKF_MOTION_NOISE_THETA)
]) ** 2

# Measurement noise (landmark observation)
Q = np.diag([
    GLOBALS.EKF_MEASURE_NOISE_RANGE,
    np.deg2rad(GLOBALS.EKF_MEASURE_NOISE_BEARING)
]) ** 2

# Data association threshold (Mahalanobis distance)
M_DIST_TH = GLOBALS.EKF_MAHAL_THRESHOLD

# Maximum observation range
MAX_RANGE = GLOBALS.EKF_MAX_RANGE


def ekf_slam(xEst: np.ndarray, PEst: np.ndarray, u: np.ndarray, z: np.ndarray):
    """
    EKF SLAM algorithm - combined predict and update.

    Args:
        xEst: State estimate [x, y, theta, lm1_x, lm1_y, ...] shape (3+2*n_lm, 1)
        PEst: State covariance matrix shape (3+2*n_lm, 3+2*n_lm)
        u: Control input [delta_s, delta_theta] shape (2, 1)
        z: Observations [[range, bearing], ...] shape (n_obs, 2)

    Returns:
        xEst: Updated state estimate
        PEst: Updated covariance matrix
    """
    # Predict step
    xEst, PEst = ekf_predict(xEst, PEst, u)

    # Update step
    xEst, PEst = ekf_update(xEst, PEst, z)

    return xEst, PEst


def ekf_predict(xEst: np.ndarray, PEst: np.ndarray, u: np.ndarray):
    """
    EKF Predict step only.

    Args:
        xEst: State estimate
        PEst: State covariance
        u: Control input [delta_s, delta_theta]

    Returns:
        xEst, PEst: Predicted state and covariance
    """
    G, Fx = jacob_motion(xEst, u)
    xEst[0:STATE_SIZE] = motion_model(xEst[0:STATE_SIZE], u)
    PEst = G.T @ PEst @ G + Fx.T @ Cx @ Fx

    # Normalize angle
    xEst[2] = pi_2_pi(xEst[2])

    return xEst, PEst


def ekf_update(xEst: np.ndarray, PEst: np.ndarray, z: np.ndarray):
    """
    EKF Update step only.

    Args:
        xEst: State estimate (after predict)
        PEst: State covariance (after predict)
        z: Observations [[range, bearing], ...]

    Returns:
        xEst, PEst: Updated state and covariance
    """
    # Initial landmark covariance
    initP = np.eye(LM_SIZE)

    # Process each observation
    for iz in range(len(z)):
        obs = z[iz, 0:2]

        # Find corresponding landmark
        min_id = search_correspond_landmark_id(xEst, PEst, obs)

        nLM = calc_n_lm(xEst)

        if min_id == nLM:
            # New landmark - extend state
            xAug = np.vstack((xEst, calc_landmark_position(xEst, obs)))
            PAug = np.vstack((
                np.hstack((PEst, np.zeros((len(xEst), LM_SIZE)))),
                np.hstack((np.zeros((LM_SIZE, len(xEst))), initP))
            ))
            xEst = xAug
            PEst = PAug

        # Get landmark position and compute innovation
        lm = get_landmark_position_from_state(xEst, min_id)
        y, S, H = calc_innovation(lm, xEst, PEst, obs, min_id)

        # Kalman gain
        K = (PEst @ H.T) @ np.linalg.inv(S)

        # State update
        xEst = xEst + (K @ y)

        # Covariance update
        PEst = (np.eye(len(xEst)) - (K @ H)) @ PEst

    # Normalize angle
    xEst[2] = pi_2_pi(xEst[2])

    return xEst, PEst


def motion_model(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Motion model using odometry increments.

    Args:
        x: Robot state [x, y, theta] shape (3, 1)
        u: Control [delta_s, delta_theta] shape (2, 1)

    Returns:
        Updated state
    """
    F = np.array([
        [1.0, 0, 0],
        [0, 1.0, 0],
        [0, 0, 1.0]
    ])

    # Use increments directly (no DT multiplication)
    B = np.array([
        [math.cos(x[2, 0]), 0],
        [math.sin(x[2, 0]), 0],
        [0.0, 1.0]
    ])

    x = (F @ x) + (B @ u)
    return x


def calc_n_lm(x: np.ndarray) -> int:
    """Calculate number of landmarks in state vector."""
    n = int((len(x) - STATE_SIZE) / LM_SIZE)
    return n


def jacob_motion(x: np.ndarray, u: np.ndarray):
    """
    Compute Jacobian of motion model.

    Args:
        x: Full state vector
        u: Control input [delta_s, delta_theta]

    Returns:
        G: State transition Jacobian
        Fx: Noise mapping matrix
    """
    Fx = np.hstack((np.eye(STATE_SIZE), np.zeros((STATE_SIZE, LM_SIZE * calc_n_lm(x)))))

    # Jacobian of motion w.r.t. state (no DT)
    jF = np.array([
        [0.0, 0.0, -u[0, 0] * math.sin(x[2, 0])],
        [0.0, 0.0,  u[0, 0] * math.cos(x[2, 0])],
        [0.0, 0.0, 0.0]
    ], dtype=float)

    G = np.eye(len(x)) + Fx.T @ jF @ Fx

    return G, Fx


def calc_landmark_position(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Calculate landmark position from robot pose and observation.

    Args:
        x: Robot state [x, y, theta]
        z: Observation [range, bearing]

    Returns:
        Landmark position [x, y] shape (2, 1)
    """
    zp = np.zeros((2, 1))

    zp[0, 0] = x[0, 0] + z[0] * math.cos(x[2, 0] + z[1])
    zp[1, 0] = x[1, 0] + z[0] * math.sin(x[2, 0] + z[1])

    return zp


def get_landmark_position_from_state(x: np.ndarray, ind: int) -> np.ndarray:
    """Extract landmark position from state vector."""
    lm = x[STATE_SIZE + LM_SIZE * ind: STATE_SIZE + LM_SIZE * (ind + 1), :]
    return lm


def search_correspond_landmark_id(xAug: np.ndarray, PAug: np.ndarray, zi: np.ndarray) -> int:
    """
    Find corresponding landmark using Mahalanobis distance.

    Args:
        xAug: Full state vector
        PAug: Full covariance matrix
        zi: Observation [range, bearing]

    Returns:
        Index of corresponding landmark (or nLM if new landmark)
    """
    nLM = calc_n_lm(xAug)

    min_dist = []

    for i in range(nLM):
        lm = get_landmark_position_from_state(xAug, i)
        y, S, H = calc_innovation(lm, xAug, PAug, zi, i)

        # Mahalanobis distance
        mahal_dist = y.T @ np.linalg.inv(S) @ y
        min_dist.append(mahal_dist[0, 0])

    min_dist.append(M_DIST_TH)  # Threshold for new landmark

    min_id = min_dist.index(min(min_dist))

    return min_id


def calc_innovation(lm: np.ndarray, xEst: np.ndarray, PEst: np.ndarray,
                    z: np.ndarray, LMid: int):
    """
    Calculate innovation (measurement residual).

    Args:
        lm: Landmark position [x, y]
        xEst: State estimate
        PEst: State covariance
        z: Observation [range, bearing]
        LMid: Landmark index

    Returns:
        y: Innovation vector
        S: Innovation covariance
        H: Observation Jacobian
    """
    delta = lm - xEst[0:2]
    q = (delta.T @ delta)[0, 0]
    z_angle = math.atan2(delta[1, 0], delta[0, 0]) - xEst[2, 0]

    # Predicted observation
    zp = np.array([[math.sqrt(q), pi_2_pi(z_angle)]])

    # Innovation
    y = (z - zp).T
    y[1] = pi_2_pi(y[1])

    # Observation Jacobian
    H = jacob_h(q, delta, xEst, LMid + 1)

    # Innovation covariance
    S = H @ PEst @ H.T + Q

    return y, S, H


def jacob_h(q: float, delta: np.ndarray, x: np.ndarray, i: int) -> np.ndarray:
    """
    Compute observation Jacobian.

    Args:
        q: Squared distance to landmark
        delta: Vector from robot to landmark
        x: State vector
        i: Landmark index (1-based)

    Returns:
        H: Observation Jacobian
    """
    sq = math.sqrt(q)

    G = np.array([
        [-sq * delta[0, 0], -sq * delta[1, 0], 0, sq * delta[0, 0], sq * delta[1, 0]],
        [delta[1, 0], -delta[0, 0], -q, -delta[1, 0], delta[0, 0]]
    ])

    G = G / q

    nLM = calc_n_lm(x)

    F1 = np.hstack((np.eye(3), np.zeros((3, 2 * nLM))))
    F2 = np.hstack((
        np.zeros((2, 3)),
        np.zeros((2, 2 * (i - 1))),
        np.eye(2),
        np.zeros((2, 2 * nLM - 2 * i))
    ))

    F = np.vstack((F1, F2))

    H = G @ F

    return H


def pi_2_pi(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi
