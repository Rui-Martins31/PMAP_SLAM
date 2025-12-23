import numpy as np

from src.lidar import Lidar
from src.ekf_slam import EKFSlam
from src.map import Map, Point
import src.__config as GLOBALS


def transform_scan_to_world(
    ranges: np.ndarray,
    angles: np.ndarray,
    homo_matrix: np.ndarray
) -> list[Point]:
    """Transform LIDAR scan from robot frame to world frame"""
    points: list = []

    for angle_deg, range_val in zip(angles, ranges):
        # Convert to robot frame coordinates
        angle_rad = np.deg2rad(angle_deg)
        robot_x = range_val * np.cos(angle_rad)
        robot_y = range_val * np.sin(angle_rad)

        # Transform to world frame
        homo_point = np.array([robot_x, robot_y, 1.0])
        world_point = homo_matrix @ homo_point

        # Create Point object
        point = Point()
        point.x = world_point[0]
        point.y = world_point[1]
        points.append(point)

    return points


def main():
    # Initialize components
    lidar           = Lidar()
    ekf             = EKFSlam()
    corner_detector = Map()

    num_scans = lidar.get_num_scans()
    print(f"Processing {num_scans} scans...")

    # Statistics tracking
    total_observations = 0
    new_landmarks_count = 0

    for idx in range(num_scans):
        # Get odometry measurements
        delta_s, delta_theta = lidar.get_odometry(idx)

        # EKF Prediction step
        ekf.predict(delta_s, delta_theta)

        # Get current robot pose estimate from EKF
        robot_pose = ekf.get_robot_pose()
        robot_H = ekf.get_homogeneous_matrix()

        # Transform LIDAR scan to world frame using EKF pose
        scan_points = transform_scan_to_world(
            lidar.get_ranges(idx),
            lidar.get_angles(),
            robot_H
        )

        # Store scan points for visualization
        ekf.store_scan_points(scan_points)

        # Detect corners and convert to observations
        observations = corner_detector.detect_corners_as_observations(
            scan_points, robot_pose
        )

        # Track statistics
        n_lm_before = len(ekf.get_landmarks())

        # EKF Update step with observations
        if observations:
            ekf.update(observations)
            total_observations += len(observations)

        n_lm_after = len(ekf.get_landmarks())
        new_landmarks_count += (n_lm_after - n_lm_before)

        # Progress output
        if (idx + 1) % 100 == 0 or idx == num_scans - 1:
            x, y, theta = ekf.get_robot_pose()
            n_landmarks = len(ekf.get_landmarks())
            print(f"  Scan {idx+1:4d}/{num_scans}: "
                  f"Pose=({x:6.2f}, {y:6.2f}, {np.degrees(theta):6.1f}°) | "
                  f"Landmarks: {n_landmarks}")

    # Final statistics
    print("EKF SLAM:")
    print(f"  Final robot pose: {ekf.get_robot_pose()}")
    print(f"  Total landmarks: {len(ekf.get_landmarks())}")
    print(f"  Total observations processed: {total_observations}")

    # Visualization
    print("Generating visualizations...")
    ekf.plot_trajectory()
    ekf.plot_landmarks()
    ekf.plot_map()
    print(f"Visualizations saved to {GLOBALS.PATH_SLAM}")


if __name__ == "__main__":
    main()
