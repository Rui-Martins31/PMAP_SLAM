import numpy as np

from src.lidar import Lidar
from src.ekf_slam import EKFSlam
from src.graph_slam_new import GraphSlam
from src.map import Map, Point
import src.__config as GLOBALS


def transform_scan_to_world(
    ranges: np.ndarray,
    angles: np.ndarray,
    homo_matrix: np.ndarray
) -> list[Point]:
    """Transform LIDAR scan from robot frame to world frame."""
    points: list = []

    for angle_deg, range_val in zip(angles, ranges):
        angle_rad = np.deg2rad(angle_deg)
        robot_x = range_val * np.cos(angle_rad)
        robot_y = range_val * np.sin(angle_rad)

        homo_point = np.array([robot_x, robot_y, 1.0])
        world_point = homo_matrix @ homo_point

        point = Point()
        point.x = world_point[0]
        point.y = world_point[1]
        points.append(point)

    return points


def main():
    # Initialize components
    lidar = Lidar()
    ekf = EKFSlam()
    corner_detector = Map()

    num_scans = lidar.get_num_scans()
    print(f"=== Phase 1: EKF SLAM ({num_scans} scans) ===")

    # Storage for Graph SLAM initialization
    odometry_data = []
    observations_by_pose = {}  # pose_idx -> list of (distance, bearing, landmark_id)

    for idx in range(num_scans):
        # Get and store odometry
        delta_s, delta_theta = lidar.get_odometry(idx)
        odometry_data.append((delta_s, delta_theta))

        # EKF Prediction
        ekf.predict(delta_s, delta_theta)

        # Get current pose from EKF
        robot_pose = ekf.get_robot_pose()
        robot_H = ekf.get_homogeneous_matrix()

        # Transform scan to world frame
        scan_points = transform_scan_to_world(
            lidar.get_ranges(idx),
            lidar.get_angles(),
            robot_H
        )

        # Detect corners
        observations = corner_detector.compute_corners_as_observations(
            scan_points, robot_pose
        )

        # EKF Update and track associations
        pose_idx = idx + 1  # Pose 0 is initial, pose 1 is after first odometry
        pose_observations = []

        for obs in observations:
            distance, bearing, descriptor = obs
            z = np.array([[distance], [bearing]])

            # Get landmark association from EKF
            landmark_id = ekf._search_correspond_landmark_id(z)
            n_lm = ekf._calc_n_lm()

            # Check if this is a new landmark
            if landmark_id == n_lm:
                # New landmark will be added
                lm_pos = ekf._calc_landmark_position(z)
                ekf.xEst = np.vstack((ekf.xEst, lm_pos))

                n = len(ekf.xEst) - 2
                ekf.PEst = np.vstack((
                    np.hstack((ekf.PEst, np.zeros((n, 2)))),
                    np.hstack((np.zeros((2, n)), ekf.initP))
                ))

            # Get landmark and update
            lm = ekf._get_landmark_position(landmark_id)
            y, S, H = ekf._calc_innovation(lm, z, landmark_id)
            K = ekf.PEst @ H.T @ np.linalg.inv(S)
            ekf.xEst = ekf.xEst + K @ y
            ekf.PEst = (np.eye(len(ekf.xEst)) - K @ H) @ ekf.PEst

            # Store observation with landmark ID
            pose_observations.append((distance, bearing, landmark_id))

        ekf.xEst[2, 0] = ekf._pi_2_pi(ekf.xEst[2, 0])

        if pose_observations:
            observations_by_pose[pose_idx] = pose_observations

        # Progress
        if (idx + 1) % 100 == 0 or idx == num_scans - 1:
            x, y, theta = ekf.get_robot_pose()
            n_landmarks = len(ekf.get_landmarks())
            print(f"  Scan {idx+1:4d}/{num_scans}: "
                  f"Pose=({x:6.2f}, {y:6.2f}, {np.degrees(theta):6.1f}deg) | "
                  f"Landmarks: {n_landmarks}")

    print(f"\nEKF complete: {len(ekf.get_landmarks())} landmarks, "
          f"{sum(len(v) for v in observations_by_pose.values())} observations")

    # Phase 2: Graph SLAM smoothing
    print(f"\n=== Phase 2: Graph SLAM Smoothing ===")

    graph_slam = GraphSlam.from_ekf(ekf, odometry_data, observations_by_pose)

    print(f"  Initialized with {len(graph_slam.nodes_robot_pos)} poses, "
          f"{len(graph_slam.corner_positions)} landmarks")

    # Run optimization
    graph_slam.optimize()
    graph_slam.print_statistics()

    # Visualization
    print("\nGenerating visualizations...")
    graph_slam.plot_results(save_path=GLOBALS.PATH_SLAM)

    # Also save EKF trajectory for comparison
    ekf.plot_trajectory()

    print(f"\nVisualizations saved to {GLOBALS.PATH_SLAM}")


if __name__ == "__main__":
    main()
