import numpy as np

from src.lidar import Lidar
from src.map import Map, Point
from src.graph_slam_new import GraphSlam
import src.__config as GLOBALS


def transform_scan_to_world(
    ranges: np.ndarray,
    angles: np.ndarray,
    homo_matrix: np.ndarray
) -> list[Point]:
    """Transform LIDAR scan from robot frame to world frame."""
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


def get_homogeneous_matrix_from_pose(pose: np.ndarray) -> np.ndarray:
    """Create 3x3 SE(2) transformation matrix from [x, y, theta]."""
    x, y, theta = pose[0], pose[1], pose[2]
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    return np.array([
        [cos_t, -sin_t, x],
        [sin_t,  cos_t, y],
        [0.0,    0.0,   1.0]
    ])


def main():
    # Initialize components
    lidar           = Lidar()
    corner_detector = Map()
    graph_slam      = GraphSlam(corner_threshold=GLOBALS.GRAPH_SLAM_CORNER_THRESHOLD)

    num_scans = lidar.get_num_scans()
    print(f"Processing {num_scans} scans for Graph SLAM...")

    # Initialize graph with origin pose
    graph_slam.add_initial_pose()

    # Data collection phase
    for idx in range(num_scans):
        # Get odometry measurements
        delta_s, delta_theta = lidar.get_odometry(idx)

        # Add pose node and odometry constraint
        graph_slam.add_odometry(delta_s, delta_theta)

        # Get current pose estimate
        current_pose = graph_slam.get_current_pose()
        homo_matrix  = get_homogeneous_matrix_from_pose(current_pose)

        # Transform LIDAR scan to world frame
        scan_points = transform_scan_to_world(
            lidar.get_ranges(idx),
            lidar.get_angles(),
            homo_matrix
        )

        # Detect corners as observations
        observations = corner_detector.compute_corners_as_observations(
            scan_points,
            (current_pose[0], current_pose[1], current_pose[2])
        )

        # Add corner observations (convert polar to Cartesian)
        for obs in observations:
            distance, bearing, descriptor = obs
            x_local = distance * np.cos(bearing)
            y_local = distance * np.sin(bearing)
            graph_slam.add_corner_observation(x_local, y_local, current_pose, descriptor)

        # Progress output
        if (idx + 1) % 100 == 0 or idx == num_scans - 1:
            pose = graph_slam.get_current_pose()
            n_corners = len(graph_slam.corner_positions)
            n_obs = sum(len(v) for v in graph_slam.observations.values())
            print(f"  Scan {idx+1:4d}/{num_scans}: "
                  f"Pose=({pose[0]:6.2f}, {pose[1]:6.2f}, {np.degrees(pose[2]):6.1f}) | "
                  f"Corners: {n_corners} | Observations: {n_obs}")

    # Optimization phase
    graph_slam.optimize()
    graph_slam.print_statistics()

    # Visualization
    print("\nGenerating visualizations...")
    graph_slam.plot_results(save_path=GLOBALS.PATH_SLAM)

    print(f"\nVisualization saved to {GLOBALS.PATH_SLAM}")


if __name__ == "__main__":
    main()
