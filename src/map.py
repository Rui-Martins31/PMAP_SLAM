import numpy as np
import matplotlib.pyplot as plt

import src.__config as GLOBALS

class Point:
    def __init__(self):
        # Position
        self.x: float = 0.0
        self.y: float = 0.0

class Map:
    def __init__(self):
        # Points
        self.points: list[tuple[Point]] = []

        # Landmarks
        self.landmarks: list[tuple[Point]] = []

    def compute_points_position(self, robot_homo_matrix: np.ndarray, lidar_ranges: np.ndarray, lidar_angles: np.ndarray) -> None:
        scan_points: list[Point] = []

        for angle, lidar_r in zip(lidar_angles, lidar_ranges):
            # Position in robot frame
            _angle_rad     = np.deg2rad(angle)
            _point_robot_x = lidar_r * np.cos(_angle_rad)
            _point_robot_y = lidar_r * np.sin(_angle_rad)

            # Transform to world frame
            _homo_point  = np.array([_point_robot_x, _point_robot_y, 1])
            _point_world = robot_homo_matrix @ _homo_point

            # Store Point
            point_world   = Point()
            point_world.x = _point_world[0]
            point_world.y = _point_world[1]

            scan_points.append(point_world)

        self.points.append(tuple(scan_points))

    def compute_corners(self):
        pass

    # Helpers
    def plot_map(self):
        plt.figure(figsize=(10, 8))

        for scan_points in self.points:
            x_coords = [point.x for point in scan_points]
            y_coords = [point.y for point in scan_points]
            plt.scatter(x_coords, y_coords, c='blue', s=1, alpha=0.5)

        plt.xlabel('X position (m)')
        plt.ylabel('Y position (m)')
        plt.title('Map Points')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(GLOBALS.PATH_OUTPUT + "odometry_map_points")
