import numpy as np
import matplotlib.pyplot as plt
import math

import src.__config as GLOBALS

class Point:
    def __init__(self):
        # Position
        self.x: float = 0.0
        self.y: float = 0.0

class Observation: 
    def __init__(self, distance: float, angle: float, robot_homo_matrix: np.ndarray):
        
        # Observation
        self.dist: float  = distance
        self.angle: float = angle

        # Robot transformation
        self.homo_matrix: np.ndarray = robot_homo_matrix

class Landmark:
    def __init__(self, x: float, y: float, scan_idx: int):
        # Position in world frame
        self.x: float = x
        self.y: float = y

        # Tracking information
        self.first_seen: int        = scan_idx
        self.last_seen: int         = scan_idx
        self.observation_count: int = 1

    def update(self, x: float, y: float, scan_idx: int):
        """Update landmark position with new observation"""
        
        # Average
        self.x = (self.x * self.observation_count + x) / (self.observation_count + 1)
        self.y = (self.y * self.observation_count + y) / (self.observation_count + 1)

        # Update
        self.last_seen          = scan_idx
        self.observation_count += 1

class Map:
    def __init__(self):
        # Points
        self.points_world_frame: list[tuple[Point]] = []          # x, y
        self.points_robot_frame: list[tuple[Observation]] = []    # dist, angle

        # Landmarks
        self.landmarks: list[Landmark] = []

        # Corner detection parameters
        self.angle_threshold: float    = GLOBALS.CORNER_DETECT_ANGLE_THRESHOLD        # [degrees]
        self.distance_threshold: float = GLOBALS.CORNER_DETECT_DISTANCE_THRESHOLD     # [m]
        self.cluster_jump_dist: float  = GLOBALS.CORNER_DETECT_CLUSTER_JUMP_DISTANCE  # [m]

    def compute_points_position(self, robot_homo_matrix: np.ndarray, lidar_ranges: np.ndarray, lidar_angles: np.ndarray, scan_idx: int) -> None:
        scan_points_world_frame: list[Point] = []
        scan_points_robot_frame: list[Observation] = []

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

            scan_points_world_frame.append(point_world)
            scan_points_robot_frame.append(Observation(lidar_r, angle, robot_homo_matrix))

        # Update
        self.points_world_frame.append(tuple(scan_points_world_frame))
        self.points_robot_frame.append(tuple(scan_points_robot_frame))

        # Detect corners
        if len(scan_points_world_frame) > 0:
            self.compute_corners(scan_points_world_frame, scan_idx)

    def compute_corners(self, scan_points: list[Point], scan_idx: int):
        """Cluster -> Smooth -> Detect Angles"""
        
        # Clustering (Split by distance jumps)
        clusters = []
        current_cluster = [scan_points[0]]

        for i in range(1, len(scan_points)):
            p_prev = scan_points[i-1]
            p_curr = scan_points[i]

            # Calculate distance between consecutive points
            dist = math.sqrt((p_curr.x - p_prev.x)**2 + (p_curr.y - p_prev.y)**2)

            if dist > self.cluster_jump_dist:
                # Start new cluster
                if len(current_cluster) > 5:
                    clusters.append(current_cluster)
                current_cluster = [p_curr]
            else:
                current_cluster.append(p_curr)
        
        # Append the last cluster
        if len(current_cluster) > 5:
            clusters.append(current_cluster)

        # Smoothing and Detection per cluster
        detected_corners = []

        for cluster in clusters:
            # We need at least 3 points to form an angle
            if len(cluster) < 5: 
                continue

            # Manual Smoothing (Simple Moving Average)
            smoothed_coords = []
            window_size     = 3
            
            for i in range(len(cluster)):
                # Handle edges by clamping
                start_idx = max(0, i - 1)
                end_idx   = min(len(cluster), i + 2)
                
                # Sum x and y
                sum_x, sum_y = 0.0, 0.0
                count = 0
                for j in range(start_idx, end_idx):
                    sum_x += cluster[j].x
                    sum_y += cluster[j].y
                    count += 1
                
                smoothed_coords.append((sum_x / count, sum_y / count))

            # Detect Angles
            for i in range(1, len(smoothed_coords) - 1):
                prev_x, prev_y = smoothed_coords[i-1]
                curr_x, curr_y = smoothed_coords[i]
                next_x, next_y = smoothed_coords[i+1]

                # Vector 1: Prev -> Curr
                v1_x = curr_x - prev_x
                v1_y = curr_y - prev_y
                
                # Vector 2: Curr -> Next
                v2_x = next_x - curr_x
                v2_y = next_y - curr_y

                # Magnitudes
                mag_v1 = math.sqrt(v1_x**2 + v1_y**2)
                mag_v2 = math.sqrt(v2_x**2 + v2_y**2)

                if mag_v1 == 0 or mag_v2 == 0:
                    continue

                # Dot product
                dot_product = (v1_x * v2_x) + (v1_y * v2_y)
                
                # Cosine rule
                cos_angle = max(-1.0, min(1.0, dot_product / (mag_v1 * mag_v2)))
                angle_rad = math.acos(cos_angle)
                angle_deg = math.degrees(angle_rad)

                # Check threshold
                if angle_deg > self.angle_threshold:
                    detected_corners.append((curr_x, curr_y))

        # Update Landmarks
        for (cx, cy) in detected_corners:
            self._associate_and_update(cx, cy, scan_idx)

    def _associate_and_update(self, x: float, y: float, scan_idx: int):
        best_landmark = None
        min_dist = self.distance_threshold

        # Find nearest
        for landmark in self.landmarks:
            d = math.sqrt((landmark.x - x)**2 + (landmark.y - y)**2)
            if d < min_dist:
                min_dist = d
                best_landmark = landmark
        
        if best_landmark:
            best_landmark.update(x, y, scan_idx)
        else:
            self.landmarks.append(Landmark(x, y, scan_idx))

    # Helpers
    def plot_snapshots(self, steps: int = 10):
        """Save each scan as a snapshot showing accumulated map points."""

        print(f"[Map](plot_snapshots) Saving {len(self.points_world_frame)} snapshots...")

        for scan_idx in range(0, len(self.points_world_frame), steps):
            plt.figure(figsize=(10, 8))

            # Plot points
            scan_points = self.points_world_frame[scan_idx]
            x_coords    = [point.x for point in scan_points]
            y_coords    = [point.y for point in scan_points]
            plt.scatter(x_coords, y_coords, c='blue', s=1, alpha=0.5)

            # Plot landmarks from current scan only
            lx = [lm.x for lm in self.landmarks if lm.first_seen == scan_idx]
            ly = [lm.y for lm in self.landmarks if lm.first_seen == scan_idx]
            if lx:
                plt.scatter(lx, ly, c='red', s=50, marker='x', label='Corners')

            plt.xlabel('X position (m)')
            plt.ylabel('Y position (m)')
            plt.title(f'Odometry Snapshot {scan_idx:04d}')
            plt.grid(True, alpha=0.3)
            plt.axis('equal')
            plt.tight_layout()
            plt.savefig(f"{GLOBALS.PATH_ODOMETRY}snapshot_{scan_idx:04d}.png")
            plt.close()

        print(f"[Map](plot_snapshots) Snapshots saved to {GLOBALS.PATH_ODOMETRY}")

    def plot_map(self):

        # Statistics
        print(f"[Map](plot_map) Detected {len(self.landmarks)} corners")

        # Plot
        plt.figure(figsize=(10, 8))

        for scan_points in self.points_world_frame:
            x_coords = [point.x for point in scan_points]
            y_coords = [point.y for point in scan_points]
            plt.scatter(x_coords, y_coords, c='blue', s=1, alpha=0.5)

        # Plot Landmarks
        lx = [lm.x for lm in self.landmarks]
        ly = [lm.y for lm in self.landmarks]
        plt.scatter(lx, ly, c='red', s=50, marker='x', label='Corners')

        plt.xlabel('X position (m)')
        plt.ylabel('Y position (m)')
        plt.title('Map Points')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(GLOBALS.PATH_OUTPUT + "odometry_map_points")
