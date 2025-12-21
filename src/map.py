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
    def __init__(self, x: float, y: float, scan_idx: int, angle: float = 0.0, 
                 median_dist: float = 0.0, variance: float = 0.0, confidence: float = 0.0):
        # Position in world frame
        self.x: float = x
        self.y: float = y

        # Corner descriptor properties
        self.angle: float       = angle          # Corner angle in degrees
        self.median_dist: float = median_dist    # Median distance to neighbors
        self.variance: float    = variance       # Variance of neighborhood distances
        self.confidence: float  = confidence     # Confidence score [0, 1]

        # Tracking information
        self.seen: list[int]        = [scan_idx]
        self.first_seen: int        = scan_idx
        self.last_seen: int         = scan_idx
        self.observation_count: int = 1

    def update(self, x: float, y: float, scan_idx: int, angle: float = None, 
               median_dist: float = None, variance: float = None, confidence: float = None):
        """Update landmark position with new observation"""
        
        # Average position
        self.x = (self.x * self.observation_count + x) / (self.observation_count + 1)
        self.y = (self.y * self.observation_count + y) / (self.observation_count + 1)

        # Update descriptor properties if provided (weighted average)
        if angle is not None:
            self.angle = (self.angle * self.observation_count + angle) / (self.observation_count + 1)
        if median_dist is not None:
            self.median_dist = (self.median_dist * self.observation_count + median_dist) / (self.observation_count + 1)
        if variance is not None:
            self.variance = (self.variance * self.observation_count + variance) / (self.observation_count + 1)
        if confidence is not None:
            self.confidence = max(self.confidence, confidence)  # Keep highest confidence

        # Update tracking
        self.last_seen          = scan_idx
        self.seen.append(scan_idx)
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

        self.use_clustering: bool      = GLOBALS.CORNER_DETECT_USE_CLUSTERING         # Enable clustering
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
            self.compute_corners(scan_points_world_frame, scan_idx, use_clustering=self.use_clustering)

    def compute_corners(self, scan_points: list[Point], scan_idx: int, use_clustering: bool = True):
        """Cluster -> Average -> Smooth -> Detect Angles"""
        
        # Prepare clusters
        clusters: list[list[Point]] = []
        cluster_min_size: int = 5

        if use_clustering:
            # Clustering (Split by distance jumps)
            current_cluster = [scan_points[0]]

            for i in range(1, len(scan_points)):
                p_prev = scan_points[i-1]
                p_curr = scan_points[i]

                # Calculate distance between consecutive points
                dist = math.sqrt((p_curr.x - p_prev.x)**2 + (p_curr.y - p_prev.y)**2)

                if dist > self.cluster_jump_dist:
                    # Start new cluster
                    if len(current_cluster) >= cluster_min_size:
                        clusters.append(current_cluster)
                    current_cluster = [p_curr]
                else:
                    current_cluster.append(p_curr)
            
            # Append the last cluster
            if len(current_cluster) >= cluster_min_size:
                clusters.append(current_cluster)

            ## DEBUG
            # print(f"[Map](compute_corners) Detected {len(clusters)} clusters (use_clustering={use_clustering}).")
        else:
            # Process entire scan as single cluster
            if len(scan_points) >= cluster_min_size:
                clusters = [scan_points]
            
            ## DEBUG
            # print(f"[Map](compute_corners) Processing entire scan as single cluster (use_clustering={use_clustering}).")

        # Compute cluster centroids
        cluster_centroids: list[tuple[float, float]] = []
        for cluster in clusters:
            if len(cluster) >= cluster_min_size:
                # Average x and y coordinates
                avg_x = sum(p.x for p in cluster) / len(cluster)
                avg_y = sum(p.y for p in cluster) / len(cluster)
                cluster_centroids.append((avg_x, avg_y))

        # If no centroids, return early
        if len(cluster_centroids) < 3:
            return

        # Detect Angles with descriptor computation
        detected_corners: list = []

        for i in range(1, len(cluster_centroids) - 1):
            prev_x, prev_y = cluster_centroids[i-1]
            curr_x, curr_y = cluster_centroids[i]
            next_x, next_y = cluster_centroids[i+1]

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

            # Compute corner descriptors from smoothed centroid neighborhood
            median_dist, variance = self._compute_centroid_neighborhood_descriptors(cluster_centroids, i)

            # Compute confidence score
            confidence = self._compute_corner_confidence(angle_deg, variance)

            # Check thresholds
            if angle_deg > self.angle_threshold and confidence > GLOBALS.CORNER_DETECT_CONFIDENCE_THRESHOLD:
                detected_corners.append({
                    'x': curr_x,
                    'y': curr_y,
                    'angle': angle_deg,
                    'median_dist': median_dist,
                    'variance': variance,
                    'confidence': confidence
                })

        # Update Landmarks
        for corner in detected_corners:
            self._associate_and_update(corner, scan_idx)

    def _compute_centroid_neighborhood_descriptors(self, centroids: list[tuple[float, float]], center_idx: int) -> tuple[float, float]:
        """Compute median distance and variance of neighborhood around a centroid point"""
        
        # Neighborhood radius (±2 centroids)
        start = max(0, center_idx - 2)
        end   = min(len(centroids), center_idx + 3)
        
        # Compute distances from center centroid
        center_x, center_y = centroids[center_idx]
        
        distances = []
        for j in range(start, end):
            if j != center_idx:
                cx, cy = centroids[j]
                d = math.sqrt((cx - center_x)**2 + (cy - center_y)**2)
                distances.append(d)
        
        if not distances:
            return 0.0, 0.0
        
        # Median
        distances_sorted = sorted(distances)
        median           = distances_sorted[len(distances_sorted) // 2]
        
        # Variance
        mean     = sum(distances) / len(distances)
        variance = sum((d - mean)**2 for d in distances) / len(distances)
        
        return median, variance

    def _compute_corner_confidence(self, angle: float, variance: float) -> float:
        """Compute confidence score based on angle and variance"""
        
        # Normalize angle
        angle_norm = min(1.0, angle / 90.0)
        
        # Variance component (lower is better)
        if variance < 0.05:
            variance_norm = 1.0
        elif variance < 0.2:
            variance_norm = 1.0 - (variance - 0.05) / 0.15
        else:
            variance_norm = max(0.0, 1.0 - variance / 0.2)
        
        # Combined score
        confidence = 0.6 * angle_norm + 0.4 * variance_norm
        
        return max(0.0, min(1.0, confidence))

    def _associate_and_update(self, corner: dict, scan_idx: int):
        """Associate detected corner with existing landmark or create new one"""

        best_landmark = None
        best_score    = float('inf')

        # Find landmark with best match
        for landmark in self.landmarks:
            # Spatial distance component
            spatial_dist = math.sqrt((landmark.x - corner['x'])**2 + (landmark.y - corner['y'])**2)
            
            # Only consider landmarks within distance threshold
            if spatial_dist > self.distance_threshold:
                continue
            
            # Angle difference component
            angle_diff = abs(landmark.angle - corner['angle'])
            
            # Combined score: 50% spatial distance, 50% angle similarity
            spatial_norm = spatial_dist / self.distance_threshold  # Normalize to [0,1]
            angle_norm   = min(1.0, angle_diff / 90.0)             # Normalize to [0,1]
            
            combined_score = 0.5 * spatial_norm + 0.5 * angle_norm
            
            if combined_score < best_score:
                best_score = combined_score
                best_landmark = landmark
        
        if best_landmark:
            best_landmark.update(
                corner['x'], 
                corner['y'], 
                scan_idx,
                angle=corner['angle'],
                median_dist=corner['median_dist'],
                variance=corner['variance'],
                confidence=corner['confidence']
            )
        else:
            self.landmarks.append(Landmark(
                corner['x'], 
                corner['y'], 
                scan_idx,
                angle=corner['angle'],
                median_dist=corner['median_dist'],
                variance=corner['variance'],
                confidence=corner['confidence']
            ))

    # Helpers
    def plot_snapshots(self, steps: int = 10):
        """Save each scan as a snapshot showing accumulated map points."""

        print(f"[Map](plot_snapshots) Saving {len(self.points_world_frame)/steps} snapshots...")

        for scan_idx in range(0, len(self.points_world_frame), steps):
            plt.figure(figsize=(10, 8))

            # Plot points
            scan_points = self.points_world_frame[scan_idx]
            x_coords    = [point.x for point in scan_points]
            y_coords    = [point.y for point in scan_points]
            plt.scatter(x_coords, y_coords, c='blue', s=1, alpha=0.5)

            # Plot landmarks from current scan only
            lx = [lm.x for lm in self.landmarks if scan_idx in lm.seen]
            ly = [lm.y for lm in self.landmarks if scan_idx in lm.seen]
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
