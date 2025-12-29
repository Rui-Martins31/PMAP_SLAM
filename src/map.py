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
        self.cluster_line_error: float = GLOBALS.CORNER_DETECT_CLUSTER_LINE_ERROR     # [m^2]

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
        """Cluster -> Fit lines -> Find intersections -> Detect corners"""

        # Corner detection logic
        detected_corners = self._detect_corners(scan_points, use_clustering)

        # Update Landmarks
        for corner in detected_corners:
            self._associate_and_update(corner, scan_idx)

    def compute_corners_as_observations(self, scan_points: list[Point], robot_pose: tuple) -> list:
        """Detect corners and return them as robot-frame observations"""

        # Corner detection logic
        detected_corners = self._detect_corners(scan_points, self.use_clustering)

        # Convert to robot-frame observations
        observations = []
        rx, ry, rtheta = robot_pose

        for corner in detected_corners:
            dx = corner['x'] - rx
            dy = corner['y'] - ry

            # Distance from robot to corner
            distance = math.sqrt(dx**2 + dy**2)

            # Bearing angle (relative to robot forward axis)
            world_angle = math.atan2(dy, dx)
            bearing = world_angle - rtheta

            # Normalize bearing to [-pi, pi]
            bearing = (bearing + math.pi) % (2 * math.pi) - math.pi

            observations.append((
                distance,
                bearing,
                {
                    'angle': corner['angle'],
                    'median_dist': corner['median_dist'],
                    'variance': corner['variance'],
                    'confidence': corner['confidence']
                }
            ))

        return observations

    def _detect_corners(self, scan_points: list[Point], use_clustering: bool = True) -> list[dict]:
        """Cluster -> Fit lines -> Find intersections"""
        
        if len(scan_points) == 0:
            return []

        # Prepare clusters
        clusters: list[list[Point]] = []
        cluster_min_size: int = 5

        if use_clustering:
            # Clustering by line fitting error
            current_cluster = [scan_points[0]]

            for i in range(1, len(scan_points)):
                p_prev = scan_points[i-1]
                p_curr = scan_points[i]

                # Check for obvious distance jumps first
                jump_dist = math.sqrt((p_curr.x - p_prev.x)**2 + (p_curr.y - p_prev.y)**2)

                if jump_dist > self.cluster_jump_dist:
                    # Obvious gap - start new cluster
                    if len(current_cluster) >= cluster_min_size:
                        clusters.append(current_cluster)
                    current_cluster = [p_curr]
                    continue

                # Try adding the new point and check line fit error
                test_cluster = current_cluster + [p_curr]

                if len(test_cluster) >= 3:
                    line_params = self._fit_line_to_cluster(test_cluster)

                    if line_params is not None:
                        error = self._compute_line_mse(test_cluster, line_params)

                        if error > self.cluster_line_error:
                            # Error too high - this point belongs to a different wall
                            if len(current_cluster) >= cluster_min_size:
                                clusters.append(current_cluster)
                            current_cluster = [p_curr]
                        else:
                            current_cluster.append(p_curr)
                    else:
                        current_cluster.append(p_curr)
                else:
                    current_cluster.append(p_curr)

            # Append the last cluster
            if len(current_cluster) >= cluster_min_size:
                clusters.append(current_cluster)

        else:
            # Process entire scan as single cluster
            if len(scan_points) >= cluster_min_size:
                clusters = [scan_points]

        # Fit lines to clusters and keep track of both
        fitted_lines_with_clusters: list[dict] = []  # Each: {line: {a,b,c}, cluster: [Point]}

        for cluster in clusters:
            if len(cluster) >= cluster_min_size:
                line_params = self._fit_line_to_cluster(cluster)
                if line_params is not None:
                    fitted_lines_with_clusters.append({
                        'line': line_params,
                        'cluster': cluster
                    })

        # Find intersections between consecutive lines
        detected_corners: list = []

        for i in range(len(fitted_lines_with_clusters) - 1):
            line1_data = fitted_lines_with_clusters[i]
            line2_data = fitted_lines_with_clusters[i + 1]

            line1 = line1_data['line']
            line2 = line2_data['line']
            cluster1 = line1_data['cluster']
            cluster2 = line2_data['cluster']

            # Compute intersection
            intersection = self._compute_line_intersection(line1, line2)

            if intersection is not None:
                x, y = intersection

                # Compute angle at intersection
                angle_deg = self._compute_intersection_angle(line1, line2)

                # Compute descriptor metrics from cluster points
                median_dist, variance = self._compute_intersection_descriptors(
                    cluster1, cluster2, x, y
                )

                # Compute confidence score
                confidence = self._compute_corner_confidence(angle_deg, variance)

                # Check thresholds
                if angle_deg > self.angle_threshold and confidence > GLOBALS.CORNER_DETECT_CONFIDENCE_THRESHOLD:
                    detected_corners.append({
                        'x': x,
                        'y': y,
                        'angle': angle_deg,
                        'median_dist': median_dist,
                        'variance': variance,
                        'confidence': confidence
                    })

        return detected_corners

    def _fit_line_to_cluster(self, cluster: list[Point]) -> dict | None:
        """Fit a line to a cluster using least squares regression"""
        
        if len(cluster) < 2:
            return None
        
        # Extract coordinates
        x_vals = [p.x for p in cluster]
        y_vals = [p.y for p in cluster]
        
        # Compute means
        x_mean = sum(x_vals) / len(x_vals)
        y_mean = sum(y_vals) / len(y_vals)
        
        # Compute covariance terms
        cov_xx = sum((x - x_mean)**2 for x in x_vals) / len(x_vals)
        cov_yy = sum((y - y_mean)**2 for y in y_vals) / len(y_vals)
        cov_xy = sum((x_vals[i] - x_mean) * (y_vals[i] - y_mean) for i in range(len(x_vals))) / len(x_vals)
        
        # Compute eigenvalues and eigenvectors using characteristic equation
        # For 2x2 matrix, we use the formula for eigenvector of smaller eigenvalue
        trace = cov_xx + cov_yy
        det = cov_xx * cov_yy - cov_xy * cov_xy
        
        if det < 0:
            return None
        
        lambda2 = (trace - math.sqrt(trace**2 - 4*det)) / 2  # Smaller eigenvalue
        
        # Eigenvector corresponding to smaller eigenvalue (normal to line)
        if abs(cov_xy) > 1e-10:
            normal_x = cov_xy
            normal_y = lambda2 - cov_xx
        else:
            if cov_xx < cov_yy:
                normal_x = 1.0
                normal_y = 0.0
            else:
                normal_x = 0.0
                normal_y = 1.0
        
        # Normalize
        normal_len = math.sqrt(normal_x**2 + normal_y**2)
        if normal_len < 1e-10:
            return None
        
        a = normal_x / normal_len
        b = normal_y / normal_len
        c = -(a * x_mean + b * y_mean)
        
        return {'a': a, 'b': b, 'c': c}

    def _compute_line_mse(self, cluster: list[Point], line_params: dict) -> float:
        """Compute mean squared distance of points to the fitted line"""

        a, b, c = line_params['a'], line_params['b'], line_params['c']
        total_error = 0.0

        for p in cluster:
            # Distance from point to line (a,b already normalized so a²+b²=1)
            dist = abs(a * p.x + b * p.y + c)
            total_error += dist ** 2

        return total_error / len(cluster)

    def _compute_line_intersection(self, line1: dict, line2: dict) -> tuple[float, float] | None:
        """Compute intersection point of two lines"""

        a1, b1, c1 = line1['a'], line1['b'], line1['c']
        a2, b2, c2 = line2['a'], line2['b'], line2['c']
        
        # Solve system: a1*x + b1*y + c1 = 0
        #               a2*x + b2*y + c2 = 0
        det = a1 * b2 - a2 * b1
        
        if abs(det) < 1e-10:
            return None  # Lines are parallel
        
        x = (b1 * c2 - b2 * c1) / det
        y = (a2 * c1 - a1 * c2) / det
        
        return (x, y)

    def _compute_intersection_angle(self, line1: dict, line2: dict) -> float:
        """Compute angle between two intersecting lines in degrees"""

        a1, b1 = line1['a'], line1['b']
        a2, b2 = line2['a'], line2['b']
        
        # Direction vectors
        dir1_x, dir1_y = -b1, a1
        dir2_x, dir2_y = -b2, a2
        
        # Normalize
        len1 = math.sqrt(dir1_x**2 + dir1_y**2)
        len2 = math.sqrt(dir2_x**2 + dir2_y**2)
        
        if len1 < 1e-10 or len2 < 1e-10:
            return 0.0
        
        dir1_x /= len1
        dir1_y /= len1
        dir2_x /= len2
        dir2_y /= len2
        
        # Compute angle using dot product
        dot_product = dir1_x * dir2_x + dir1_y * dir2_y
        dot_product = max(-1.0, min(1.0, dot_product))
        
        angle_rad = math.acos(dot_product)
        angle_deg = math.degrees(angle_rad)
        
        # Return acute angle (0-90 degrees)
        if angle_deg > 90:
            angle_deg = 180 - angle_deg
        
        return angle_deg

    def _compute_intersection_descriptors(self, cluster1: list[Point], cluster2: list[Point], 
                                          corner_x: float, corner_y: float) -> tuple[float, float]:
        """Compute median distance and variance from cluster points to the intersection point"""

        # Combine points from both clusters
        all_points = cluster1 + cluster2
        
        if not all_points:
            return 0.0, 0.0
        
        # Compute distances from all points to intersection
        distances = []
        for point in all_points:
            d = math.sqrt((point.x - corner_x)**2 + (point.y - corner_y)**2)
            distances.append(d)
        
        if not distances:
            return 0.0, 0.0
        
        # Median
        distances_sorted = sorted(distances)
        median = distances_sorted[len(distances_sorted) // 2]
        
        # Variance
        mean = sum(distances) / len(distances)
        variance = sum((d - mean)**2 for d in distances) / len(distances)
        
        return median, variance

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

        print(f"[Map](plot_snapshots) Saving {int(len(self.points_world_frame)/steps)} snapshots...")

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
