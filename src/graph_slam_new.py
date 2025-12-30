import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import least_squares

import src.__config as GLOBALS


class GraphSlam:

    def __init__(self, corner_threshold: float = 0.3):
        self.nodes_robot_pos: list = []
        self.odom_path: list = []
        self.odom_pose: np.ndarray = np.array([0.0, 0.0, 0.0])

        # Odometry constraints: (delta_s, delta_theta) for each step
        self.odometry_constraints: list = []

        # Observations: pose_idx -> list of (distance, angle, corner_id)
        self.observations: dict = {}
        self.corner_id_counter = 0
        self.corner_positions: dict = {}

        # Data association threshold
        self.corner_threshold = corner_threshold

        # Optimized results
        self.optimized_poses: np.ndarray = None

    def add_initial_pose(self) -> None:
        self.nodes_robot_pos.append(np.array([0.0, 0.0, 0.0]))
        self.odom_path.append(np.array([0.0, 0.0]))

    def add_odometry(self, delta_s: float, delta_theta: float) -> None:
        # Store odometry constraint
        self.odometry_constraints.append((delta_s, delta_theta))

        # Update odometry-only path
        self.odom_pose[2] += delta_theta
        self.odom_pose[0] += delta_s * np.cos(self.odom_pose[2])
        self.odom_pose[1] += delta_s * np.sin(self.odom_pose[2])
        self.odom_path.append(self.odom_pose[:2].copy())

        # Compute new pose estimate
        prev_pose = self.nodes_robot_pos[-1]
        new_theta = prev_pose[2] + delta_theta
        new_x = prev_pose[0] + delta_s * np.cos(new_theta)
        new_y = prev_pose[1] + delta_s * np.sin(new_theta)
        self.nodes_robot_pos.append(np.array([new_x, new_y, new_theta]))

    def get_current_pose(self) -> np.ndarray:
        return self.nodes_robot_pos[-1].copy()

    def add_corner_observation(self, x_local: float, y_local: float,
                                current_pose: np.ndarray, descriptor: dict = None) -> None:
        pose_idx = len(self.nodes_robot_pos) - 1

        # Convert to polar
        distance = math.sqrt(x_local**2 + y_local**2)
        angle = math.atan2(y_local, x_local)

        # Compute global position for data association
        theta = current_pose[2]
        x_global = current_pose[0] + x_local * np.cos(theta) - y_local * np.sin(theta)
        y_global = current_pose[1] + x_local * np.sin(theta) + y_local * np.cos(theta)

        # Data association
        corner_id = self._associate_corner(x_global, y_global)

        # Store observation
        if pose_idx not in self.observations:
            self.observations[pose_idx] = []
        self.observations[pose_idx].append((distance, angle, corner_id))

    def _associate_corner(self, x: float, y: float) -> int:
        best_id = None
        best_dist = float('inf')

        for cid, pos in self.corner_positions.items():
            dist = math.sqrt((x - pos[0])**2 + (y - pos[1])**2)
            if dist < best_dist:
                best_dist = dist
                best_id = cid

        if best_dist < self.corner_threshold:
            return best_id

        new_id = self.corner_id_counter
        self.corner_id_counter += 1
        self.corner_positions[new_id] = [x, y]
        return new_id

    def optimize(self) -> None:
        """Graph SLAM optimization using least squares."""
        print("\nGraph SLAM Optimization")

        n_poses = len(self.nodes_robot_pos)
        n_corners = len(self.corner_positions)

        if n_poses < 2:
            print("Not enough poses")
            return

        # Initialize state from odometry estimates
        x0 = np.zeros(n_poses * 3)
        for i, pose in enumerate(self.nodes_robot_pos):
            x0[i*3:i*3+3] = pose

        # Pre-compute odometry arrays
        odom_array = np.array(self.odometry_constraints)  # (n-1, 2): delta_s, delta_theta

        # Pre-compute loop closure edges
        corner_obs = {}
        for pose_idx, obs_list in self.observations.items():
            for (dist, angle, corner_id) in obs_list:
                if corner_id not in corner_obs:
                    corner_obs[corner_id] = []
                corner_obs[corner_id].append((pose_idx, dist, angle))

        # Build edge arrays - only keep edges between temporally distant poses
        MIN_POSE_GAP = 1
        MAX_EDGES_PER_CORNER = 50
        loop_edges = []
        for corner_id, obs_list in corner_obs.items():
            if len(obs_list) < 2:
                continue
            corner_edges = []
            for i in range(len(obs_list)):
                for j in range(i + 1, len(obs_list)):
                    p1, d1, a1 = obs_list[i]
                    p2, d2, a2 = obs_list[j]
                    if abs(p2 - p1) >= MIN_POSE_GAP:
                        corner_edges.append((p1, p2, d1, d2, a1, a2, abs(p2 - p1)))
            # Keep edges with largest pose gaps (most informative for loop closure)
            corner_edges.sort(key=lambda e: e[6], reverse=True)
            loop_edges.extend([e[:6] for e in corner_edges[:MAX_EDGES_PER_CORNER]])

        loop_edges = np.array(loop_edges) if loop_edges else np.empty((0, 6))
        n_loop_edges = len(loop_edges)

        print(f"  Poses: {n_poses}, Corners: {n_corners}")
        print(f"  Odometry edges: {len(odom_array)}, Loop closure edges: {n_loop_edges}")

        # Noise weights
        w_odom_xy = 1.0 / GLOBALS.GRAPH_SLAM_ODOM_NOISE_XY
        w_odom_theta = 1.0 / GLOBALS.GRAPH_SLAM_ODOM_NOISE_THETA
        w_obs = 1.0 / GLOBALS.GRAPH_SLAM_OBS_NOISE_DIST

        # Pre-compute sizes for array allocation
        n_odom = n_poses - 1
        n_res = 3 + 3 * n_odom + 2 * n_loop_edges

        # Pre-compute loop edge indices (avoid repeated casting)
        if n_loop_edges > 0:
            loop_p1 = loop_edges[:, 0].astype(int)
            loop_p2 = loop_edges[:, 1].astype(int)
            loop_d1, loop_d2 = loop_edges[:, 2], loop_edges[:, 3]
            loop_a1, loop_a2 = loop_edges[:, 4], loop_edges[:, 5]

        # Iteration counter for progress
        iteration_counter = [0]

        def residuals(x):
            x = x.reshape(-1, 3)  # (n_poses, 3)

            # Pre-allocate result array
            res = np.empty(n_res)

            # Anchor constraint (indices 0-2)
            res[0] = x[0, 0] * 10000
            res[1] = x[0, 1] * 10000
            res[2] = x[0, 2] * 10000

            # Odometry constraints (vectorized)
            t1 = x[:-1, 2]
            t1_after = t1 + odom_array[:, 1]
            x2_exp = x[:-1, 0] + odom_array[:, 0] * np.cos(t1_after)
            y2_exp = x[:-1, 1] + odom_array[:, 0] * np.sin(t1_after)

            idx = 3
            res[idx:idx + n_odom] = (x[1:, 0] - x2_exp) * w_odom_xy
            idx += n_odom
            res[idx:idx + n_odom] = (x[1:, 1] - y2_exp) * w_odom_xy
            idx += n_odom
            res[idx:idx + n_odom] = np.arctan2(
                np.sin(x[1:, 2] - t1_after),
                np.cos(x[1:, 2] - t1_after)
            ) * w_odom_theta
            idx += n_odom

            # Loop closure constraints (vectorized)
            if n_loop_edges > 0:
                cx1 = x[loop_p1, 0] + loop_d1 * np.cos(x[loop_p1, 2] + loop_a1)
                cy1 = x[loop_p1, 1] + loop_d1 * np.sin(x[loop_p1, 2] + loop_a1)
                cx2 = x[loop_p2, 0] + loop_d2 * np.cos(x[loop_p2, 2] + loop_a2)
                cy2 = x[loop_p2, 1] + loop_d2 * np.sin(x[loop_p2, 2] + loop_a2)

                res[idx:idx + n_loop_edges] = (cx1 - cx2) * w_obs
                idx += n_loop_edges
                res[idx:idx + n_loop_edges] = (cy1 - cy2) * w_obs

            # Progress tracking
            iteration_counter[0] += 1
            if iteration_counter[0] % 50 == 0:
                cost = np.sum(res ** 2)
                print(f"    Iteration {iteration_counter[0]:4d}: cost = {cost:.4f}")

            return res

        ## DEBUG ----
        # Compute initial cost breakdown
        init_res  = residuals(x0)
        iteration_counter[0] = 0  # Reset after initial evaluation
        anchor_cost = np.sum(init_res[0:3] ** 2)
        odom_cost = np.sum(init_res[3:3 + 3*n_odom] ** 2)
        loop_cost = np.sum(init_res[3 + 3*n_odom:] ** 2) if n_loop_edges > 0 else 0
        print(f"  Initial cost breakdown:")
        print(f"    Anchor:       {anchor_cost:.2f}")
        print(f"    Odometry:     {odom_cost:.2f}")
        print(f"    Loop closure: {loop_cost:.2f}")
        print(f"    TOTAL:        {anchor_cost + odom_cost + loop_cost:.2f}")
        ## DEBUG ----

        print("  Running optimization...")
        print(f"  Max nfev: {n_poses * GLOBALS.GRAPH_SLAM_MAX_ITERATIONS}")
        result = least_squares(residuals, x0, method='trf',
                               loss='soft_l1',
                               f_scale=1.2,
                               max_nfev=GLOBALS.GRAPH_SLAM_MAX_ITERATIONS * n_poses,
                            #    ftol=1e-6,
                            #    xtol=1e-6,
                               verbose=2)

        print(f"  Converged: {result.success}, Iterations: {result.nfev}")
        print(f"  Final cost: {result.cost:.4f}")

        # Final cost breakdown
        final_res = residuals(result.x)
        iteration_counter[0] = 0  # Reset
        anchor_cost = np.sum(final_res[0:3] ** 2)
        odom_cost = np.sum(final_res[3:3 + 3*n_odom] ** 2)
        loop_cost = np.sum(final_res[3 + 3*n_odom:] ** 2) if n_loop_edges > 0 else 0
        print(f"  Final cost breakdown:")
        print(f"    Anchor:       {anchor_cost:.2f}")
        print(f"    Odometry:     {odom_cost:.2f}")
        print(f"    Loop closure: {loop_cost:.2f}")

        # Store optimized poses
        self.optimized_poses = np.zeros((3, n_poses))
        for i in range(n_poses):
            self.optimized_poses[:, i] = result.x[i*3:i*3+3]

        print("Optimization completed")

    def _normalize_angle(self, angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    def get_optimized_trajectory(self) -> np.ndarray:
        if self.optimized_poses is None:
            return np.array(self.nodes_robot_pos)
        return self.optimized_poses.T

    def get_optimized_landmarks(self) -> np.ndarray:
        if not self.corner_positions:
            return np.array([]).reshape(0, 2)

        poses = self.get_optimized_trajectory()
        corner_estimates = {}

        for pose_idx, obs_list in self.observations.items():
            if pose_idx >= len(poses):
                continue
            pose = poses[pose_idx]
            x, y, theta = pose[0], pose[1], pose[2]

            for (dist, angle, corner_id) in obs_list:
                world_angle = theta + angle
                cx = x + dist * math.cos(world_angle)
                cy = y + dist * math.sin(world_angle)

                if corner_id not in corner_estimates:
                    corner_estimates[corner_id] = []
                corner_estimates[corner_id].append([cx, cy])

        corners = []
        for corner_id, estimates in corner_estimates.items():
            avg = np.mean(estimates, axis=0)
            corners.append(avg)

        return np.array(corners) if corners else np.array([]).reshape(0, 2)

    def get_odometry_path(self) -> np.ndarray:
        return np.array(self.odom_path)

    def plot_results(self, save_path: str = None) -> None:
        odom_path = self.get_odometry_path()
        opt_path = self.get_optimized_trajectory()
        corners = self.get_optimized_landmarks()

        plt.figure(figsize=(12, 10))

        plt.plot(odom_path[:, 0], odom_path[:, 1], 'r--', linewidth=2, label='Odometry only')
        plt.plot(opt_path[:, 0], opt_path[:, 1], 'b-', linewidth=2, label='Graph SLAM')

        if len(corners) > 0:
            plt.scatter(corners[:, 0], corners[:, 1], c='k', marker='x', s=100,
                       linewidths=2, label=f'Landmarks ({len(corners)})')

        plt.plot(0, 0, 'go', markersize=12, label='Start')
        plt.plot(odom_path[-1, 0], odom_path[-1, 1], 'r^', markersize=10, label='End (Odom)')
        plt.plot(opt_path[-1, 0], opt_path[-1, 1], 'b^', markersize=10, label='End (SLAM)')

        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        plt.title('Graph SLAM: Odometry vs Optimized')
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')

        if save_path:
            filepath = f"{save_path}/graph_slam_trajectory.png"
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {filepath}")

        plt.show()

    def print_statistics(self) -> None:
        print(f"\nTotal poses: {len(self.nodes_robot_pos)}")
        print(f"Total corners: {len(self.corner_positions)}")
        print(f"Total observations: {sum(len(v) for v in self.observations.values())}")

        if self.optimized_poses is not None:
            odom_end = self.odom_path[-1]
            opt_end = self.optimized_poses[:2, -1]
            drift = np.sqrt((odom_end[0] - opt_end[0])**2 + (odom_end[1] - opt_end[1])**2)
            print(f"Drift correction: {drift:.3f} m")
