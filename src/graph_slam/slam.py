#!/usr/bin/env python

import argparse
import itertools
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import g2opy as g2o
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

import config
import icp
import pose_graph


def debug_log(msg):
    """Print debug message if DEBUG_LOGGING is enabled."""
    if config.DEBUG_LOGGING:
        print(f"[DEBUG] {msg}")


@dataclass
class Laser:
    angle_min: float
    angle_max: float
    angle_increment: float
    max_distance: float
    distances: list[float] | NDArray[float]


@dataclass
class Odometry:
    transform: NDArray[float]


@dataclass
class Reading:
    data: Laser | Odometry
    timestamp: float = field(default_factory=time.time)

    def __eq__(self, other):
        self.timestamp == other.timestamp

    def __lt__(self, other):
        self.timestamp < other.timestamp

    def __le__(self, other):
        (self < other) or (self == other)

    def __gt__(self, other):
        other < self

    def __ge__(self, other):
        other <= self


class SLAM:
    def __init__(self):
        self.optimizer = pose_graph.PoseGraphOptimization()
        self.vertex_counter = itertools.count(0)
        self.prev_odom = g2o.SE2(g2o.Isometry2d(np.eye(3)))
        self.optimizer.add_vertex(next(self.vertex_counter), self.prev_odom, True)
        self.last_vertex_idx = 0
        self.registered_scans = dict()

    def add_reading(self, reading: Reading):
        data = reading.data

        cleanup = False

        if isinstance(data, Laser):
            num_beams = len(data.distances)
            angles = np.linspace(data.angle_min, data.angle_max, num_beams)
            positions = (
                np.array([np.cos(angles), np.sin(angles)]).T
                * data.distances[:, np.newaxis]
            )
            mask = np.linalg.norm(positions, axis=1) < data.max_distance
            positions = positions[mask]

            if not len(self.registered_scans):
                self.registered_scans[self.last_vertex_idx] = positions
                return

            prev_idx = next(reversed(self.registered_scans.keys()))
            prev_scan = self.registered_scans[prev_idx]
            prev_pose = self.optimizer.get_pose(prev_idx)
            current_pose = self.optimizer.get_pose(self.last_vertex_idx)
            transform = prev_pose.inverse() * current_pose
            if (
                np.linalg.norm(transform.translation()) > config.SCAN_MIN_TRANSLATION
                or abs(transform.rotation().angle()) > config.SCAN_MIN_ROTATION
            ):
                # Try ICP with multiple previous scans
                best_result = None
                best_quality = float('inf')
                best_prev_idx = None
                
                # Get indices of recent scans to try (including the immediate previous)
                scan_indices = list(reversed(self.registered_scans.keys()))
                num_candidates = min(config.ICP_NUM_CANDIDATES, len(scan_indices))
                
                for i in range(num_candidates):
                    candidate_idx = scan_indices[i]
                    candidate_scan = self.registered_scans[candidate_idx]
                    candidate_pose = self.optimizer.get_pose(candidate_idx)
                    candidate_transform = candidate_pose.inverse() * current_pose
                    
                    transformation, distances, iter = icp.icp(
                        positions,
                        candidate_scan,
                        candidate_transform.to_isometry().matrix(),
                        max_iterations=config.ICP_MAX_ITERATIONS,
                        tolerance=config.ICP_TOLERANCE,
                    )

                    # Quality check
                    valid_distances = distances[distances < config.ICP_MAX_CORRESPONDENCE_DIST]
                    mean_dist = np.mean(valid_distances) if len(valid_distances) > 0 else float('inf')

                    if len(valid_distances) > config.SCAN_MIN_VALID_POINTS:
                        quality_score = mean_dist
                        
                        if quality_score < best_quality:
                            best_quality = quality_score
                            best_result = (candidate_idx, transformation, distances, iter)
                            best_prev_idx = candidate_idx
                        
                        debug_log(f"ICP candidate {i}: vertex {candidate_idx}, valid_pts={len(valid_distances)}, mean_dist={mean_dist:.4f}, iters={iter}")

                # Use the best result if it passes quality threshold
                if best_result is not None and best_quality < config.SCAN_QUALITY_THRESHOLD:
                    candidate_idx, transformation, distances, iter = best_result
                    valid_distances = distances[distances < config.ICP_MAX_CORRESPONDENCE_DIST]
                    
                    self.optimizer.add_edge(
                        [candidate_idx, self.last_vertex_idx],
                        g2o.SE2(g2o.Isometry2d(transformation)),
                        information=config.ICP_INFO_WEIGHT / best_quality * np.eye(3),
                    )
                    self.registered_scans[self.last_vertex_idx] = positions
                    cleanup = True
                    debug_log(f"ICP ACCEPTED: vertex {self.last_vertex_idx} matched to {best_prev_idx}, valid_pts={len(valid_distances)}, mean_dist={best_quality:.4f}, iters={iter}")

                    # Loop closure only makes sense if the last reading is a valid laser scan
                    self.loop_closure()
                else:
                    # Log why ICP was rejected
                    if best_result is None:
                        debug_log(f"ICP REJECTED: vertex {self.last_vertex_idx}, no valid candidates found")
                    else:
                        debug_log(f"ICP REJECTED: vertex {self.last_vertex_idx}, best quality {best_quality:.4f} exceeds threshold {config.SCAN_QUALITY_THRESHOLD}")

        elif isinstance(data, Odometry):
            current_odom = g2o.SE2(g2o.Isometry2d(data.transform))
            transform = self.prev_odom.inverse() * current_odom

            if (np.linalg.norm(transform.translation()) < np.finfo(float).eps) and (
                transform.rotation().angle() < np.finfo(float).eps
            ):
                return

            vertex_idx = next(self.vertex_counter)
            assert data.transform.shape == (3, 3)
            self.optimizer.add_vertex(vertex_idx, current_odom)
            self.optimizer.add_edge(
                [vertex_idx - 1, vertex_idx],
                transform,
                information=config.ODOM_INFO_WEIGHT * np.eye(3),
            )
            self.last_vertex_idx = vertex_idx
            self.prev_odom = current_odom
        else:
            raise ValueError(f"Unknown data type: {type(data)}")

        if self.last_vertex_idx > 1:
            # self.optimizer.set_verbose(True)
            self.optimizer.optimize()

            # Disabled: vertex cleanup was breaking the pose graph
            # if cleanup:
            #     keys = reversed(self.registered_scans.keys())
            #     end = next(keys)
            #     start = next(keys)
            #     for vertex_idx in range(start + 1, end):
            #         self.optimizer.remove_vertex(self.optimizer.vertex(vertex_idx))

    def loop_closure(self):
        if len(self.registered_scans) < 2:
            return

        scans = enumerate(reversed(self.registered_scans.items()))
        scan_idx, (current_idx, current_scan) = next(scans)
        assert current_idx == self.last_vertex_idx
        current_pose = self.optimizer.get_pose(current_idx)

        for prev_scan_idx, (prev_idx, prev_scan) in scans:
            if prev_scan_idx - scan_idx < config.LOOP_CLOSURE_MIN_SCANS:
                # Too recent
                continue

            vertex = self.optimizer.vertex(prev_idx)
            inv_hessian, valid = self.optimizer.compute_marginals(vertex)

            if not valid:
                continue

            prev_pose = vertex.estimate()
            position_diff = (prev_pose.inverse() * current_pose).translation()

            for index, block in enumerate(inv_hessian.block_cols()):
                if block:
                    cov = block[index]
                    break

            cov = np.linalg.inv(cov)
            position_cov = cov[:2, :2]

            mahalanobis_dist = np.sqrt(
                position_diff.T @ np.linalg.inv(position_cov) @ position_diff
            )

            if mahalanobis_dist < config.LOOP_CLOSURE_MAHAL_THRESHOLD:
                debug_log(f"LOOP CLOSURE CANDIDATE: current={current_idx} -> prev={prev_idx}, mahal_dist={mahalanobis_dist:.2f}")
                transformation, distances, iter = icp.icp(
                    prev_scan,
                    current_scan,
                    (prev_pose.inverse() * current_pose).to_isometry().matrix(),
                    max_iterations=config.ICP_MAX_ITERATIONS,
                    tolerance=config.ICP_TOLERANCE,
                )

                # Use consistent quality check with valid_distances
                valid_distances = distances[distances < config.ICP_MAX_CORRESPONDENCE_DIST]
                mean_dist = np.mean(valid_distances) if len(valid_distances) > 0 else float('inf')

                if len(valid_distances) > config.SCAN_MIN_VALID_POINTS and mean_dist < config.LOOP_CLOSURE_ICP_THRESHOLD:
                    rk = g2o.RobustKernelDCS(10)
                    self.optimizer.add_edge(
                        [current_idx, prev_idx],
                        g2o.SE2(g2o.Isometry2d(transformation)),
                        robust_kernel=rk,
                        information=config.LOOP_INFO_WEIGHT / mean_dist * np.eye(3),
                    )
                    debug_log(f"LOOP CLOSURE ADDED: {current_idx} -> {prev_idx}, mean_dist={mean_dist:.4f}, info_weight={config.LOOP_INFO_WEIGHT / mean_dist:.4f}")
                else:
                    debug_log(f"LOOP CLOSURE REJECTED: {current_idx} -> {prev_idx}, valid_pts={len(valid_distances)}, mean_dist={mean_dist:.4f}")


def homogeneous_transform(vector):
    x, y, theta = vector
    return np.array(
        [
            [np.cos(theta), -np.sin(theta), x],
            [np.sin(theta), np.cos(theta), y],
            [0.0, 0.0, 1.0],
        ]
    )


def load_data_from_file(data_file: Path):
    """
    Load data from file with format:
    - Column 0: delta_s (incremental distance traveled)
    - Column 1: delta_theta (incremental rotation)
    - Columns 2-122: 121 LIDAR measurements (-60 to 60 degrees)
    """
    data = np.loadtxt(data_file)
    readings = []
    x, y, theta = 0.0, 0.0, 0.0

    for row in data:
        delta_s = row[0]
        delta_theta = row[1]
        lidar_distances = row[2:123]

        # Update pose using odometry model:
        # dx/ds = cos(theta), dy/ds = sin(theta)
        x += delta_s * np.cos(theta)
        y += delta_s * np.sin(theta)
        theta += delta_theta

        readings.append({
            'odometry': homogeneous_transform([x, y, theta]),
            'laser': Laser(
                angle_min=np.radians(config.LIDAR_ANGLE_MIN),
                angle_max=np.radians(config.LIDAR_ANGLE_MAX),
                angle_increment=np.pi / 180,  # 1 degree step
                max_distance=config.LIDAR_MAX_RANGE,
                distances=lidar_distances,
            ),
        })

    return readings


parser = argparse.ArgumentParser(description="Python Graph SLAM")
parser.add_argument("--input", type=Path, help="Input data file.", required=True)
args = parser.parse_args()

# Create snapshots directory if enabled
if config.SAVE_SNAPSHOTS:
    snapshot_dir = Path(config.SNAPSHOT_DIR)
    snapshot_dir.mkdir(exist_ok=True)
    print(f"Snapshots will be saved to: {snapshot_dir.absolute()}")

# Setup matplotlib for interactive plotting
plt.ion()
fig, ax = plt.subplots(figsize=(10, 10))

slam = SLAM()
update_interval = config.VIS_UPDATE_INTERVAL

readings = load_data_from_file(args.input)

for idx, reading in enumerate(readings):
    # Process odometry first
    slam.add_reading(Reading(data=Odometry(transform=reading['odometry'])))
    # Then process laser
    slam.add_reading(Reading(data=reading['laser']))

    # Update visualization periodically
    if idx % update_interval == 0:
        ax.clear()

        # Collect all points and trajectory
        trajectory_x = []
        trajectory_y = []
        all_points_x = []
        all_points_y = []

        for vertex_idx, scan in slam.registered_scans.items():
            pose = slam.optimizer.get_pose(vertex_idx)
            rotation_matrix = pose.rotation().rotation_matrix()
            translation = pose.translation()

            # Transform scan points to world frame
            scan_world = (rotation_matrix @ scan.T + translation[:, np.newaxis]).T

            all_points_x.extend(scan_world[:, 0])
            all_points_y.extend(scan_world[:, 1])
            trajectory_x.append(translation[0])
            trajectory_y.append(translation[1])

        # Plot point cloud
        if all_points_x:
            ax.scatter(all_points_x, all_points_y, s=1, c='blue', alpha=0.5, label='Point Cloud')

        # Plot trajectory
        if trajectory_x:
            ax.plot(trajectory_x, trajectory_y, 'r-', linewidth=2, label='Trajectory')

        # Draw robot position as triangle
        if slam.last_vertex_idx >= 0:
            pose = slam.optimizer.get_pose(slam.last_vertex_idx)
            transform_matrix = pose.to_isometry().matrix()
            size = 0.3
            tip = transform_matrix @ np.array([size, 0.0, 1.0])
            left = transform_matrix @ np.array([-size * 0.5, -size * 0.5, 1.0])
            right = transform_matrix @ np.array([-size * 0.5, size * 0.5, 1.0])
            triangle = plt.Polygon(
                [[tip[0], tip[1]], [left[0], left[1]], [right[0], right[1]]],
                color='red', fill=True
            )
            ax.add_patch(triangle)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'SLAM - Reading {idx + 1}/{len(readings)}')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

        plt.pause(0.01)

        # Save snapshot for debugging
        if config.SAVE_SNAPSHOTS:
            snapshot_path = Path(config.SNAPSHOT_DIR) / f"frame_{idx+1:04d}.png"
            fig.savefig(snapshot_path, dpi=100, bbox_inches='tight')

plt.ioff()
plt.title(f'SLAM - Final Result ({len(readings)} readings)')
plt.show()