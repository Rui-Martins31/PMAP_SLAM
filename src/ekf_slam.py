import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from src.__config import (
    EKF_STATE_SIZE,
    EKF_LM_SIZE,
    EKF_SIGMA_DELTA_S,
    EKF_SIGMA_DELTA_THETA,
    EKF_SIGMA_RANGE,
    EKF_SIGMA_BEARING,
    EKF_MAHALANOBIS_THRESHOLD,
    EKF_INITIAL_P,
    PATH_SLAM,
)


class EKFSlam:

    def __init__(self):
        # State vector [x, y, theta]
        self.xEst = np.zeros((EKF_STATE_SIZE, 1))

        # Covariance matrix
        self.PEst = np.eye(EKF_STATE_SIZE) * EKF_INITIAL_P

        # Process noise covariance
        self.Q = np.diag([
            EKF_SIGMA_DELTA_S ** 2,
            EKF_SIGMA_DELTA_S ** 2,
            EKF_SIGMA_DELTA_THETA ** 2
        ])

        # Measurement noise covariance
        self.R = np.diag([
            EKF_SIGMA_RANGE ** 2,
            EKF_SIGMA_BEARING ** 2
        ])

        # Initial covariance for new landmarks
        self.initP = np.eye(EKF_LM_SIZE) * 1.0

        # Mahalanobis distance threshold for new landmark
        self.mahal_threshold = EKF_MAHALANOBIS_THRESHOLD

        # History for visualization
        self.history_x = [0.0]
        self.history_y = [0.0]
        self.history_theta = [0.0]

        # Store scan points for visualization
        self.scan_points_history: list[list[tuple]] = []

    def predict(self, delta_s: float, delta_theta: float):
        """EKF prediction step using odometry"""
        
        u = np.array([[delta_s], [delta_theta]])

        # Compute Jacobians
        G, Fx = self._jacob_motion(self.xEst, u)

        # Predict state
        self.xEst[0:EKF_STATE_SIZE] = self._motion_model(
            self.xEst[0:EKF_STATE_SIZE], u
        )

        # Predict covariance
        self.PEst = G @ self.PEst @ G.T + Fx.T @ self.Q @ Fx

        # Normalize angle
        self.xEst[2, 0] = self._pi_2_pi(self.xEst[2, 0])

        # Store history
        self.history_x.append(self.xEst[0, 0])
        self.history_y.append(self.xEst[1, 0])
        self.history_theta.append(self.xEst[2, 0])

    def update(self, observations: list):
        """EKF update step with corner observations."""

        for obs in observations:
            distance, bearing, descriptor = obs
            z = np.array([[distance], [bearing]])

            # Data association using Mahalanobis distance
            min_id = self._search_correspond_landmark_id(z)

            n_lm   = self._calc_n_lm()

            if min_id == n_lm:
                # New landmark - augment state
                lm_pos    = self._calc_landmark_position(z)
                self.xEst = np.vstack((self.xEst, lm_pos))

                # Augment covariance
                n = len(self.xEst) - EKF_LM_SIZE
                self.PEst = np.vstack((
                    np.hstack((self.PEst, np.zeros((n, EKF_LM_SIZE)))),
                    np.hstack((np.zeros((EKF_LM_SIZE, n)), self.initP))
                ))

            # Get landmark position
            lm = self._get_landmark_position(min_id)

            # Calculate innovation
            y, S, H = self._calc_innovation(lm, z, min_id)

            # Kalman gain
            K = self.PEst @ H.T @ np.linalg.inv(S)

            # Update state
            self.xEst = self.xEst + K @ y

            # Update covariance
            self.PEst = (np.eye(len(self.xEst)) - K @ H) @ self.PEst

        # Normalize angle
        self.xEst[2, 0] = self._pi_2_pi(self.xEst[2, 0])

    def get_robot_pose(self) -> tuple:
        """Return current robot pose estimate."""
        return (
            self.xEst[0, 0],
            self.xEst[1, 0],
            self.xEst[2, 0]
        )

    def get_homogeneous_matrix(self) -> np.ndarray:
        """Return 3x3 homogeneous transformation matrix."""
        x, y, theta = self.get_robot_pose()
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        return np.array([
            [cos_t, -sin_t, x],
            [sin_t,  cos_t, y],
            [0.0,    0.0,   1.0]
        ])

    def get_landmarks(self) -> list:
        """Return list of landmark positions as dicts."""
        n_lm = self._calc_n_lm()
        landmarks = []
        for i in range(n_lm):
            lm = self._get_landmark_position(i)
            landmarks.append({
                'x': lm[0, 0],
                'y': lm[1, 0],
                'id': i
            })
        return landmarks

    def get_robot_covariance(self) -> np.ndarray:
        """Return 3x3 robot pose covariance."""
        return self.PEst[0:EKF_STATE_SIZE, 0:EKF_STATE_SIZE]

    def get_landmark_covariance(self, lm_id: int) -> np.ndarray:
        """Return 2x2 covariance for a specific landmark."""
        start = EKF_STATE_SIZE + EKF_LM_SIZE * lm_id
        end = start + EKF_LM_SIZE
        return self.PEst[start:end, start:end]


    # Motion model
    def _motion_model(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Motion model: predict next state from current state and control"""

        theta       = x[2, 0]
        delta_s     = u[0, 0]
        delta_theta = u[1, 0]

        new_theta = theta + delta_theta
        new_x     = x[0, 0] + delta_s * np.cos(new_theta)
        new_y     = x[1, 0] + delta_s * np.sin(new_theta)

        return np.array([[new_x], [new_y], [new_theta]])

    def _jacob_motion(self, x: np.ndarray, u: np.ndarray) -> tuple:
        """Compute Jacobian of motion model """

        theta       = x[2, 0]
        delta_s     = u[0, 0]
        delta_theta = u[1, 0]
        new_theta   = theta + delta_theta

        n_lm = self._calc_n_lm()

        # State selection matrix
        Fx = np.hstack((
            np.eye(EKF_STATE_SIZE),
            np.zeros((EKF_STATE_SIZE, EKF_LM_SIZE * n_lm))
        ))

        # Jacobian of motion w.r.t. state
        jF = np.array([
            [0.0, 0.0, -delta_s * np.sin(new_theta)],
            [0.0, 0.0,  delta_s * np.cos(new_theta)],
            [0.0, 0.0,  0.0]
        ])

        # Full state Jacobian
        G = np.eye(len(x)) + Fx.T @ jF @ Fx

        return G, Fx

    # Observation model
    def _calc_landmark_position(self, z: np.ndarray) -> np.ndarray:
        """Calculate landmark position from observation"""

        x     = self.xEst[0, 0]
        y     = self.xEst[1, 0]
        theta = self.xEst[2, 0]

        distance = z[0, 0]
        bearing  = z[1, 0]

        lm_x = x + distance * np.cos(theta + bearing)
        lm_y = y + distance * np.sin(theta + bearing)

        return np.array([[lm_x], [lm_y]])

    def _observation_model(self, lm: np.ndarray) -> np.ndarray:
        """Compute expected observation from landmark"""

        dx = lm[0, 0] - self.xEst[0, 0]
        dy = lm[1, 0] - self.xEst[1, 0]

        distance = np.sqrt(dx**2 + dy**2)
        bearing  = np.arctan2(dy, dx) - self.xEst[2, 0]
        bearing  = self._pi_2_pi(bearing)

        return np.array([[distance], [bearing]])

    def _calc_innovation(self, lm: np.ndarray, z: np.ndarray, lm_id: int) -> tuple:
        """Calculate innovation (measurement residual)"""

        delta = lm - self.xEst[0:2]
        q     = float(delta.T @ delta)

        z_pred = self._observation_model(lm)

        # Innovation
        y       = z - z_pred
        y[1, 0] = self._pi_2_pi(y[1, 0])  # Normalize angle

        # Observation Jacobian
        H = self._jacob_observation(q, delta, lm_id + 1)

        # Innovation covariance
        S = H @ self.PEst @ H.T + self.R

        return y, S, H

    def _jacob_observation(self, q: float, delta: np.ndarray, lm_idx: int) -> np.ndarray:
        """Compute Jacobian of observation model"""

        sq = np.sqrt(q)

        # Jacobian w.r.t. [robot_x, robot_y, theta, lm_x, lm_y]
        G = np.array([
            [-sq * delta[0, 0], -sq * delta[1, 0], 0, sq * delta[0, 0], sq * delta[1, 0]],
            [delta[1, 0], -delta[0, 0], -q, -delta[1, 0], delta[0, 0]]
        ]) / q

        n_lm = self._calc_n_lm()

        # Selection matrices
        F1 = np.hstack((np.eye(3), np.zeros((3, 2 * n_lm))))
        F2 = np.hstack((
            np.zeros((2, 3)),
            np.zeros((2, 2 * (lm_idx - 1))),
            np.eye(2),
            np.zeros((2, 2 * n_lm - 2 * lm_idx))
        ))
        F = np.vstack((F1, F2))

        H = G @ F
        return H

    # Data association
    def _search_correspond_landmark_id(self, z: np.ndarray) -> int:
        """Find corresponding landmark using Mahalanobis distance"""

        n_lm      = self._calc_n_lm()
        distances = []

        for i in range(n_lm):
            lm = self._get_landmark_position(i)
            y, S, H = self._calc_innovation(lm, z, i)

            # Mahalanobis distance
            d = float(y.T @ np.linalg.inv(S) @ y)
            distances.append(d)

        # Add threshold for new landmark
        distances.append(self.mahal_threshold)

        return int(np.argmin(distances))

    # Utilities
    def _calc_n_lm(self) -> int:
        """Calculate number of landmarks in state"""
        return (len(self.xEst) - EKF_STATE_SIZE) // EKF_LM_SIZE

    def _get_landmark_position(self, idx: int) -> np.ndarray:
        """Get landmark position from state vector"""
        start = EKF_STATE_SIZE + EKF_LM_SIZE * idx
        end = start + EKF_LM_SIZE
        return self.xEst[start:end, :]

    @staticmethod
    def _pi_2_pi(angle: float) -> float:
        """Normalize angle to [-pi, pi]"""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    # Visualization
    def plot_trajectory(self, save_path: str = None):
        """Plot estimated robot trajectory"""
        fig, ax = plt.subplots(figsize=(10, 10))

        ax.plot(self.history_x, self.history_y, 'b-', linewidth=1.5, label='EKF Trajectory')
        ax.plot(self.history_x[0], self.history_y[0], 'go', markersize=10, label='Start')
        ax.plot(self.history_x[-1], self.history_y[-1], 'ro', markersize=10, label='End')

        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_title('EKF SLAM - Robot Trajectory')
        ax.legend()
        ax.grid(True)
        ax.axis('equal')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.savefig(f"{PATH_SLAM}/ekf_trajectory.png", dpi=150, bbox_inches='tight')
        plt.close()

    def plot_landmarks(self, save_path: str = None):
        """Plot landmarks with uncertainty ellipses"""
        fig, ax = plt.subplots(figsize=(12, 12))

        # Plot trajectory
        ax.plot(self.history_x, self.history_y, 'b-', linewidth=1, alpha=0.7, label='Trajectory')

        # Plot landmarks with covariance ellipses
        landmarks = self.get_landmarks()
        for lm in landmarks:
            ax.plot(lm['x'], lm['y'], 'r^', markersize=10)

            # Get covariance and plot ellipse
            cov = self.get_landmark_covariance(lm['id'])
            self._plot_covariance_ellipse(ax, lm['x'], lm['y'], cov, color='r', alpha=0.3)

        # Plot robot pose with covariance
        x, y, theta = self.get_robot_pose()
        ax.plot(x, y, 'go', markersize=12, label='Robot')
        robot_cov = self.get_robot_covariance()[0:2, 0:2]
        self._plot_covariance_ellipse(ax, x, y, robot_cov, color='g', alpha=0.3)

        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_title(f'EKF SLAM - Map ({len(landmarks)} landmarks)')
        ax.legend()
        ax.grid(True)
        ax.axis('equal')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.savefig(f"{PATH_SLAM}/ekf_landmarks.png", dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_covariance_ellipse(self, ax, x, y, cov, color='b', alpha=0.5, n_std=2.0):
        """Plot covariance as an ellipse"""
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Ensure non-negative eigenvalues
        eigenvalues = np.maximum(eigenvalues, 1e-6)

        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        width, height = 2 * n_std * np.sqrt(eigenvalues)

        ellipse = Ellipse(
            xy=(x, y),
            width=width,
            height=height,
            angle=angle,
            facecolor=color,
            alpha=alpha,
            edgecolor=color,
            linewidth=2
        )
        ax.add_patch(ellipse)

    def store_scan_points(self, scan_points: list):
        """Store scan points for later visualization"""

        points = [(p.x, p.y) for p in scan_points]
        self.scan_points_history.append(points)

    def plot_map(self, save_path: str = None):
        """Plot complete map with trajectory, wall points, and landmarks"""
        
        fig, ax = plt.subplots(figsize=(14, 12))

        # Plot all scan points (walls)
        all_x = []
        all_y = []
        for scan in self.scan_points_history:
            for px, py in scan:
                all_x.append(px)
                all_y.append(py)

        if all_x:
            ax.scatter(all_x, all_y, c='blue', s=1, alpha=0.3, label='Wall Points')

        # Plot trajectory
        ax.plot(self.history_x, self.history_y, 'g-', linewidth=2, alpha=0.8, label='Trajectory')
        ax.plot(self.history_x[0], self.history_y[0], 'go', markersize=12, label='Start')
        ax.plot(self.history_x[-1], self.history_y[-1], 'g^', markersize=12, label='End')

        # Plot landmarks with covariance ellipses
        landmarks = self.get_landmarks()
        for i, lm in enumerate(landmarks):
            if i == 0:
                ax.plot(lm['x'], lm['y'], 'r^', markersize=10, label='Landmarks')
            else:
                ax.plot(lm['x'], lm['y'], 'r^', markersize=10)

            # Plot covariance ellipse
            cov = self.get_landmark_covariance(lm['id'])
            self._plot_covariance_ellipse(ax, lm['x'], lm['y'], cov, color='red', alpha=0.2)

        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_title(f'EKF SLAM - Complete Map ({len(landmarks)} landmarks, {len(self.scan_points_history)} scans)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.savefig(f"{PATH_SLAM}/ekf_map.png", dpi=150, bbox_inches='tight')
        plt.close()

    def generate_animation(self, output_path: str = None, step: int = 5, fps: int = 20):
        """
        Generate an animation showing the robot's journey through the map.

        Args:
            output_path: Path to save the animation (default: output/slam/ekf_animation.mp4)
            step: Only render every Nth frame to speed up generation
            fps: Frames per second in output video
        """
        from matplotlib.animation import FuncAnimation, FFMpegWriter
        import os

        if output_path is None:
            output_path = f"{PATH_SLAM}/ekf_animation.mp4"

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        n_frames = len(self.scan_points_history)
        frame_indices = list(range(0, n_frames, step))

        print(f"Generating animation with {len(frame_indices)} frames...")

        # Compute bounds from all data
        all_x = [x for scan in self.scan_points_history for x, y in scan]
        all_y = [y for scan in self.scan_points_history for x, y in scan]
        all_x.extend(self.history_x)
        all_y.extend(self.history_y)

        x_min, x_max = min(all_x) - 1, max(all_x) + 1
        y_min, y_max = min(all_y) - 1, max(all_y) + 1

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))

        # Initialize plot elements
        scan_scatter = ax.scatter([], [], c='blue', s=1, alpha=0.3, label='Wall Points')
        traj_line, = ax.plot([], [], 'g-', linewidth=2, alpha=0.8, label='Trajectory')
        robot_dot, = ax.plot([], [], 'go', markersize=15, label='Robot')
        landmark_scatter = ax.scatter([], [], c='red', s=100, marker='^', label='Landmarks')
        start_dot, = ax.plot([0], [0], 'ko', markersize=10, label='Start')

        # Info text
        info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                           fontsize=10, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        def init():
            scan_scatter.set_offsets(np.empty((0, 2)))
            traj_line.set_data([], [])
            robot_dot.set_data([], [])
            landmark_scatter.set_offsets(np.empty((0, 2)))
            info_text.set_text('')
            return scan_scatter, traj_line, robot_dot, landmark_scatter, info_text

        def update(frame_num):
            idx = frame_indices[frame_num]

            # Accumulated scan points up to this frame
            acc_x = []
            acc_y = []
            for i in range(idx + 1):
                for px, py in self.scan_points_history[i]:
                    acc_x.append(px)
                    acc_y.append(py)

            if acc_x:
                scan_scatter.set_offsets(np.column_stack([acc_x, acc_y]))

            # Trajectory up to this frame
            traj_line.set_data(self.history_x[:idx+2], self.history_y[:idx+2])

            # Current robot position
            robot_dot.set_data([self.history_x[idx+1]], [self.history_y[idx+1]])

            # Landmarks (we need to track when each was discovered)
            # For now, show all landmarks that exist by end
            landmarks = self.get_landmarks()
            if landmarks:
                lm_x = [lm['x'] for lm in landmarks]
                lm_y = [lm['y'] for lm in landmarks]
                landmark_scatter.set_offsets(np.column_stack([lm_x, lm_y]))

            # Count observations at this frame (approximate)
            n_obs = len([s for s in self.scan_points_history[:idx+1]])

            # Info text
            robot_x = self.history_x[idx+1]
            robot_y = self.history_y[idx+1]
            robot_theta = self.history_theta[idx+1]
            info_text.set_text(
                f'Scan: {idx+1}/{n_frames}\n'
                f'Robot: ({robot_x:.2f}, {robot_y:.2f})\n'
                f'Heading: {np.degrees(robot_theta):.1f}°\n'
                f'Landmarks: {len(landmarks)}'
            )

            ax.set_title(f'EKF SLAM Animation - Scan {idx+1}/{n_frames}')

            return scan_scatter, traj_line, robot_dot, landmark_scatter, info_text

        anim = FuncAnimation(fig, update, init_func=init,
                            frames=len(frame_indices), interval=50, blit=True)

        # Save animation
        try:
            writer = FFMpegWriter(fps=fps, bitrate=1800)
            anim.save(output_path, writer=writer)
            print(f"Animation saved to: {output_path}")
        except Exception as e:
            # Fallback to gif if ffmpeg not available
            gif_path = output_path.replace('.mp4', '.gif')
            print(f"FFmpeg not available, saving as GIF: {gif_path}")
            anim.save(gif_path, writer='pillow', fps=fps)
            print(f"Animation saved to: {gif_path}")

        plt.close()
