import numpy as np
import matplotlib.pyplot as plt

import src.__config as GLOBALS

class Robot:
    def __init__(self):
        # Position
        self.x: list[float] = [0.0]
        self.y: list[float] = [0.0]

        # Orientation
        self.theta: list[float] = [0.0]

    def step(self, delta_s: float, delta_theta: float):
        """Update robot position based on odometry measurements"""

        # Update orientation
        new_theta = self.theta[-1] + delta_theta

        # Update position
        new_x = self.x[-1] + delta_s * np.cos(new_theta)
        new_y = self.y[-1] + delta_s * np.sin(new_theta)

        # Store new state
        self.x.append(new_x)
        self.y.append(new_y)
        self.theta.append(new_theta)

    # Getters
    def get_current_pose(self) -> tuple[float, float, float]:
        """Get current robot pose"""

        return (self.x[-1], self.y[-1], self.theta[-1])

    def get_homogeneous_matrix(self) -> np.ndarray:
        """Get homogeneous transformation matrix from robot frame to world frame"""

        x, y, theta = self.get_current_pose()

        H = np.array([
            [np.cos(theta), -np.sin(theta), x],
            [np.sin(theta),  np.cos(theta), y],
            [0,              0,              1]
        ])

        return H
    
    # Helpers
    def plot_trajectory(self):

        plt.figure(figsize=(10, 8))
        plt.plot(self.x, self.y, 'b-', linewidth=1.5, label='Robot Trajectory')

        # Mark start and end positions
        plt.plot(self.x[0], self.y[0], 'go', markersize=10, label='Start')
        plt.plot(self.x[-1], self.y[-1], 'ro', markersize=10, label='End')

        plt.xlabel('X position (m)')
        plt.ylabel('Y position (m)')
        plt.title('Robot Trajectory (Odometry)')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.legend()
        plt.tight_layout()
        plt.savefig(GLOBALS.PATH_OUTPUT + "odometry_robot_trajectory")
        # plt.show()