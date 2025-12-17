from src.lidar import Lidar
from src.robot import Robot
from src.map import Map

def main():
    # Load Lidar
    lidar: Lidar = Lidar()
    lidar.viz_dataframe()

    # Robot and Map
    robot: Robot = Robot()
    map: Map     = Map()

    for idx in range(lidar.get_num_scans()):
        delta_s, delta_theta = lidar.get_odometry(idx)

        robot.step(delta_s, delta_theta)

        map.compute_points_position(
            robot_homo_matrix=robot.get_homogeneous_matrix(),
            lidar_ranges=lidar.get_ranges(idx),
            lidar_angles=lidar.get_angles(),
            scan_idx=idx
        )

    # Visualization
    robot.plot_trajectory()
    map.plot_map()
        



if __name__ == "__main__":
    main()