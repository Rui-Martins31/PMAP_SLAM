# ASSIGNMENT 03

The goal of this subproject is the implementation and testing of a simultaneous localization and mapping (SLAM) system for a robot equipped with a LIDAR. The LIDAR provides range measurements along 121 angles from −60º to 60º with a 1º step and centered with the robot forward axis.

The data slam.txt contains data from the robot while moving in an area. The first two columns are, respectively, the mesasured travelled distance (∆s) and the measured variation of the robot orientation (∆θ). The last 121 columns are the LIDAR measurements (assumed instantaneous). All measurements are noisy. Assume a simple odometry model:

dx/ds = cos(θ)

dy/ds = sin(θ)

## Task1

Implement an algorithm that processes each LIDAR measurement (121 beams) and detects corners (convex or concave). Process all the measurements and locate the detected corners with respect to the robot position.

## Task2

Design and implement a system that processes the variations of the robot pose and also the LIDAR readings to estimate in real time the pose of the robot as well as the locations of the corners. Whenever a corner is detected, check whether it is a known one or not (define a condition for that and in this last case update the state of the filter.

## Task3

Based on all the measurments, estimate the evolution of the pose of the robot and of the map of the
environment.

**Suggestion:** Look for information about smoothing in the literature.

Produce a small report with the relevant data in each task. Also produce videos showing the evolution
of the real time robot pose estimate in Task 2