import numpy as np
import matplotlib.pyplot as plt

import src.__config as GLOBALS


# Load data function
def load_data(data_path, debug:bool=False):
    data = np.loadtxt(data_path)

    travelled_distance=data[:,0]
    measured_variation=data[:,1]
    lidar_measurements=data[:,2:]

    #DEBUG
    if (debug):
        print("Data shape:"+str(data.shape))
        print("Travelled distance example values:"+str(travelled_distance[:20]))
        print("Measured variation example values:"+str(measured_variation[:20]))
        print("Lidar measurements shape:"+str(lidar_measurements.shape))


    return travelled_distance, measured_variation, lidar_measurements

#Convert data to points
def convert_data_to_points(distance, debug:bool=False):
    angles = np.deg2rad(np.linspace(-60, 60, 121))
    x = distance * np.cos(angles)
    y = distance * np.sin(angles)

    if debug:
        print("x shape:"+str(x.shape))
        print("y shape:"+str(y.shape))

    return x, y

# Main
if __name__ == "__main__":
    travelled_distance, measured_variation, lidar_measurements=load_data(data_path=GLOBALS.PATH_DATASET, debug=GLOBALS.DEBUG)
    x, y=convert_data_to_points(lidar_measurements, debug=GLOBALS.DEBUG)