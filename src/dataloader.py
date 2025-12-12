import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import src.__config as GLOBALS


# Test
def load_data_using_pd(data_path, debug: bool = False):
    
    
    pass

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

def smoth_filter_data(lidar_data, window_size: int=7):

    pad_size = window_size // 2
    data_padded = np.pad(lidar_data, (pad_size, pad_size), mode='edge')
    
    return np.convolve(data_padded, np.ones(window_size)/window_size, mode='valid')

def calc_corner(x, y):

    angles=[]
    idx_corners=[]

    angles.append(0)

    for i in range(1, (len(x)-1)):

        prev_vetor_x=x[i]-x[i-1]
        prev_vetor_y=y[i]-y[i-1]
        next_vetor_x=x[i+1]-x[i]
        next_vetor_y=y[i+1]-y[i]

        prod_vet=prev_vetor_x*next_vetor_x + prev_vetor_y*next_vetor_y 
        
        mod_prev=np.sqrt(prev_vetor_x**2 + prev_vetor_y**2)
        mod_next=np.sqrt(next_vetor_x**2 + next_vetor_y**2)

        if (mod_prev==0 or mod_next==0):
            angles.append(np.nan)
            continue
        
        cos_ang=prod_vet/(mod_prev*mod_next)

        cos_ang=np.clip(cos_ang, -1.0, 1.0)

        angle=np.rad2deg(np.arccos(cos_ang))
        angles.append(angle)
        
        if (angle>30):
            idx_corners.append(i)


    
    angles.append(0)

    return angles, idx_corners

def plot_points(x: np.ndarray, y: np.ndarray, idx_corners, save_plot: bool = False):

    plt.figure(figsize=(8, 6))

    # Todos os pontos (azul)
    plt.scatter(x, y, color='blue', alpha=0.5, label='LiDAR Points')

    # Pontos dos corners (vermelho)
    if len(idx_corners) > 0:
        plt.scatter(x[idx_corners], y[idx_corners], 
                    color='red', s=50, label='Corners', zorder=5)

    plt.title('LiDAR Points with Corners Highlighted')
    plt.xlabel('X coordinates')
    plt.ylabel('Y coordinates')
    plt.grid(True)
    plt.legend()
    plt.show()

    if save_plot:
        plt.savefig('scatter_plot.png')

# Main
if __name__ == "__main__":
    travelled_distance, measured_variation, lidar_measurements=load_data(data_path=GLOBALS.PATH_DATASET, debug=GLOBALS.DEBUG)
    print(lidar_measurements.shape)
    for i in range(len(lidar_measurements)):
        lidar_filtered=smoth_filter_data(lidar_measurements[20])
        x, y=convert_data_to_points(lidar_filtered, debug=GLOBALS.DEBUG)
        angles, idx_corners=calc_corner(x,y)
        if False:
            for j in range(len(x)): 
                print( "x " + str(x[j]) + " y " + str(y[j]) + ":" + str(angles[j]))
        print(idx_corners)
        plot_points(x, y, idx_corners)
        break