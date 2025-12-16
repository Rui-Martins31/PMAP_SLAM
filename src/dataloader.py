import numpy as np
import matplotlib.pyplot as plt


import src.__config as GLOBALS



def initialize_matrixes():

    state_initial=np.array([0.0, 0.0, 0.0])

    P_initial=np.diag([0.01, 0.01, 0.01])

    Q_initial=np.diag([0.01, 0.01, 0.005])

    R_initial=np.diag([0.05, 0.05])
    
    return state_initial, P_initial, Q_initial, R_initial


def calc_point_dist_to_line(first_point_line, last_point_line, point_to_calc):
    
    if ((first_point_line[0]==last_point_line[0]) and (first_point_line[1]==last_point_line[1])):
        return np.sqrt((last_point_line[0]-point_to_calc[0])**2 + (last_point_line[1]-point_to_calc[1])**2)

    #Coeficients of general equation of a line between to points
    A=last_point_line[1]-first_point_line[1]
    B=first_point_line[0]-last_point_line[0]
    C=first_point_line[1]*(last_point_line[0]-first_point_line[0])-(last_point_line[1]-first_point_line[1])*first_point_line[0]

    #Distance froma a point to the line
    return (abs(A*point_to_calc[0]+B*point_to_calc[1]+C) / np.sqrt(A**2 + B**2))

def calc_corners(x, y, idx_start, idx_end, threshold):

    if (idx_start==idx_end):
        return [] 
    
    point_start=[x[idx_start], y[idx_start]]
    point_end=[x[idx_end], y[idx_end]]

    max_dist=0
    idx_max_dist=-1

    for i in range(idx_start+1, idx_end):

        point_to_calc=[x[i], y[i]]
        
        dist=calc_point_dist_to_line(point_start, point_end, point_to_calc)

        if (dist>max_dist):
            max_dist=dist
            idx_max_dist=i
    
    if max_dist>threshold:
        first_half_corners=calc_corners(x, y, idx_start, idx_max_dist, threshold)
        second_half_corners=calc_corners(x, y, idx_max_dist, idx_end, threshold)

        return first_half_corners + [idx_max_dist] + second_half_corners
    
    else:
        return[]


def update_pos_and_orientation_and_path(pos_robot, theta_robot, path_robot, delta_dist, delta_theta):
        
    theta_robot=theta_robot+delta_theta

    pos_robot[0]=pos_robot[0] + delta_dist * np.cos(theta_robot)
    pos_robot[1]=pos_robot[1] + delta_dist * np.sin(theta_robot)

    path_robot.append([pos_robot[0], pos_robot[1]])
    
    return pos_robot, theta_robot, path_robot

def verify_corner_known(new_corner, corner_map, threshold: float=0.1):

    

    if(len(corner_map)==0):
        corner_map.append(new_corner)
        return corner_map, False

    is_known = False

    for known_corner in corner_map:
        dist=np.sqrt((new_corner[0] - known_corner[0])**2 + (new_corner[1]-known_corner[1])**2)
        if (dist < threshold):
            #print("Corner Removed")
            is_known = True
            break
            
    
    if not is_known:
        corner_map.append(new_corner)
            
    return corner_map, is_known

def load_data(data_path, debug:bool=False):
    data = np.loadtxt(data_path)

    travelled_distance=data[:,0]
    angle_variation=data[:,1]
    lidar_measurements=data[:,2:]

    #DEBUG
    if (debug):
        print("Data shape:"+str(data.shape))
        print("Travelled distance example values:"+str(travelled_distance[:20]))
        print("Measured variation example values:"+str(angle_variation[:20]))
        print("Lidar measurements shape:"+str(lidar_measurements.shape))


    return travelled_distance, angle_variation, lidar_measurements

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

def remove_outliers(lidar_data, threshold=0.2):
    
    data_copy=lidar_data.copy()

    for i in range(1, len(lidar_data)-1):

        prev_dist=abs(lidar_data[i]-lidar_data[i-1])
        next_dist=abs(lidar_data[i+1]-lidar_data[i])

        if (prev_dist>threshold and next_dist>threshold):
            data_copy[i]=(lidar_data[i-1]+lidar_data[i+1])/2
            #print("REMOVED")
        
    
    return data_copy

def EKF_predict_phase(X_state, P, Q, distance_variation, angle_variation):

    theta=X_state[2]
    
    X_state[0]=X_state[0] + distance_variation * np.cos(theta)
    X_state[1]=X_state[1] + distance_variation * np.sin(theta)
    X_state[2]=X_state[2]+angle_variation

    n=len(X_state)
    F=np.eye(n)

    #Jacobiano
    F[0][2]=-distance_variation*np.sin(theta)
    F[1][2]=distance_variation*np.cos(theta)

    Q_new=np.zeros((n,n))
    Q_new[0:3][0:3]=Q

    P = F @ P @ F.T + Q_new

    return X, P

def add_corner_to_model(X, P, global_corner, R):

    n=len(X)
    
    X=np.append(X, global_corner[0])
    X=np.append(X, global_corner[1])

    #VERIFICAR SE O P É ATUALIZADO DESTA MANEIRA

    P_new=np.zeros((n+2, n+2))
    P_new[:n,:n]=P
    P_new[n:,n:]=R

    return X, P_new

def verify_register_corner(X, corner_to_compare, threshold=0.1):

    #PODERÁ SE SUBSTITUIR PELA Mahalanobis data association

    dist_min=999999
    idx_min=-1

    for i in range(3, len(X), 2):
        dist = np.sqrt((corner_to_compare[0]-X[i])**2+(corner_to_compare[1]-X[i+1])**2)
        if(dist<dist_min):
            dist_min=dist
            idx_min=i


    if dist_min<threshold:
        return idx_min
    
    else:
        return None

def EKF_update_phase(X, P, global_corner, corner_idx, R):

    #CONTINUAR

    pass

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
# Main
if __name__ == "__main__":


    travelled_distance, angle_variation, lidar_measurements=load_data(data_path=GLOBALS.PATH_DATASET, debug=GLOBALS.DEBUG)
    
    #print(lidar_measurements.shape)

    #TASK 1-----------------------------------------
    if 1 in GLOBALS.TASK:
        ## DEBUG
        print(f"Starting Task 1...")

        x_vetor=[]
        y_vetor=[]
        idx_corners_vetor=[]

        for i in range (len(lidar_measurements)):

            data_no_outliers=remove_outliers(lidar_measurements[i], threshold=0.07)
            lidar_filtered=smoth_filter_data(data_no_outliers)

            x, y=convert_data_to_points(lidar_filtered, debug=GLOBALS.DEBUG)
            x_vetor.append(x)
            y_vetor.append(y)

            idx_corners=calc_corners(x, y, 0, len(x)-1, threshold=0.02)
            idx_corners.sort()

            idx_corners_vetor.append(idx_corners)

            #plot_points(x_vetor[i], y_vetor[i], idx_corners_vetor[i])


            
        plot_points(x_vetor[GLOBALS.NUM_EXAMPLE], y_vetor[GLOBALS.NUM_EXAMPLE], idx_corners_vetor[GLOBALS.NUM_EXAMPLE])
    


    #TASK 2-----------------------------------------
    if 2 in GLOBALS.TASK:
        ## DEBUG
        print(f"Starting Task 2...")

        path_robot=[]
        corner_map=[]

        X,P,Q,R=initialize_matrixes()
        
        for i in range(len(lidar_measurements)): 


            X, P = EKF_predict_phase(X, P, Q, travelled_distance[i], angle_variation[i])

            path_robot.append(X[0:2])
                    
            data_no_outliers=remove_outliers(lidar_measurements[i], threshold=0.07)
            lidar_filtered=smoth_filter_data(data_no_outliers)
            x_local, y_local=convert_data_to_points(lidar_filtered, debug=GLOBALS.DEBUG)   
            
            idx_corners=calc_corners(x_local, y_local, 0, len(x_local)-1, threshold=0.02)
            idx_corners.sort()
            

            for idx in idx_corners:

                x_global_corner = X[0] + x_local[idx] * np.cos(X[2]) - y_local[idx] * np.sin(X[2])
                y_global_corner = X[1] + x_local[idx] * np.sin(X[2]) + y_local[idx] * np.cos(X[2])

                global_corner=[x_global_corner, y_global_corner]

                registered_corner_idx=verify_register_corner(X, global_corner, threshold=0.1)

                if registered_corner_idx is None:
                    X, P=add_corner_to_model(X, P, global_corner, R)
                
                else:
                    X, P=EKF_update_phase(X, P, global_corner, registered_corner_idx, R)

                    


        path_array = np.array(path_robot)
        
        # Extrair cantos do mapa
        map_x = [c[0] for c in corner_map]
        map_y = [c[1] for c in corner_map]

        plt.figure(figsize=(10, 8))
        
        #Desenha a linha do caminho do robô (Preto)
        if len(path_array) > 0:
            plt.plot(path_array[:, 0], path_array[:, 1], color='black', label='Robot Path', linewidth=1)
        
        #Desenha os cantos globais encontrados (Vermelho)
        plt.scatter(map_x, map_y, color='red', marker='x', label='Global Corners')

        plt.title("SLAM Task 2: Robot Path & Global Map")
        plt.xlabel("Global X")
        plt.ylabel("Global Y")
        plt.axis('equal') # Importante para não distorcer o mapa
        plt.grid(True)
        plt.legend()
        plt.show()
    


