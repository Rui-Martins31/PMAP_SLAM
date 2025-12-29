import numpy as np
import matplotlib.pyplot as plt





#ADICIONADO

from scipy.optimize import least_squares

def update_pose_nodes_and_constrains(nodes_robot_pos, constrains_robot_mov, distance_variation, angle_variation):

    iteration=len(nodes_robot_pos)

    new_x=nodes_robot_pos[iteration-1][0]+distance_variation*np.cos(nodes_robot_pos[iteration-1][2])
    new_y=nodes_robot_pos[iteration-1][1]+distance_variation*np.sin(nodes_robot_pos[iteration-1][2])
    new_theta=nodes_robot_pos[iteration-1][2]+angle_variation

    nodes_robot_pos.append(np.asarray([new_x, new_y, new_theta]))

    constrains_robot_mov.append((iteration-1, iteration, distance_variation, angle_variation))

    return nodes_robot_pos, constrains_robot_mov


def verify_and_add_corner(nodes_corners, corner_to_verify,  threshold):

    dist_min=999999
    idx_min=-1

    for i in range(0, len(nodes_corners)):
        dist = np.sqrt((corner_to_verify[0]-nodes_corners[i][0])**2+(corner_to_verify[1]-nodes_corners[i][1])**2)
        if(dist<dist_min):
            dist_min=dist
            idx_min=i


    if dist_min>=threshold:
        idx_min=-1
    
    if idx_min==-1:
        nodes_corners.append(corner_to_verify)
        return nodes_corners, len(nodes_corners)-1
    else:
        return nodes_corners, idx_min
 
    
    

def add_corner_constrain(constrains_observation_corners, idx_pos, x_local, y_local, idx_corner):
    
    constrains_observation_corners.append((idx_pos, idx_corner, x_local, y_local))
    
    return constrains_observation_corners



def initialize_state(nodes_robot_pos, nodes_corners):
    
    state=[]

    for node_p in nodes_robot_pos:
        state.extend(node_p)

    for node_c in nodes_corners:
        state.extend(node_c)
    #TEM DE FICAR COM AS VARIAVEIS TODAS NUMA LINHA SÓ PORQUE PARA OTIMIZAR SO FUNCIONA ASSIM
    #state=[x1_robot, y1_robot, theta1_robot, x2_robot, y2_robot, theta2_robot, ...,x1_corn, y1_corn,x2_corn,y2_corn,...]
    return np.array(state)

def function_error_slam(state, constrains_robot_mov, constrains_observation_corners, num_pos, sigma_xy_odom:float=0.05, sigma_corner:float=0.10, sigma_theta_odom:float=(np.deg2rad(2.0))):

    error_state=[]

    for constrain in constrains_robot_mov:

        idx_pos_orig, idx_pos_dest, distance, theta_variation=constrain
        idx_pos_orig=3*idx_pos_orig
        idx_pos_dest=3*idx_pos_dest

        x_real, y_real, theta_real=state[idx_pos_dest:idx_pos_dest+3]

        x_pred=state[idx_pos_orig]+distance*np.cos(state[idx_pos_orig+2])
        y_pred=state[idx_pos_orig+1]+distance*np.sin(state[idx_pos_orig+2])
        theta_pred=normalize_angle(state[idx_pos_orig+2]+theta_variation)

        e_x=x_real-x_pred
        e_y=y_real-y_pred
        e_theta=normalize_angle(theta_real-theta_pred)

        error_state.extend([e_x/sigma_xy_odom, e_y/sigma_xy_odom, e_theta/sigma_theta_odom])

    for constrain in constrains_observation_corners:

        idx_pos, idx_corner, x_corner_real, y_corner_real=constrain
        idx_pos=3*idx_pos
        idx_corner=idx_corner*2+num_pos*3

        x_robot, y_robot, theta_robot=state[idx_pos:idx_pos+3]

        x_corner, y_corner=state[idx_corner:idx_corner+2]

        var_x=x_corner-x_robot
        var_y=y_corner-y_robot

        #Referencial do robo
        x_corner_pred =np.cos(theta_robot)*var_x+np.sin(theta_robot)*var_y
        y_corner_pred =-np.sin(theta_robot)*var_x+np.cos(theta_robot)*var_y        


        e_x=x_corner_real-x_corner_pred
        e_y=y_corner_real-y_corner_pred
        error_state.extend([e_x/sigma_corner, e_y/sigma_corner])

    #FIXA A POSICAO DA PRIMEIRA POSIÇÃO NA ORIGEM
    error_state.extend([state[0]/sigma_xy_odom, state[1]/sigma_xy_odom, normalize_angle(state[2])/sigma_theta_odom])

    return np.array(error_state)


def normalize_angle(theta):
    return np.arctan2(np.sin(theta), np.cos(theta))


def optimize_graph_least_squares(nodes_robot_pos, nodes_corners, constrains_robot_mov, constrains_observation_corners):

    print("STARTING THE OPTIMIZER")
    
    state_initial=initialize_state(nodes_robot_pos, nodes_corners)

    state_optimal=least_squares(function_error_slam, state_initial, args=(constrains_robot_mov,constrains_observation_corners, len(nodes_robot_pos)), verbose=2, xtol=1e-8, ftol=1e-8, method="lm")

    return state_optimal.x

def reconstruct_nodes_from_optimal_vector_state(state_optimal, num_pos):

    robot_pos=[]
    corners_pos=[]

    for i in range(num_pos):
        idx_pos=3*i
        robot_pos.append(np.asarray(state_optimal[idx_pos:idx_pos+3]))

    for i in range((len(state_optimal)-num_pos*3)//2):
        idx_pos=2*i+num_pos*3
        corners_pos.append(np.asarray(state_optimal[idx_pos: idx_pos+2]))

    return robot_pos, corners_pos




if __name__ == "__main__":
    ## DEBUG
    print(f"Starting Task 3...")
    
    #nodes_robot_pos=[x, y, theta]
    nodes_robot_pos=[]
    #nodes_corners=[x, y]
    nodes_corners=[]
    
    #constrains_robot_mov=[node_pos_origin, node_pos_dest, distance_variation, angle_variation]
    constrains_robot_mov=[]
    #constrains_observation_corners=[pos_node_id, corner_node_id, x_local, y_local]
    constrains_observation_corners=[]

    nodes_robot_pos.append([0, 0, 0])

    # Trajetória apenas com odometria
    odom_path = []
    odom_pose = np.array([0.0, 0.0, 0.0])
    odom_path.append(odom_pose[:2].copy())


    for i in range(len(lidar_measurements)):

        # Atualização apenas com odometria (dead-reckoning)
        odom_pose[2] += angle_variation[i]
        odom_pose[0] += travelled_distance[i] * np.cos(odom_pose[2])
        odom_pose[1] += travelled_distance[i] * np.sin(odom_pose[2])
        odom_path.append(odom_pose[:2].copy())

        
        update_pose_nodes_and_constrains(nodes_robot_pos, constrains_robot_mov, travelled_distance[i], angle_variation[i])

        data_no_outliers=remove_outliers(lidar_measurements[i], threshold=0.07)
        lidar_filtered=smoth_filter_data(data_no_outliers)
        x_local, y_local=convert_data_to_points(lidar_filtered, debug=GLOBALS.DEBUG)   

        idx_corners=calc_corners(x_local, y_local, 0, len(x_local)-1, threshold=0.075)
        idx_corners.sort()

        for idx in idx_corners:

            x_global_corner = nodes_robot_pos[-1][0] + x_local[idx] * np.cos(nodes_robot_pos[-1][2]) - y_local[idx] * np.sin(nodes_robot_pos[-1][2])
            y_global_corner = nodes_robot_pos[-1][1] + x_local[idx] * np.sin(nodes_robot_pos[-1][2]) + y_local[idx] * np.cos(nodes_robot_pos[-1][2])

            global_corner=[x_global_corner, y_global_corner]

            nodes_corners, idx_verified_corner=verify_and_add_corner(nodes_corners, global_corner, threshold=0.2)
            constrains_observation_corners=add_corner_constrain(constrains_observation_corners, len(nodes_robot_pos)-1, x_local[idx], y_local[idx], idx_verified_corner)

    
    state_optimal=optimize_graph_least_squares(nodes_robot_pos, nodes_corners, constrains_robot_mov, constrains_observation_corners)
    robot_pos_optimal, corners_optimal=reconstruct_nodes_from_optimal_vector_state(state_optimal, len(nodes_robot_pos))

    robot_pos_optimal = np.array(robot_pos_optimal)
    corners_optimal = np.array(corners_optimal)

    odom_path = np.array(odom_path)
    
    plt.figure(figsize=(10,8))

    # Trajetória apenas com odometria
    plt.plot(
        odom_path[:,0],
        odom_path[:,1],
        'r--',
        linewidth=2,
        label="Odometry only"
    )

    # Trajetória otimizada pelo Graph SLAM
    plt.plot(
        robot_pos_optimal[:,0],
        robot_pos_optimal[:,1],
        'b-',
        linewidth=2,
        label="Graph SLAM"
    )

    # Corners otimizados
    plt.scatter(
        corners_optimal[:,0],
        corners_optimal[:,1],
        c='k',
        marker='x',
        label="Landmarks"
    )

    plt.axis("equal")
    plt.grid()
    plt.legend()
    plt.title("Task 3 – Trajectory comparison: Odometry vs Graph SLAM")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.show()