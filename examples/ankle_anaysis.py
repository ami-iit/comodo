from comodo.mujocoSimulator.mujocoSimulator import MujocoSimulator
from comodo.robotModel.robotModel import RobotModel
from comodo.centroidalMPC.centroidalMPC import CentroidalMPC
from comodo.centroidalMPC.mpcParameterTuning import MPCParameterTuning
from comodo.TSIDController.TSIDParameterTuning import TSIDParameterTuning
import os
import xml.etree.ElementTree as ET
from comodo.TSIDController.TSIDController import TSIDController
import numpy as np 
from datetime import timedelta
from scipy.io import savemat
import time 
import pygad 
from dataBaseFitnessFunction import DatabaseFitnessFunction
import pickle
def matrix_to_rpy(matrix):
    """Converts a rotation matrix to roll, pitch, and yaw angles in radians.

    Args:
        matrix (numpy.ndarray): 3x3 rotation matrix.

    Returns:
        tuple: Tuple containing the roll, pitch, and yaw angles in radians.
    """
    assert matrix.shape == (3, 3), "Input matrix must be a 3x3 rotation matrix."

    # Extract rotation angles from the rotation matrix
    yaw = np.arctan2(matrix[1, 0], matrix[0, 0])
    pitch = np.arctan2(-matrix[2, 0], np.sqrt(matrix[2, 1]**2 + matrix[2, 2]**2))
    roll = np.arctan2(matrix[2, 1], matrix[2, 2])
    rpy = np.zeros(3)
    rpy[0] = roll
    rpy[1] = pitch
    rpy[2] = yaw
    return rpy



## Defining the urdf path of both the startin, del and the modified one
common_path = os.path.dirname(os.path.abspath(__file__))
# urdf_path_original = common_path + "/models/urdf/ergoCub/robots/ergoCubSN000/model.urdf"
# mujoco_path = common_path+ "/models/urdf/ergoCub/robots/ergoCubSN000/muj_model.xml"
urdf_path_original = "./ergoCubSN002/model.urdf"
mujoco_path = "muj_model.xml"

joint_name_list = [
       "r_hip_pitch",#0
        "r_hip_roll",#1
        "r_hip_yaw",#2
        "r_knee",#3
        "r_ankle_pitch",#4
        "r_ankle_roll",#5
        "l_hip_pitch",#6
        "l_hip_roll",#7
        "l_hip_yaw",#8
        "l_knee",#9
        "l_ankle_pitch", #10
        "l_ankle_roll",#11
        "r_shoulder_pitch", #12
        "r_shoulder_roll",#13
        "r_shoulder_yaw",#14
        "r_elbow",#15
        "l_shoulder_pitch",#16
        "l_shoulder_roll",#17
        "l_shoulder_yaw",#18
        "l_elbow"#19
]
# Load the URDF file
tree = ET.parse(urdf_path_original)
root = tree.getroot()

# Convert the XML tree to a string
robot_urdf_string_original = ET.tostring(root)


robot_model_init = RobotModel(robot_urdf_string_original.decode('utf8'), "stickBot", joint_name_list)
# mujoco_string = robot_model_init.get_mujoco_model()
robot_model_init.set_foot_corner(np.asarray([0.1, 0.05, 0.0]),np.asarray([0.1, -0.05, 0.0]),np.asarray([-0.1, -0.05, 0.0]),np.asarray([-0.1, 0.05, 0.0]))
# Defining initial robot configuration 
s_des = np.array( [ 0.56056952, 0.01903913, -0.0172335, -1.2220763, -0.52832664, -0.02720832, 0.56097981, 0.0327311 ,-0.02791293,-1.22200495,  -0.52812215, -0.04145696,0.02749586, 0.25187149, -0.14300417, 0.6168618, 0.03145343, 0.25644825, -0.14427671, 0.61634549,])
contact_frames_pose = {robot_model_init.left_foot_frame: np.eye(4),robot_model_init.right_foot_frame: np.eye(4)}
H_b = robot_model_init.get_base_pose_from_contacts(s_des, contact_frames_pose)
xyz_rpy = np.zeros(6)
xyz_rpy[:3] = H_b[:3,3]
rpy = matrix_to_rpy(H_b[:3,:3])
xyz_rpy[3:] = rpy
energy_tot = 0 

def on_generation(ga_instance):

    common_path = os.path.dirname(os.path.abspath(__file__))
    dataBase_instance = DatabaseFitnessFunction(
    common_path + "/results/fitnessDatabase"
    )
    generation_completed = ga_instance.generations_completed
    population = ga_instance.population
    fitness = ga_instance.last_generation_fitness
    for indx in range(len(fitness)): 
        fitness_value = fitness[indx]
        x_k = population[indx]
        dataBase_instance.update(x_k, fitness_value,generation_completed)
    filename_i = common_path + "/results/genetic" + str(generation_completed)+ ".p"
    pickle.dump(ga_instance.population, open(filename_i, "wb"))



def compute_fitness(pygadClass, x_k, idx):
    tsid_gain = x_k[:28]
    mpc_gains = x_k[28:]
    
    # Defining the TSID Controller 
    tsid_parameter = TSIDParameterTuning()
    tsid_parameter.set_from_x_k(tsid_gain)
    mpc_parameters = MPCParameterTuning()
    mpc_parameters.set_from_xk(mpc_gains)
    TSID_controller_instance = TSIDController(frequency=0.01,robot_model=robot_model_init)
    
    TSID_controller_instance.define_tasks(tsid_parameter)

    # Defining the simulator 
    mujoco_instance = MujocoSimulator()
    # mujoco_instance.load_model(robot_model_init, s= s_des, xyz_rpy=xyz_rpy,kv_motors=None, Im=None,  mujoco_path="muj_test.xml")
    mujoco_instance.load_model(robot_model_init, s= s_des, xyz_rpy=xyz_rpy,kv_motors=None, Im=None,  mujoco_path=mujoco_path)
    
    sim_frequency = 0.0001
    mujoco_instance.set_simulation_frequency(sim_frequency)
    s,ds,tau = mujoco_instance.get_state()
    t = mujoco_instance.get_simulation_time()
    H_b = mujoco_instance.get_base()
    w_b = mujoco_instance.get_base_velocity()

    TSID_controller_instance.set_state_with_base(s,ds,H_b,w_b,t)
    mpc = CentroidalMPC(robot_model=robot_model_init, step_length=0.1)

    n_step = int(TSID_controller_instance.frequency/mujoco_instance.get_simulation_frequency())
    n_step_mpc_tsid = int(mpc.get_frequency_seconds()/TSID_controller_instance.frequency)

    
    mpc.intialize_mpc(mpc_parameters=mpc_parameters)
    mpc.configure(s_init=s_des, H_b_init=H_b)
    TSID_controller_instance.compute_com_position()
    mpc.define_test_com_traj(TSID_controller_instance.COM.toNumPy())
    print(TSID_controller_instance.COM.toNumPy())
    TIME_TH = 20
    mujoco_instance.set_visualize_robot_flag(False)
    mujoco_instance.step(1)
    s,ds,tau = mujoco_instance.get_state()
    
    H_b = mujoco_instance.get_base()
    w_b = mujoco_instance.get_base_velocity()
    t = mujoco_instance.get_simulation_time()
    mpc.set_state_with_base(s=s, s_dot=ds, H_b=H_b, w_b=w_b,t=t)
    mpc.initialize_centroidal_integrator(s=s, s_dot=ds,H_b=H_b, w_b=w_b,t=t)
    mpc_output = mpc.plan_trajectory()  

    # Update MPC and getting the state
    mpc.set_state_with_base(s=s, s_dot=ds, H_b=H_b, w_b=w_b,t=t)
    TSID_controller_instance.set_state_with_base(s,ds,H_b,w_b,t)
    counter = 0 
    mpc_success = True
    succeded_controller = True
    energy_tot = 0.0
    vector_length = int(TIME_TH/TSID_controller_instance.frequency) + 4
    joint_positions_list = np.zeros([20,vector_length])
    joint_torques_list = np.zeros([20,vector_length])
    joint_velocities_list = np.zeros([20,vector_length])
    t_list = [] 
    i = 0 
    total_misalignment=0
    while(t<TIME_TH):
        print(t)
        # Reading robot state from simulator
        s,ds,tau = mujoco_instance.get_state()
        H_b = mujoco_instance.get_base()
        w_b = mujoco_instance.get_base_velocity()
        t = mujoco_instance.get_simulation_time()
        TSID_controller_instance.set_state_with_base(s=s,s_dot=ds,H_b=H_b, w_b=w_b,t=t)
        ## Updating for ankle analysis 
        
        joint_torques_list[:,i] = tau
        joint_positions_list[:,i] = s
        joint_velocities_list[:,i]=ds
        t_list.append(t)    
        i = i+1 
        if(counter == 0):
            mpc.set_state_with_base(s=s, s_dot=ds, H_b=H_b, w_b=w_b,t=t)
            mpc.update_references()
            mpc_success = mpc.plan_trajectory()
            mpc.contact_planner.advance_swing_foot_planner()
            if(not(mpc_success)): 
                print("MPC failed")
                break

        com, dcom,forces_left, forces_right, ang_mom = mpc.get_references()
        left_foot, right_foot = mpc.contact_planner.get_references_swing_foot_planner()
        TSID_controller_instance.update_task_references_mpc(com=com, dcom=dcom,ddcom=np.zeros(3),left_foot_desired=left_foot, right_foot_desired=right_foot,s_desired=np.array(s_des), wrenches_left = forces_left, wrenches_right = forces_right)

        succeded_controller = TSID_controller_instance.run()
        
        feet_normal, misalignment_feet = mujoco_instance.check_feet_status(s, H_b)
        total_misalignment += misalignment_feet
        # print(misalignment_feet)
        if(not(succeded_controller)): 
            print("Controller failed")
            break
        
        tau = TSID_controller_instance.get_torque()
        mujoco_instance.set_input(tau)
        mujoco_instance.step(n_step=n_step)
        counter = counter+ 1 

        if(counter == n_step_mpc_tsid): 
            counter = 0 
      
    succed_overall = succeded_controller and mpc_success
    joint_positions_array = np.array(joint_positions_list)
    joint_torques_array = np.array(joint_torques_list)
    joint_velocities_array = np.array(joint_velocities_list)
    t_array = np.array(t_list)

    # SAVING JOINT QUANTITIES INFO--------------------------------
    path = 'numeric_data/joint_quantities/'
    name = 'joint_torques.mat'
    dict = {label: value for label, value in zip(robot_model_init.joint_name_list, joint_torques_array)}
    savemat(path + name, dict)
    name = 'joint_positions.mat'
    dict = {label: value for label, value in zip(robot_model_init.joint_name_list, joint_positions_array)}
    savemat(path + name, dict)
    name = 'joint_velocities.mat'
    dict = {label: value for label, value in zip(robot_model_init.joint_name_list, joint_velocities_array)}
    savemat(path + name, dict)
 
    fitness_minimize = 12*total_misalignment + 100*(TIME_TH + mujoco_instance.get_simulation_frequency()-t)
    fitness = 1/fitness_minimize
    return fitness


## Definition of x_k 

# TSID parameters
# In the interval  (0,1000]
arms_gains = np.asarray([130,80,20,60])* 2.8 # from 0-3
legs_gains = np.asarray([150,20,20,180,200,230])* 3.2 # from 4-9
arms_gains_kd = 0.0*np.power(arms_gains, 1 / 2) / 10 # from 10-13
legs_gains_kd = np.power(legs_gains, 1 / 2) / 80 # from 14-19
foot_kp_lin = np.asarray([200.0]) # 20 
foot_kd_lin = np.asarray([7.0]) # 21
foot_kp_ang = np.asarray([100.0]) # 22
foot_kd_ang = np.asarray([12.0]) # 23
root_kp_ang = np.asarray([300.0]) # 24
root_kd_ang = np.asarray([10.0]) # 25
com_kp = np.asarray([9.0]) # 26
com_kd = np.asarray([7.0]) # 27 
tsid_param = np.concatenate((arms_gains,legs_gains,arms_gains_kd,legs_gains_kd, foot_kp_lin, foot_kd_lin, foot_kp_ang,
                             foot_kd_ang, root_kp_ang, root_kd_ang, com_kp, com_kd))

# MPC Parameters
# In the interval (0,1000]
com_weight = np.asarray([8,8,700]) # from 0-2
contact_position_weight = np.asarray([1e4])/1e1 # 3
force_rate_change_weight =np.asarray([10.0,10.0,10.0])/1e1 # from 4-6
angular_momentum_weight = np.asarray([1e5])/1e3 # 7 
contact_force_symmetry_weight = np.asarray([40.0])/1e2
mpc_param = np.concatenate((com_weight, contact_position_weight, force_rate_change_weight, angular_momentum_weight,contact_force_symmetry_weight))
x_k = np.concatenate((tsid_param, mpc_param))

low_limits_arms_gains = np.asarray([10,10,10,10]) # from 0-3
low_limits_legs_gains = np.asarray([10,10,10,10,10,10]) # from 4-9
low_limits_arms_gains_kd = 0.0*np.power(arms_gains, 1 / 2) / 10 # from 10-13
low_limits_legs_gains_kd = 0.0*np.power(legs_gains, 1 / 2) / 80 # from 14-19
low_limits_foot_kp_lin = np.asarray([50.0]) # 20 
low_limits_foot_kd_lin = np.asarray([1.0]) # 21
low_limits_foot_kp_ang = np.asarray([50.0]) # 22
low_limits_foot_kd_ang = np.asarray([1.0]) # 23
low_limits_root_kp_ang = np.asarray([50.0]) # 24
low_limits_root_kd_ang = np.asarray([1.0]) # 25
low_limits_com_kp = np.asarray([1.0]) # 26
low_limits_com_kd = np.asarray([1.0]) # 27 
low_limits_tsid_param = np.concatenate((low_limits_arms_gains,low_limits_legs_gains,low_limits_arms_gains_kd,low_limits_legs_gains_kd, low_limits_foot_kp_lin, low_limits_foot_kd_lin, low_limits_foot_kp_ang,
                             low_limits_foot_kd_ang, low_limits_root_kp_ang, low_limits_root_kd_ang, low_limits_com_kp, low_limits_com_kd))

high_selection_arms_gains = np.asarray([300,300,300,300]) # from 0-3
high_selection_legs_gains = np.asarray([300,300,300,300,300,300]) # from 4-9
high_selection_arms_gains_kd = 100*np.asarray([1,1,1,1]) # from 10-13
high_selection_legs_gains_kd = 100*np.asarray([1,1,1,1,1,1]) # from 14-19
high_selection_foot_kp_lin = np.asarray([500.0]) # 20 
high_selection_foot_kd_lin = np.asarray([50.0]) # 21
high_selection_foot_kp_ang = np.asarray([500.0]) # 22
high_selection_foot_kd_ang = np.asarray([50.0]) # 23
high_selection_root_kp_ang = np.asarray([500.0]) # 24
high_selection_root_kd_ang = np.asarray([50.0]) # 25
high_selection_com_kp = np.asarray([20.0]) # 26
high_selection_com_kd = np.asarray([20.0]) # 27 
high_selection_tsid_param = np.concatenate((high_selection_arms_gains,high_selection_legs_gains,high_selection_arms_gains_kd,high_selection_legs_gains_kd,high_selection_foot_kp_lin, high_selection_foot_kd_lin, high_selection_foot_kp_ang,
                             high_selection_foot_kd_ang, high_selection_root_kp_ang, high_selection_root_kd_ang, high_selection_com_kp, high_selection_com_kd))


# MPC Parameters
# In the interval (0,1000]
low_limits_com_weight = np.asarray([1,1,50]) # from 0-2
low_limits_contact_position_weight = np.asarray([1e1])/1e1 # 3
low_limits_force_rate_change_weight =np.asarray([1.0,1.0,1.0]) # from 4-6
low_limits_angular_momentum_weight = np.asarray([1e1])/1e3 # 7 
low_limits_contact_force_symmetry_weight = np.asarray([5.0])
low_limits_mpc_param = np.concatenate((low_limits_com_weight, low_limits_contact_position_weight, low_limits_force_rate_change_weight, low_limits_angular_momentum_weight,low_limits_contact_force_symmetry_weight))

high_limits_com_weight = np.asarray([50,50,1000]) # from 0-2
high_limits_contact_position_weight = np.asarray([1e6])/1e1 # 3
high_limits_force_rate_change_weight =np.asarray([100.0,100.0,100.0])/1e1 # from 4-6
high_limits_angular_momentum_weight = np.asarray([1e8])/1e3 # 7 
high_limits_contact_force_symmetry_weight = np.asarray([1000.0])/1e2
high_limits_mpc_param = np.concatenate((high_limits_com_weight, high_limits_contact_position_weight, high_limits_force_rate_change_weight, high_limits_angular_momentum_weight,high_limits_contact_force_symmetry_weight))


x_k_low_limits = np.concatenate((low_limits_tsid_param, low_limits_mpc_param))
x_k_high_limits = np.concatenate((high_selection_tsid_param,high_limits_mpc_param))
bounds = []
for index in range(len(x_k_low_limits)):
    print(index)
    bounds.append({"low":x_k_low_limits[index], "high": x_k_high_limits[index], 'step':1})

# N_CORE = 100 
# ga_instance = pygad.GA(
#     num_generations=3000,
#     num_parents_mating=100, #10 
#     sol_per_pop=100,
#     fitness_func=compute_fitness,
#     gene_space=bounds,
#     num_genes=len(x_k), 
#     parent_selection_type="tournament",
#     K_tournament=4,
#     crossover_type="two_points", # double point 
#     allow_duplicate_genes=True,
#     on_generation=on_generation,
#     parallel_processing=['process', N_CORE], 
#     keep_elitism = 10, 
#     mutation_type="random"
# )

# ga_instance.run()

population = pickle.load(open('results/genetic700.p', 'rb'))
compute_fitness(None, population[0], None)
# print(population[0])
# i = 0 

# for item in bounds: 
#     print(item)
# for item in population:
#     print(i)
#     i = i + 1 
#     compute_fitness(None, item, None)
# Fitness Computation 
# t_start = time.time()
# # f_k= compute_fitness(x_k)

# t_end = time.time()
# print("elapsed time", t_end-t_start)
# print("f(x_k)=",f_k)
# print("succeded = ", status)
