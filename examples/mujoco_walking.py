from comodo.mujocoSimulator.mujocoSimulator import MujocoSimulator
from comodo.robotModel.robotModel import RobotModel
from comodo.robotModel.createUrdf import createUrdf
from comodo.centroidalMPC.centroidalMPC import CentroidalMPC
from comodo.centroidalMPC.mpcParameterTuning import MPCParameterTuning
from comodo.TSIDController.TSIDParameterTuning import TSIDParameterTuning
from comodo.TSIDController.TSIDController import TSIDController

import xml.etree.ElementTree as ET
import numpy as np
import tempfile
# from git import Repo

## Defining the urdf path of both the startin, del and the modified one
# common_path = os.path.dirname(os.path.abspath(__file__))
urdf_path_original = "/home/carlotta/iit_ws/element_hardware-intelligence/Software/OptimizationControlAndHardware/models/model.urdf"
# Getting stickbot urdf file
# temp_dir = tempfile.TemporaryDirectory()
# git_url = "https://github.com/icub-tech-iit/ergocub-gazebo-simulations.git"
# Repo.clone_from(git_url, temp_dir.name)
# urdf_path_original = temp_dir.name + "/models/stickBot/model.urdf"
# Load the URDF file
tree = ET.parse(urdf_path_original)
root = tree.getroot()

# Convert the XML tree to a string
robot_urdf_string_original = ET.tostring(root)

create_urdf_instance = createUrdf(original_urdf_path=urdf_path_original,save_gazebo_plugin = False)


legs_link_names = ["hip_3","lower_leg"]
joint_name_list = [
    "r_shoulder_pitch",
    "r_shoulder_roll",
    "r_shoulder_yaw",
    "r_elbow",
    "l_shoulder_pitch",
    "l_shoulder_roll",
    "l_shoulder_yaw",
    "l_elbow",
    "r_hip_pitch",
    "r_hip_roll",
    "r_hip_yaw",
    "r_knee",
    "r_ankle_pitch",
    "r_ankle_roll",
    "l_hip_pitch",
    "l_hip_roll",
    "l_hip_yaw",
    "l_knee",
    "l_ankle_pitch",
    "l_ankle_roll"
]

modifications = {}
for item in legs_link_names:    
    left_leg_item = "l_" + item 
    right_leg_item = "r_" + item 
    modifications.update({left_leg_item:1.2})
    modifications.update({right_leg_item:1.2})

# Defining the robot model 
create_urdf_instance.modify_lengths(modifications)
urdf_robot_string= create_urdf_instance.write_urdf_to_file()
create_urdf_instance.reset_modifications()
# Instantiate the robot model

robot_model_init = RobotModel(urdf_robot_string, "stickBot", joint_name_list)
# Defining the TSID Controller 
tsid_parameter = TSIDParameterTuning()
# tsid_parameter.set_from_x_k(tsid_gain)
mpc_parameters = MPCParameterTuning()
# mpc_parameters.set_from_xk(mpc_gains)
TSID_controller_instance = TSIDController(frequency=0.01,robot_model=robot_model_init)

TSID_controller_instance.define_tasks(tsid_parameter)

# Defining the simulator 
mujoco_instance = MujocoSimulator()

s_des, xyz_rpy, H_b= robot_model_init.compute_desired_position_walking()
step_lenght = 0.1 
mujoco_instance.load_model(robot_model_init, s= s_des, xyz_rpy=xyz_rpy)
s,ds,tau = mujoco_instance.get_state()
t = mujoco_instance.get_simulation_time()
H_b = mujoco_instance.get_base()
w_b = mujoco_instance.get_base_velocity()

TSID_controller_instance.set_state_with_base(s,ds,H_b,w_b,t)
mpc = CentroidalMPC(robot_model=robot_model_init, step_length=step_lenght)

n_step = int(TSID_controller_instance.frequency/mujoco_instance.get_simulation_frequency())
n_step_mpc_tsid = int(mpc.get_frequency_seconds()/TSID_controller_instance.frequency)


mpc.intialize_mpc(mpc_parameters=mpc_parameters)
mpc.configure(s_init=s_des, H_b_init=H_b)
TSID_controller_instance.compute_com_position()
mpc.define_test_com_traj(TSID_controller_instance.COM.toNumPy())
TIME_TH = 20
mujoco_instance.set_visualize_robot_flag(True)
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
energy_tot = 0.0
succeded_controller  = True
while(t<TIME_TH):
    # Reading robot state from simulator
    s,ds,tau = mujoco_instance.get_state()
    energy_i =np.linalg.norm(tau)
    H_b = mujoco_instance.get_base()
    w_b = mujoco_instance.get_base_velocity()
    t = mujoco_instance.get_simulation_time()
    TSID_controller_instance.set_state_with_base(s=s,s_dot=ds,H_b=H_b, w_b=w_b,t=t)

    if(counter == 0):
        mpc.set_state_with_base(s=s, s_dot=ds, H_b=H_b, w_b=w_b,t=t)
        mpc.update_references()
        mpc_success = mpc.plan_trajectory()
        mpc.contact_planner.advance_swing_foot_planner()
        if(not(mpc_success)): 
            print("MPC failed")
            break

    com, dcom,forces_left, forces_right = mpc.get_references()
    left_foot, right_foot = mpc.contact_planner.get_references_swing_foot_planner()
    TSID_controller_instance.update_task_references_mpc(com=com, dcom=dcom,ddcom=np.zeros(3),left_foot_desired=left_foot, right_foot_desired=right_foot,s_desired=np.array(s_des), wrenches_left = forces_left, wrenches_right = forces_right)
    succeded_controller = TSID_controller_instance.run()
    
    
    if(not(succeded_controller)): 
        print("Controller failed")
        break
    
    tau = TSID_controller_instance.get_torque()
    mujoco_instance.set_input(tau)
    # mujoco_instance.step(n_step=n_step)
    mujoco_instance.step_with_motors(n_step=n_step, torque=tau)
    counter = counter+ 1 

    if(counter == n_step_mpc_tsid): 
        counter = 0 
mujoco_instance.close_visualization()

