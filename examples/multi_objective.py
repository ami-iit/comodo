import random
from deap import base, creator, tools, algorithms
import numpy as np
import matplotlib.pyplot as plt

import multiprocessing
LIMIT_HARDWARE_LOW = 0.5
LIMIT_HARDWARE_HIGH = 2.0
# Limits for each gene in the chromosome
LIMITS = [(LIMIT_HARDWARE_LOW,LIMIT_HARDWARE_HIGH), (LIMIT_HARDWARE_LOW, LIMIT_HARDWARE_HIGH)]
NUM_GENES = len(LIMITS)
POPULATION_SIZE = 10
GENERATIONS = 5

# Define a fitness class with weights (-1.0, -1.0) for minimization problems
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))

# Define an individual as a list of floats
creator.create("Individual", list, fitness=creator.FitnessMulti)

def create_individual():
    return [random.uniform(lim[0], lim[1]) for lim in LIMITS]

# Register the individual and population creation functions in the toolbox
toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

## TODO improve this is only to see if things work 
# Comodo import
from comodo.mujocoSimulator.mujocoSimulator import MujocoSimulator
from comodo.robotModel.robotModel import RobotModel
from comodo.robotModel.createUrdf import createUrdf
from comodo.ergonomyPlanner.planErgonomyTrajectory import PlanErgonomyTrajectory
from comodo.payloadLiftingController.payloadLiftingController import PayloadLiftingController
from comodo.payloadLiftingController.payloadLiftingParameterTuning import PayloadLiftingControllerParameters
# Comodo import
from comodo.centroidalMPC.centroidalMPC import CentroidalMPC
from comodo.centroidalMPC.mpcParameterTuning import MPCParameterTuning
from comodo.TSIDController.TSIDParameterTuning import TSIDParameterTuning
from comodo.TSIDController.TSIDController import TSIDController

# General  import
import xml.etree.ElementTree as ET
import numpy as np
import tempfile
import urllib.request

# Getting stickbot urdf file and convert it to string
urdf_robot_file = tempfile.NamedTemporaryFile(mode="w+")
url = "https://raw.githubusercontent.com/icub-tech-iit/ergocub-gazebo-simulations/master/models/stickBot/model.urdf"
urllib.request.urlretrieve(url, urdf_robot_file.name)
# Load the URDF file
tree = ET.parse(urdf_robot_file.name)
root = tree.getroot()
# Convert the XML tree to a string
robot_urdf_string_original = ET.tostring(root)
create_urdf_instance = createUrdf(
    original_urdf_path=urdf_robot_file.name, save_gazebo_plugin=False
)
# Define parametric links and controlled joints
link_names = ["upper_arm", "lower_leg"]
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
    "l_ankle_roll",
]


## This will be to be better organize

def compute_fitness_payload_lifting(modifications,motors_param): 
    # Modify the robot model and initialize
    create_urdf_instance.modify_lengths(modifications)
    urdf_robot_string = create_urdf_instance.write_urdf_to_file()
    create_urdf_instance.reset_modifications()
    robot_model_init = RobotModel(urdf_robot_string, "stickBot", joint_name_list)
    robot_model_init.set_with_box(True)
    ## Planning the ergonomic trajectory for payload lifting 
    plan_trajectory = PlanErgonomyTrajectory(robot_model=robot_model_init)

    if(not(plan_trajectory.plan_trajectory())):
       raise ValueError("unable to plan the trajectory ")
       # Define simulator and set initial position
    mujoco_instance = MujocoSimulator()
    mujoco_instance.load_model(
        robot_model_init, s=plan_trajectory.s_opti[0], xyz_rpy=plan_trajectory.xyz_rpy[0], kv_motors=motors_param["k_v"], Im=motors_param["I_m"]
    )
    s, ds, tau = mujoco_instance.get_state()
    t = mujoco_instance.get_simulation_time()
    H_b = mujoco_instance.get_base()
    w_b = mujoco_instance.get_base_velocity()
    mujoco_instance.set_visualize_robot_flag(False)
    ## Defining controller 
    lifting_controller_instance = PayloadLiftingController(frequency=0.01, robot_model=robot_model_init)
    lifting_controller_instance.set_state(s,ds,t)
    param_lifting_controller = PayloadLiftingControllerParameters()
    lifting_controller_instance.set_control_gains(
    postural_Kp=param_lifting_controller.joints_Kp_parameters,
    CoM_Kp=param_lifting_controller.CoM_Kp,
    CoM_Ki=param_lifting_controller.CoM_Ki,
    )
    lifting_controller_instance.set_time_interval_state_machine(1, 5, 5)
    lifting_controller_instance.initialize_state_machine(
        joint_pos_1=plan_trajectory.s_opti[1], joint_pos_2=plan_trajectory.s_opti[2]
    )
    n_step = int(lifting_controller_instance.frequency/mujoco_instance.get_simulation_frequency())
    TIME_TH = 20

    while(t<TIME_TH):     
        # Updating the states 
        s,ds,tau = mujoco_instance.get_state()
        t = mujoco_instance.get_simulation_time()
        lifting_controller_instance.set_state(s,ds,t)

        # Running the controller 
        controller_succed= lifting_controller_instance.run()
        if(not(controller_succed)): 
            break
        tau = lifting_controller_instance.get_torque()
        mujoco_instance.set_input(tau)
        mujoco_instance.step(int(n_step))
    # Closing visualization
    mujoco_instance.close_visualization()
    return TIME_TH -t

def compute_fitness_walking(modifications, motors_param):
    # Modify the robot model and initialize
    create_urdf_instance.modify_lengths(modifications)
    urdf_robot_string = create_urdf_instance.write_urdf_to_file()
    create_urdf_instance.reset_modifications()
    robot_model_init = RobotModel(urdf_robot_string, "stickBot", joint_name_list)
    s_des, xyz_rpy, H_b = robot_model_init.compute_desired_position_walking()
    # Define simulator and set initial position
    mujoco_instance = MujocoSimulator()
    mujoco_instance.load_model(
        robot_model_init, s=s_des, xyz_rpy=xyz_rpy, kv_motors=motors_param["k_v"], Im=motors_param["I_m"]
    )
    s, ds, tau = mujoco_instance.get_state()
    t = mujoco_instance.get_simulation_time()
    H_b = mujoco_instance.get_base()
    w_b = mujoco_instance.get_base_velocity()
    mujoco_instance.set_visualize_robot_flag(False)
    # Define the controller parameters  and instantiate the controller
    # Controller Parameters
    tsid_parameter = TSIDParameterTuning()
    mpc_parameters = MPCParameterTuning()

    # TSID Instance
    TSID_controller_instance = TSIDController(frequency=0.01, robot_model=robot_model_init)
    TSID_controller_instance.define_tasks(tsid_parameter)
    TSID_controller_instance.set_state_with_base(s, ds, H_b, w_b, t)

    # MPC Instance
    step_lenght = 0.1
    mpc = CentroidalMPC(robot_model=robot_model_init, step_length=step_lenght)
    mpc.intialize_mpc(mpc_parameters=mpc_parameters)

    # Set desired quantities
    mpc.configure(s_init=s_des, H_b_init=H_b)
    TSID_controller_instance.compute_com_position()
    mpc.define_test_com_traj(TSID_controller_instance.COM.toNumPy())

    # Set initial robot state  and plan trajectories
    mujoco_instance.step(1)

    # Reading the state
    s, ds, tau = mujoco_instance.get_state()
    H_b = mujoco_instance.get_base()
    w_b = mujoco_instance.get_base_velocity()
    t = mujoco_instance.get_simulation_time()

    # MPC
    mpc.set_state_with_base(s=s, s_dot=ds, H_b=H_b, w_b=w_b, t=t)
    mpc.initialize_centroidal_integrator(s=s, s_dot=ds, H_b=H_b, w_b=w_b, t=t)
    mpc_output = mpc.plan_trajectory()
    # Set loop variables
    TIME_TH = 20

    # Define number of steps
    n_step = int(
        TSID_controller_instance.frequency / mujoco_instance.get_simulation_frequency()
    )
    n_step_mpc_tsid = int(mpc.get_frequency_seconds() / TSID_controller_instance.frequency)

    counter = 0
    mpc_success = True
    energy_tot = 0.0
    succeded_controller = True
    # Simulation-control loop
    while t < TIME_TH:
        # Reading robot state from simulator
        s, ds, tau = mujoco_instance.get_state()
        energy_i = np.linalg.norm(tau)
        H_b = mujoco_instance.get_base()
        w_b = mujoco_instance.get_base_velocity()
        t = mujoco_instance.get_simulation_time()

        # Update TSID
        TSID_controller_instance.set_state_with_base(s=s, s_dot=ds, H_b=H_b, w_b=w_b, t=t)

        # MPC plan
        if counter == 0:
            mpc.set_state_with_base(s=s, s_dot=ds, H_b=H_b, w_b=w_b, t=t)
            mpc.update_references()
            mpc_success = mpc.plan_trajectory()
            mpc.contact_planner.advance_swing_foot_planner()
            if not (mpc_success):
                print("MPC failed")
                break

        # Reading new references
        com, dcom, forces_left, forces_right, ang_mom = mpc.get_references()
        left_foot, right_foot = mpc.contact_planner.get_references_swing_foot_planner()

        # Update references TSID
        TSID_controller_instance.update_task_references_mpc(
            com=com,
            dcom=dcom,
            ddcom=np.zeros(3),
            left_foot_desired=left_foot,
            right_foot_desired=right_foot,
            s_desired=np.array(s_des),
            wrenches_left=forces_left,
            wrenches_right=forces_right,
        )

        # Run control
        succeded_controller = TSID_controller_instance.run()

        if not (succeded_controller):
            print("Controller failed")
            break

        tau = TSID_controller_instance.get_torque()

        # Step the simulator
        mujoco_instance.set_input(tau)
        mujoco_instance.step_with_motors(n_step=n_step, torque=tau)
        counter = counter + 1

        if counter == n_step_mpc_tsid:
            counter = 0
    # Closing visualization
    mujoco_instance.close_visualization()
    return TIME_TH-t 

def evaluate(individual):
    # Define the robot modifications
    modifications = {}
    for idx,item in enumerate(link_names):
        left_leg_item = "l_" + item
        right_leg_item = "r_" + item
        modifications.update({left_leg_item: individual[idx]})
        modifications.update({right_leg_item: individual[idx]})
    # Motors Parameters
    Im_arms = 1e-3 * np.ones(4)  # from 0-4
    Im_legs = 1e-3 * np.ones(6)  # from 5-10
    kv_arms = 0.001 * np.ones(4)  # from 11-14
    kv_legs = 0.001 * np.ones(6)  # from 20

    Im = np.concatenate((Im_arms, Im_arms, Im_legs, Im_legs))
    kv = np.concatenate((kv_arms, kv_arms, kv_legs, kv_legs))
    motors_param = {}
    motors_param.update({"I_m": Im})
    motors_param.update({"k_v":kv})
    metric1 = compute_fitness_payload_lifting(modifications,motors_param)
    metric2 = compute_fitness_walking(modifications, motors_param)
    print("METRIC 1", metric1)
    print("METRIC 2", metric2)

    return metric1, metric2

toolbox.register("evaluate", evaluate)

def mate(ind1, ind2):
    tools.cxTwoPoint(ind1, ind2)
    for i, (low, up) in enumerate(LIMITS):
        ind1[i] = min(max(ind1[i], low), up)
        ind2[i] = min(max(ind2[i], low), up)
    return ind1, ind2

def mutate(ind):
    for i, (low, up) in enumerate(LIMITS):
        if random.random() < 0.2:  # Mutation probability
            ind[i] += random.gauss(0, 1)
            ind[i] = min(max(ind[i], low), up)
    return ind,

toolbox.register("mate", mate)
toolbox.register("mutate", mutate)
toolbox.register("select", tools.selNSGA2)

def main():
    # Initialize multiprocessing pool
    pool = multiprocessing.Pool(10)
    toolbox.register("map", pool.map)
    population = toolbox.population(n=POPULATION_SIZE)
    
    # Use the NSGA-II algorithm
    algorithms.eaMuPlusLambda(population, toolbox, mu=POPULATION_SIZE, lambda_=POPULATION_SIZE, 
                              cxpb=0.6, mutpb=0.3, ngen=GENERATIONS, 
                              stats=None, halloffame=None, verbose=True)
    # Close the pool
    pool.close()
    pool.join()
    return population

if __name__ == "__main__":
    final_population = main()
    
    # Extract the Pareto front
    pareto_front = tools.sortNondominated(final_population, len(final_population), first_front_only=True)[0]

    # Print the solutions in the Pareto front
    for ind in pareto_front:
        print(f"Individual: {ind}, Fitness: {ind.fitness.values}")

        # Plot the Pareto front
    fitnesses = [ind.fitness.values for ind in pareto_front]
    fitness1 = [f[0] for f in fitnesses]
    fitness2 = [f[1] for f in fitnesses]
    
    plt.scatter(fitness1, fitness2, c='red')
    plt.xlabel('Metric 1')
    plt.ylabel('Metric 2')
    plt.title('Pareto Front')
    plt.grid(True)
    plt.show()