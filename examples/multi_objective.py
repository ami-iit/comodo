import random
from deap import base, creator, tools, algorithms
import numpy as np
import matplotlib.pyplot as plt
import pickle  # For saving data
from include.chromosomeGenerator import ChromosomeGenerator, SubChromosome, NameChromosome 
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import os 
from include.dataBaseFitnessFunction import DatabaseFitnessFunction
import copy
from matplotlib import cm, colors  # Import color map

import idyntree.bindings as iDynTree
import time 

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

POPULATION_SIZE = 100
GENERATIONS = 3000
common_path = os.path.dirname(os.path.abspath(__file__))
SAVE_PATH = common_path + "/result/"
os.mkdir(SAVE_PATH)
 
chrom_generator = ChromosomeGenerator()
link_names = ["root_link","upper_arm", "forearm","lower_leg","hip_3"]

#**************************************************************HARDWARE**************************************************************#

## length multiplier 
length_multiplier = SubChromosome()
length_multiplier.type = NameChromosome.LENGTH
length_multiplier.isFloat = True
length_multiplier.limits = [0.5,2.5]
length_multiplier.dimension = len(link_names)
chrom_generator.add_parameters(length_multiplier)

# density 
density_param = SubChromosome()
density_param.type = NameChromosome.DENSITY
density_param.isDiscrete = True 
density_param.feasible_set= [ 2129.2952964, 1199.07622408, 893.10763518, 626.60271872, 1661.68632652, 727.43130782, 600.50011475, 2222.0327914,1064.6476482, 599.53811204, 446.55381759, 313.30135936, 830.84316326, 363.71565391, 300.250057375, 1111.0163957,4258.5905928, 2398.15244816, 1786.21527036, 1253.20543744, 3323.37265304, 1454.86261564, 1201.0002295, 4444.0655828]
density_param.dimension = len(link_names)
chrom_generator.add_parameters(density_param)

## joint type 
jointTypeCh = SubChromosome()
jointTypeCh.type = NameChromosome.JOINT_TYPE
jointTypeCh.dimension = 9
jointTypeCh.isDiscrete = True 
jointTypeCh.feasible_set = [0,1]
chrom_generator.add_parameters(jointTypeCh)

## motors inerita 
motors_inertia = SubChromosome()
motors_inertia.type = NameChromosome.MOTOR_INERTIA
motors_inertia.dimension = 10 # TODO find a way to automatically ensure symetry  
motors_inertia.isFloat = True 
motors_inertia.limits=[1e-10, 1e-1]
chrom_generator.add_parameters(motors_inertia)

## motors friction 
motors_friction = SubChromosome()
motors_friction.type = NameChromosome.MOTOR_FRICTION
motors_friction.dimension = 10 # TODO find a way to automatically ensure symetry  
motors_friction.isFloat = True 
motors_friction.limits=[1e-6, 0.1]
chrom_generator.add_parameters(motors_friction)


#**************************************************************CONTROL**************************************************************#

## TSID 
tsid_parameters = SubChromosome()
tsid_parameters.type = NameChromosome.TSID_PARAMTERES
tsid_parameters.isFloat = True
tsid_parameters.dimension = 4 
tsid_parameters.limits = np.array([[4, 50],[100, 400],[20, 100],[4, 50]])
chrom_generator.add_parameters(tsid_parameters)

## MPC 
mpc_parameters = SubChromosome()
mpc_parameters.type = NameChromosome.MPC_PARAMETERS
mpc_parameters.isFloat = True 
mpc_parameters.dimension = 7
mpc_parameters.limits = np.array([[5,100],[20,300],[10,340],[10,340],[10,340],[10,340],[10,340],[10,340]])
chrom_generator.add_parameters(mpc_parameters)

## PAYLOAD LIFTING
lifting_parameters = SubChromosome()
lifting_parameters.type = NameChromosome.PAYLOAD_LIFTING
lifting_parameters.isFloat = True
lifting_parameters.dimension = 6
lifting_parameters.limits = np.array([[0.5,10],[0.5,10],[0.01, 0.05],[0.01, 0.05],[20,100],[20,100]])
chrom_generator.add_parameters(lifting_parameters)

# Define a fitness class with weights (-1.0, -1.0) for minimization problems
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))

# Define an individual as a list of floats
creator.create("Individual", list, fitness=creator.FitnessMulti)

# Register the individual and population creation functions in the toolbox
toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, chrom_generator.generate_chromosome)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

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
# urdf_robot_file = tempfile.NamedTemporaryFile(mode="w+")
# url = "https://raw.githubusercontent.com/icub-tech-iit/ergocub-gazebo-simulations/master/models/stickBot/model.urdf"
# urllib.request.urlretrieve(url, urdf_robot_file.name)
urdf_robot_file_name = "/home/iit.local/csartore/software/multi_objective_optimization/comodo/examples/model.urdf"
# Load the URDF file
tree = ET.parse(urdf_robot_file_name)
root = tree.getroot()
# Convert the XML tree to a string
robot_urdf_string_original = ET.tostring(root)
create_urdf_instance = createUrdf(
    original_urdf_path=urdf_robot_file_name, save_gazebo_plugin=False
)
# Define parametric links and controlled joints
joint_name_list = [
    "r_shoulder_pitch", #0 
    "r_shoulder_roll",#1 
    "r_shoulder_yaw",#2 
    "r_elbow",#3 
    "l_shoulder_pitch",#4 
    "l_shoulder_roll",#5 
    "l_shoulder_yaw",#6 
    "l_elbow",#7 
    "r_hip_pitch",#8 
    "r_hip_roll",#9 
    "r_hip_yaw",#10 
    "r_knee",#11 
    "r_ankle_pitch",#12 
    "r_ankle_roll",#13 
    "l_hip_pitch",#14 
    "l_hip_roll",#15 
    "l_hip_yaw",#16 
    "l_knee",#17 
    "l_ankle_pitch", #18 
    "l_ankle_roll",#19 
]
torso_link = ['root_link', 'torso_1', 'torso_2', 'chest']

## This will be to be better organize

def compute_fitness_payload_lifting(modifications_length, modifications_densities,motors_param,joint_name_list_updated,joint_active, lifting_ch): 
    # Modify the robot model and initialize
    create_urdf_instance.modify_lengths(modifications_length)
    create_urdf_instance.modify_densities(modifications_densities)
    urdf_robot_string = create_urdf_instance.write_urdf_to_file()
    create_urdf_instance.reset_modifications()
    robot_model_init = RobotModel(urdf_robot_string, "stickBot", joint_name_list_updated)
    robot_model_init.set_with_box(True)
    ## Planning the ergonomic trajectory for payload lifting 
    plan_trajectory = PlanErgonomyTrajectory(robot_model=robot_model_init)
    TIME_TH = 16

    if(not(plan_trajectory.plan_trajectory())):
        return 250
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
    param_lifting_controller = PayloadLiftingControllerParameters(joint_active)
    param_lifting_controller.set_from_xk(lifting_ch)
    lifting_controller_instance.set_control_gains(
        param_lifting_controller
    )
    lifting_controller_instance.set_time_interval_state_machine(1, 5, 5)
    lifting_controller_instance.initialize_state_machine(
        joint_pos_1=plan_trajectory.s_opti[1], joint_pos_2=plan_trajectory.s_opti[2]
    )
    n_step = int(lifting_controller_instance.frequency/mujoco_instance.get_simulation_frequency())
    torque = []
    state_machine_on = True
    target_number_state = 5 
    while(state_machine_on):    
        # Updating the states 
        # print(t)
        s,ds,tau = mujoco_instance.get_state()
        # print("tracking error", np.linalg.norm(s-plan_trajectory.s_opti[0]))
        t = mujoco_instance.get_simulation_time()
        lifting_controller_instance.set_state(s,ds,t)
        torque.append(np.linalg.norm(tau))
        # Running the controller 
        state_machine_on = lifting_controller_instance.update_desired_configuration()
        controller_succed= lifting_controller_instance.run()
        if(not(controller_succed)):
            # fitness_pl = 10*(TIME_TH-t) + 0.1*np.mean(torque)
            # return fitness_pl
            break 
        tau = lifting_controller_instance.get_torque()
        mujoco_instance.set_input(tau)
        mujoco_instance.step(int(n_step))
    # Closing visualization
    current_state = lifting_controller_instance.state_machine.current_state
    if(not(state_machine_on)):
        current_state = lifting_controller_instance.state_machine.current_state +1
    achieve_goal = target_number_state - current_state
    torque_mean = np.mean(torque)
    torque_diff = np.mean(np.diff(torque))
    weight_achieve_goal = 30 
    weight_torque_mean  = 0.5 
    weight_torque_diff = 100 
    # print("achieve goal", achieve_goal)
    # print("torque_mean", torque_mean)
    # print("torque_diff", torque_diff)
    mujoco_instance.close_visualization()
    fitness_pl = weight_achieve_goal*achieve_goal + weight_torque_mean*torque_mean + weight_torque_diff*torque_diff
    if(fitness_pl>250):
        fitness_pl = 250
    return fitness_pl

def compute_fitness_walking(modifications_length, modifications_densities, motors_param,joint_name_list_updated,joint_active, mpc_chr, tsid_chr):
    # Modify the robot model and initialize
     # Set loop variables
    TIME_TH = 15
    create_urdf_instance.modify_lengths(modifications_length)
    create_urdf_instance.modify_densities(modifications_densities)
    urdf_robot_string = create_urdf_instance.write_urdf_to_file()
    create_urdf_instance.reset_modifications()
    robot_model_init = RobotModel(urdf_robot_string, "stickBot", joint_name_list_updated)
    solved, s_des, xyz_rpy, H_b = robot_model_init.compute_desired_position_walking()
    if(not(solved)):
        fitness_wal = 600
        return fitness_wal
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
    tsid_parameter = TSIDParameterTuning(joint_active)
    tsid_parameter.set_from_x_k(tsid_chr)
    mpc_parameters = MPCParameterTuning()
    mpc_parameters.set_from_xk(mpc_chr)
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
    
    # Define number of steps
    n_step = int(
        TSID_controller_instance.frequency / mujoco_instance.get_simulation_frequency()
    )
    n_step_mpc_tsid = int(mpc.get_frequency_seconds() / TSID_controller_instance.frequency)
    counter = 0
    mpc_success = True
    succeded_controller = True
    torque = []
    # Simulation-control loop
    while t < TIME_TH:
        # print(t)
        # Reading robot state from simulator
        s, ds, tau = mujoco_instance.get_state()
        H_b = mujoco_instance.get_base()
        w_b = mujoco_instance.get_base_velocity()
        t = mujoco_instance.get_simulation_time()
        torque.append(np.linalg.norm(tau))
        # Update TSID
        TSID_controller_instance.set_state_with_base(s=s, s_dot=ds, H_b=H_b, w_b=w_b, t=t)
        # TSID_controller_instance.compute_com_position()
        # print(mpc.final_goal[0]-TSID_controller_instance.COM.toNumPy()[0])
        # MPC plan
        if counter == 0:
            mpc.set_state_with_base(s=s, s_dot=ds, H_b=H_b, w_b=w_b, t=t)
            mpc.update_references()
            mpc_success = mpc.plan_trajectory()
            mpc.contact_planner.advance_swing_foot_planner()
            if not (mpc_success):
                print("MPC failed")
                # fitness_wal = 10*(TIME_TH -t) + 0.1*np.mean(torque)
                # return fitness_wal
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
            # fitness_wal = 10*(TIME_TH -t) + 0.1*np.mean(torque)
            # return fitness_wal
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
    TSID_controller_instance.compute_com_position()
    achieve_goal = abs(mpc.final_goal[0]-TSID_controller_instance.COM.toNumPy()[0])  # Walk until reference final position for the com 
    if(succeded_controller):
        achieve_goal = 0.0
    torque_mean = np.mean(torque)
    torque_diff = np.mean(np.diff(torque))
    error_norm =  np.linalg.norm(mpc.final_goal- TSID_controller_instance.COM.toNumPy())
    time_diff = TIME_TH-t
    weight_achieve_goal = 150  
    weight_torque_mean  = 0.5 
    weight_torque_diff = 50 

    # print("achieve goal", achieve_goal)
    # print("torque_mean", torque_mean)
    # print("torque_diff", torque_diff)
    # print("final goal",mpc.final_goal)
    # print("com measured", TSID_controller_instance.COM.toNumPy())
    # print("error norm", error_norm)
    # print("Time diffeence times error",weight_achieve_goal*time_diff*error_norm)
    fitness_wal = weight_achieve_goal*time_diff*error_norm + weight_torque_diff*torque_diff + weight_torque_mean*torque_mean
    # if(fitness_wal>200):
    #     fitness_wal = 200
    # print("fitness_walking",fitness_wal)
    return fitness_wal 

def evaluate_from_database(individual):
    data_base = DatabaseFitnessFunction(SAVE_PATH + "database")
    fit1, fit2 = data_base.get_fitness_value(individual)
    return fit1, fit2

def evaluate(individual):
    # Define the robot modifications
    dict_return = chrom_generator.get_chromosome_dict(individual)
    links_length = dict_return[NameChromosome.LENGTH]
    density = dict_return[NameChromosome.DENSITY]
    joint_active_temp = dict_return[NameChromosome.JOINT_TYPE]
    # joint_active_temp[3] = 1 # the elbow always active for now, if not there are issues in attaching the box to the hand in mujoco 
    # motor_inertia_param_temp = dict_return[0.8880754212296234, 1.2677991421185684, 0.5560596513676981, 0.869871841987746, 1.734054392727016, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1][NameChromosome.MOTOR_INERTIA]
    motor_friction_temp = dict_return[NameChromosome.MOTOR_FRICTION]
    motor_inertia_temp = dict_return[NameChromosome.MOTOR_INERTIA]
    joint_name_list_updated =[]
    Im = []
    Kv = []
    joint_active = []
    joint_arms = []
    joint_arms.extend(joint_active_temp[:3])
    joint_arms.extend([1]) # The elbow is always active 
    joint_active.extend(joint_arms)
    joint_active.extend(joint_arms)
    joint_active.extend(joint_active_temp[3:])
    joint_active.extend(joint_active_temp[3:])
    motor_inertia_param = []
    motor_inertia_param.extend(motor_inertia_temp[:4])
    motor_inertia_param.extend(motor_inertia_temp[:4])
    motor_inertia_param.extend(motor_inertia_temp[4:])
    motor_inertia_param.extend(motor_inertia_temp[4:])
    motor_friction_param = []
    motor_friction_param.extend(motor_friction_temp[:4])
    motor_friction_param.extend(motor_friction_temp[:4])
    motor_friction_param.extend(motor_friction_temp[4:])
    motor_friction_param.extend(motor_friction_temp[4:])
    
    # This is needed because not all joints will be active, and only the active one will have motor characteristics 
    for idx_joint in range(len(joint_active)):
        if(joint_active[idx_joint]== 1):
            joint_name_list_updated.append(joint_name_list[idx_joint])
            Im.append(motor_inertia_param[idx_joint])
            Kv.append(motor_friction_param[idx_joint])


    modifications = {}
    modifications_density = {}
    for idx,item in enumerate(link_names):
        if(item in torso_link):
            modifications.update({item: links_length[idx]})
            modifications_density.update({item:density[idx]})
        else:    
            left_leg_item = "l_" + item
            right_leg_item = "r_" + item
            
            modifications.update({left_leg_item: links_length[idx]})
            modifications.update({right_leg_item: links_length[idx]})
            modifications_density.update({left_leg_item: density[idx]})
            modifications_density.update({right_leg_item: density[idx]})
    

    motors_param = {}
    motors_param.update({"I_m": Im})
    motors_param.update({"k_v":Kv})
    mpc_p = dict_return[NameChromosome.MPC_PARAMETERS]
    tsid_p = dict_return[NameChromosome.TSID_PARAMTERES]
    lifting_p = dict_return[NameChromosome.PAYLOAD_LIFTING]
    metric2 = compute_fitness_walking(modifications,modifications_density, motors_param,joint_name_list_updated,joint_active, mpc_p, tsid_p)
    metric1 = compute_fitness_payload_lifting(modifications,modifications_density,motors_param,joint_name_list_updated,joint_active, lifting_p)
    # metric1 = 0 
    return metric1, metric2

toolbox.register("evaluate", evaluate_from_database)

def compute_fitness_list_chromosome(list_chromosome):
    n_processors = len(list_chromosome)
    # compute fitness function
    with multiprocessing.Pool(processes=n_processors) as pool:
        output_map = pool.map(
            evaluate,
            list_chromosome,
        )

    data_base = DatabaseFitnessFunction(SAVE_PATH + "database")
    for i, chromosome in enumerate(list_chromosome):
        data_base.update(chromosome=chromosome,fitness_value_1= output_map[i][0],fitness_value_2=output_map[i][1])

def mate(ind1, ind2):
    tools.cxTwoPoint(ind1, ind2)
    # ind1 = chrom_generator.check_chromosome_in_set(ind1)
    # ind2 = chrom_generator.check_chromosome_in_set(ind2)
    return ind1, ind2

def mutate(ind):
    #TODO to be implemented in the chromosome generator 
    ind_new = chrom_generator.generate_chromosome()
    idx= random.randint(0, len(ind_new))
    if(random.random()>0.5):
        ind[:idx] = ind_new[:idx]
    else:
         ind[idx:] = ind_new[idx:]
    return ind

toolbox.register("mate", mate)
toolbox.register("mutate", mutate)
toolbox.register("select", tools.selNSGA2)

def main():
    population = toolbox.population(n=POPULATION_SIZE)
    all_generations = []  # List to store all generations
    invalid_chromo = [chromo for chromo in population if not chromo.fitness.valid]
    compute_fitness_list_chromosome(invalid_chromo)
    fitnesses = toolbox.map(toolbox.evaluate, invalid_chromo)
    for chromo, fit in zip(invalid_chromo, fitnesses):
        chromo.fitness.values = fit
    population = toolbox.select(population, POPULATION_SIZE)

    # Use the NSGA-II algorithm
    for gen in range(GENERATIONS):
        # Select the next generation individuals
        offspring = tools.selTournamentDCD(population, len(population))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
        
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values
    
        for mutant in offspring:
            if random.random() < 0.6:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
       
        compute_fitness_list_chromosome(invalid_ind)
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # Replace the old population with the offspring
        population = toolbox.select(population + offspring, len(population))
        # population[:] = offspring
        gen_temp = [toolbox.clone(ind) for ind in population]
        gen_path = SAVE_PATH+"/generation" + str(gen) + ".p"

        with(open(gen_path,"wb")) as f : 
            pickle.dump(gen_temp, f)
        # Append the current generation to the list
        all_generations.append([toolbox.clone(ind) for ind in population])
    
    # Close the pool
    # pool.close()
    # pool.join()

    with open("all_generations.pkl", "wb") as f:
        pickle.dump(all_generations, f)
    
    return population, all_generations

def print_main_hardware_charact(list_chromosome):
    masses = []
    heights = []
    com_z_values = []
    avg_motor_friction = []
    avg_motor_inertia = []
    active_joint_counts = []

    item_idx = 0 
    for item in list_chromosome:
        item_idx += 1
        # Define the robot modifications
        dict_return = chrom_generator.get_chromosome_dict(item)
        links_length = dict_return[NameChromosome.LENGTH]
        density = dict_return[NameChromosome.DENSITY]
        joint_active_temp = dict_return[NameChromosome.JOINT_TYPE]
        motor_friction_temp = dict_return[NameChromosome.MOTOR_FRICTION]
        motor_inertia_temp = dict_return[NameChromosome.MOTOR_INERTIA]
        joint_name_list_updated =[]
        Im = []
        Kv = []
        joint_active = []
        joint_arms = []
        joint_arms.extend(joint_active_temp[:3])
        joint_arms.extend([1]) # The elbow is always active 
        joint_active.extend(joint_arms)
        joint_active.extend(joint_arms)
        joint_active.extend(joint_active_temp[3:])
        joint_active.extend(joint_active_temp[3:])
        motor_inertia_param = []
        motor_inertia_param.extend(motor_inertia_temp[:4])
        motor_inertia_param.extend(motor_inertia_temp[:4])
        motor_inertia_param.extend(motor_inertia_temp[4:])
        motor_inertia_param.extend(motor_inertia_temp[4:])
        motor_friction_param = []
        motor_friction_param.extend(motor_friction_temp[:4])
        motor_friction_param.extend(motor_friction_temp[:4])
        motor_friction_param.extend(motor_friction_temp[4:])
        motor_friction_param.extend(motor_friction_temp[4:])
        
        # This is needed because not all joints will be active, and only the active one will have motor characteristics 
        for idx_joint in range(len(joint_active)):
            if(joint_active[idx_joint] == 1):
                joint_name_list_updated.append(joint_name_list[idx_joint])
                Im.append(motor_inertia_param[idx_joint])
                Kv.append(motor_friction_param[idx_joint])

        modifications = {}
        modifications_density = {}
        for idx, item in enumerate(link_names):
            if item in torso_link:
                modifications.update({item: links_length[idx]})
                modifications_density.update({item: density[idx]})
            else:    
                left_leg_item = "l_" + item
                right_leg_item = "r_" + item
                modifications.update({left_leg_item: links_length[idx]})
                modifications.update({right_leg_item: links_length[idx]})
                modifications_density.update({left_leg_item: density[idx]})
                modifications_density.update({right_leg_item: density[idx]})        

        motors_param = {}
        motors_param.update({"I_m": Im})
        motors_param.update({"k_v": Kv})

        create_urdf_instance.modify_lengths(modifications)
        create_urdf_instance.modify_densities(modifications_density)
        urdf_robot_string = create_urdf_instance.write_urdf_to_file()
        create_urdf_instance.reset_modifications()
        robot_model_init = RobotModel(urdf_robot_string, "stickBot", joint_name_list_updated)
        
        root_link = "root_link"
        left_foot_frame = "l_sole"
        head = "head"
        upper_arm_init = "l_shoulder_2"
        elbow = "l_elbow_1"
        fore_arm = "l_wrist_1"
        upper_leg = "l_hip_2"
        knee = "l_lower_leg"
        ankle = "l_ankle_2"
        torso_end = "neck_2"
        distances_to_compute = {
            "height": [head, left_foot_frame],
            "total arm": [upper_arm_init, fore_arm],
            "upper arm": [upper_arm_init, elbow],
            "lower arm": [elbow, fore_arm],
            "upper leg": [upper_leg, knee],
            "lower_leg": [knee, ankle],
            "torso": [upper_leg, torso_end],
        }

        joints = np.zeros(len(joint_name_list_updated))
        w_H_torso = robot_model_init.forward_kinematics_fun(root_link)
        w_H_leftFoot = robot_model_init.forward_kinematics_fun(left_foot_frame)

        w_H_torso_num = np.array(w_H_torso(np.eye(4), joints))
        w_H_lefFoot_num = np.array(w_H_leftFoot(np.eye(4), joints))
        w_H_init = np.linalg.inv(w_H_lefFoot_num) @ w_H_torso_num

        for key, value in distances_to_compute.items():
            w_h_frame_1 = robot_model_init.forward_kinematics_fun(value[0])
            w_h_frame_2 = robot_model_init.forward_kinematics_fun(value[1])
            w_h_frame_1_val = np.array(w_h_frame_1(w_H_init, joints))
            w_h_frame_2_val = np.array(w_h_frame_2(w_H_init, joints))
            frame_1_h_frame_2 = np.linalg.inv(w_h_frame_1_val) @ w_h_frame_2_val
            if key == "height":
                height = frame_1_h_frame_2[2, 3]
                heights.append(height)

        mass = robot_model_init.get_total_mass()
        masses.append(mass)
        com_fun = robot_model_init.CoM_position_fun()
        com_val = np.array(com_fun(w_H_init, joints))
        com_z_values.append(com_val[2])  # z-component of CoM

        avg_motor_friction.append(np.mean(Kv))
        avg_motor_inertia.append(np.mean(Im))
        active_joint_counts.append(len(joint_name_list_updated))

        # print("TOTAL MASS", mass)
        # print("CoM (z)", com_val[2]) 
        # print("Joint active", joint_name_list_updated)


    # Plotting Mass vs Height
    plt.figure(figsize=(10, 6))
    plt.scatter(masses, heights, color='blue', marker='o')
    plt.title('Mass vs Height')
    plt.xlabel('Mass (kg)')
    plt.ylabel('Height (m)')
    plt.grid(True)
    plt.savefig(SAVE_PATH+'mass_vs_height.png')  # Save the plot
    # plt.show()

    # Plotting Center of Mass vs Total Mass
    plt.figure(figsize=(10, 6))
    plt.scatter(masses, com_z_values, color='green', marker='o')
    plt.title('Center of Mass (Z) vs Total Mass')
    plt.xlabel('Mass (kg)')
    plt.ylabel('Center of Mass Z (m)')
    plt.grid(True)
    plt.savefig(SAVE_PATH+'com_vs_mass.png')  # Save the plot
    # plt.show()

    # Plotting Average Motor Friction vs Average Motor Inertia
    plt.figure(figsize=(10, 6))
    plt.scatter(avg_motor_friction, avg_motor_inertia, color='red', marker='o')
    plt.title('Average Motor Friction vs Average Motor Inertia')
    plt.xlabel('Average Motor Friction')
    plt.ylabel('Average Motor Inertia')
    plt.grid(True)
    plt.savefig(SAVE_PATH+'friction_vs_inertia.png')  # Save the plot
    # plt.show()

    # Plotting Height vs Total Number of Active Joints
    plt.figure(figsize=(10, 6))
    plt.scatter(heights, active_joint_counts, color='purple', marker='o')
    plt.title('Height vs Total Number of Active Joints')
    plt.xlabel('Height (m)')
    plt.ylabel('Total Number of Active Joints')
    plt.grid(True)
    plt.savefig(SAVE_PATH+'height_vs_active_joints.png')  # Save the plot
    file_name_i = SAVE_PATH+str(item_idx)+".png"
    visualize_model(urdf_robot_string,file_name_i)
    # plt.show()

def print_main_hardware_charact_old(list_chromosome):
    item_idx = 0 
    for item in list_chromosome:
        item_idx=item_idx+1
        # Define the robot modifications
        dict_return = chrom_generator.get_chromosome_dict(item)
        links_length = dict_return[NameChromosome.LENGTH]
        density = dict_return[NameChromosome.DENSITY]
        joint_active_temp = dict_return[NameChromosome.JOINT_TYPE]
        motor_friction_temp = dict_return[NameChromosome.MOTOR_FRICTION]
        motor_inertia_temp = dict_return[NameChromosome.MOTOR_INERTIA]
        joint_name_list_updated =[]
        Im = []
        Kv = []
        joint_active = []
        joint_arms = []
        joint_arms.extend(joint_active_temp[:3])
        joint_arms.extend([1]) # The elbow is always active 
        joint_active.extend(joint_arms)
        joint_active.extend(joint_arms)
        joint_active.extend(joint_active_temp[3:])
        joint_active.extend(joint_active_temp[3:])
        motor_inertia_param = []
        motor_inertia_param.extend(motor_inertia_temp[:4])
        motor_inertia_param.extend(motor_inertia_temp[:4])
        motor_inertia_param.extend(motor_inertia_temp[4:])
        motor_inertia_param.extend(motor_inertia_temp[4:])
        motor_friction_param = []
        motor_friction_param.extend(motor_friction_temp[:4])
        motor_friction_param.extend(motor_friction_temp[:4])
        motor_friction_param.extend(motor_friction_temp[4:])
        motor_friction_param.extend(motor_friction_temp[4:])
        
        # This is needed because not all joints will be active, and only the active one will have motor characteristics 
        for idx_joint in range(len(joint_active)):
            if(joint_active[idx_joint]== 1):
                joint_name_list_updated.append(joint_name_list[idx_joint])
                Im.append(motor_inertia_param[idx_joint])
                Kv.append(motor_friction_param[idx_joint])


        modifications = {}
        modifications_density = {}
        for idx,item in enumerate(link_names):
            if(item in torso_link):
                modifications.update({item: links_length[idx]})
                modifications_density.update({item:density[idx]})
            else:    
                left_leg_item = "l_" + item
                right_leg_item = "r_" + item
                
                modifications.update({left_leg_item: links_length[idx]})
                modifications.update({right_leg_item: links_length[idx]})
                modifications_density.update({left_leg_item: density[idx]})
                modifications_density.update({right_leg_item: density[idx]})        

        motors_param = {}
        motors_param.update({"I_m": Im})
        motors_param.update({"k_v":Kv})

        create_urdf_instance.modify_lengths(modifications)
        create_urdf_instance.modify_densities(modifications_density)
        urdf_robot_string = create_urdf_instance.write_urdf_to_file()
        create_urdf_instance.reset_modifications()
        robot_model_init = RobotModel(urdf_robot_string, "stickBot", joint_name_list_updated)
        root_link = "root_link"
        left_foot_frame = "l_sole"
        head = "head"
        upper_arm_init = "l_shoulder_2"
        elbow = "l_elbow_1"
        fore_arm = "l_wrist_1"
        upper_leg = "l_hip_2"
        knee = "l_lower_leg"
        ankle = "l_ankle_2"
        torso_end = "neck_2"
        distances_to_compute = {
            "heigth": [head, left_foot_frame],
            "total arm": [upper_arm_init, fore_arm],
            "upper arm": [upper_arm_init, elbow],
            "lower arm": [elbow, fore_arm],
            "upper leg": [upper_leg, knee],
            "lower_leg": [knee, ankle],
            "torso": [upper_leg, torso_end],
        }

        joints = np.zeros(len(joint_name_list_updated))
        w_H_torso = robot_model_init.forward_kinematics_fun(root_link)
        w_H_leftFoot = robot_model_init.forward_kinematics_fun(left_foot_frame)

        w_H_torso_num = np.array(w_H_torso(np.eye(4), joints))
        w_H_lefFoot_num = np.array(w_H_leftFoot(np.eye(4), joints))
        w_H_init = np.linalg.inv(w_H_lefFoot_num) @ w_H_torso_num

        for key, value in distances_to_compute.items():
            print(key)
            w_h_frame_1 = robot_model_init.forward_kinematics_fun(value[0])
            w_h_frame_2 = robot_model_init.forward_kinematics_fun(value[1])
            w_h_frame_1_val = np.array(w_h_frame_1(w_H_init, joints))
            w_h_frame_2_val = np.array(w_h_frame_2(w_H_init, joints))

            frame_1_h_frame_2 = np.linalg.inv(w_h_frame_1_val) @ w_h_frame_2_val
            print(frame_1_h_frame_2[:3, 3])
        mass = robot_model_init.get_total_mass()
        com_fun = robot_model_init.CoM_position_fun()
        com_val = np.array(com_fun(w_H_init, joints))
        print("TOTAL MASS", mass)
        print("com", com_val[2]) 
        print("joint active", joint_name_list_updated)
        file_name_i = SAVE_PATH+str(item_idx)+".png"
        visualize_model(urdf_robot_string,file_name_i)

def visualize_model(urdf_string_old, file_name):

    ## Modify the urdf string to be all grey 
    # Parse the URDF string
    root = ET.fromstring(urdf_string_old)

    # Define the new RGBA color
    new_rgba = "0.5 0.5 0.5 1."

    # Find all material elements and update their color
    for material in root.findall('material'):
        color = material.find('color')
        if color is not None:
            color.set('rgba', new_rgba)

    # Convert the modified XML tree back to a string
    urdf_string = ET.tostring(root, encoding='unicode')

    viz = iDynTree.Visualizer()
    vizOpt = iDynTree.VisualizerOptions()
    vizOpt.winWidth = 1500
    vizOpt.winHeight = 1500
    if not viz.init(vizOpt):
        raise Exception("Could not initialize iDynTree Visualizer")

    env = viz.enviroment()
    env.setElementVisibility("floor_grid",  False)
    env.setElementVisibility("world_frame", False)
    viz.setColorPalette("meshcat")
    env.setElementVisibility("world_frame", False)
    frames = viz.frames()
    cam = viz.camera()
    cam.setPosition(iDynTree.Position(3, 0, 1.2))
    viz.camera().animator().enableMouseControl(True)

    mdlLoader = iDynTree.ModelLoader()
    mdlLoader.loadModelFromString(urdf_string)
    viz.addModel(mdlLoader.model(), "model")
    time_out_viz = 4
    time_now = time.time()

    time_now = time.time()
    while(time.time()-time_now<time_out_viz and viz.run()): 
        viz.draw()
    viz.drawToFile(file_name)

if __name__ == "__main__":
    final_population, all_generations = main()
    all_gen = pickle.load( open("all_generations.pkl" , "rb" ) )
    final_population = all_gen[-1]
    analyse_output = False

    if(analyse_output):
        n_gen_tot = 271

        # Create a subset of the 'Blues' colormap
        cmap = cm.Blues 
        cmap_subset = colors.LinearSegmentedColormap.from_list(
            'Blues_subset', cmap(np.linspace(0.3, 1, 256))
        )

        # Normalize the color map to the number of generations
        norm = colors.Normalize(vmin=0, vmax=n_gen_tot)

        # Initialize the plot
        fig, ax = plt.subplots()

        for i in range(n_gen_tot):

            # Plot and evaluate single instance
            path_pop = "result/generation" + str(i) + ".p"
            final_population = pickle.load(open(path_pop, "rb"))
            # Extract the Pareto front
            pareto_front = tools.ParetoFront()
            pareto_front.update(final_population)

            # Print the solutions in the Pareto front
            # for ind in pareto_front:
            #     print(f"Individual: {ind}, Fitness: {ind.fitness.values}")
                # Plot the Pareto front
            fitnesses = [ind.fitness.values for ind in pareto_front.items]
            fitness1 = [f[0] for f in fitnesses]
            fitness2 = [f[1] for f in fitnesses]

            color = cmap_subset(norm(i))  # Normalize index to [0, 1] range
            ax.scatter(fitness1, fitness2, color=color, linewidths=5)       
            # if(i == n_gen_tot - 1):
                # ax.plot(fitness1, fitness2, color=color)

        # Add the color bar
        sm = plt.cm.ScalarMappable(cmap=cmap_subset, norm=norm)
        sm.set_array([])  # Provide a dummy array
        cbar = plt.colorbar(sm, ax=ax)  # Associate the colorbar with the correct axes
        cbar.set_label('Generation', fontsize=20)

        save_path_i = "Pareto_tot.png"
        ax.set_xlabel('Payload lifting', fontsize=40)
        ax.set_ylabel('Walking', fontsize=40)
        ax.set_title('Pareto Front', fontsize=60)
        # plt.yscale("log")
        # plt.xscale("log")
        ax.grid(True)
        fig.set_size_inches((21, 16), forward=False)
        plt.savefig(SAVE_PATH + save_path_i)
        plt.close()

        # Plot and evaluate single instance
        path_pop = "result/generation" + str(n_gen_tot-1)+".p"
        final_population = pickle.load(open(path_pop, "rb"))
        # Extract the Pareto front
        pareto_front = tools.ParetoFront()
        pareto_front.update(final_population)

        # Print the solutions in the Pareto front
        for ind in pareto_front:
            print(f"Individual: {ind}, Fitness: {ind.fitness.values}")
            # evaluate(ind)
            # Plot the Pareto front
        fitnesses = [ind.fitness.values for ind in pareto_front.items]
        fitness1 = [f[0] for f in fitnesses]
        fitness2 = [f[1] for f in fitnesses]

        color = cmap_subset(norm(n_gen_tot))  # Normalize index to [0, 1] range
        plt.figure()
        plt.scatter(fitness1, fitness2, color=color, linewidths=5)        
        plt.plot(fitness1, fitness2, color=color)
        # # plt.xscale("log")
        # plt.yscale("log")
        save_path_i = "Pareto"+str(2)+".png"
        plt.xlabel('Payload lifting', fontsize="40")
        plt.ylabel('Walking', fontsize="40")
        plt.title('Pareto Front', fontsize="60")
        plt.grid(True)
        fig = plt.gcf()
        fig.set_size_inches((21, 16), forward=False)
        plt.savefig(SAVE_PATH + save_path_i)
        plt.close()

        # Create a subset of the 'Blues' colormap
        cmap = cm.Blues 
        cmap_subset = colors.LinearSegmentedColormap.from_list(
            'Blues_subset', cmap(np.linspace(0.2, 1, 256))
        )

        # Load the population of the last generation
        path_pop = "result/generation" + str(n_gen_tot - 1) + ".p"  # Adjust index for 0-based index
        final_population = pickle.load(open(path_pop, "rb"))

        # Sort the population into Pareto fronts
        pareto_fronts = tools.sortNondominated(final_population, len(final_population), first_front_only=False)

        # Initialize the plot
        plt.figure()

        # Loop through all Pareto fronts and plot them
        for rank, front in enumerate(pareto_fronts):
            # Get the fitness values of the individuals in the current Pareto front
            fitnesses = [ind.fitness.values for ind in front]
            fitness1 = [f[0] for f in fitnesses]
            fitness2 = [f[1] for f in fitnesses]

            # Determine the color for this Pareto front
            color = cmap_subset(norm(rank))  # Normalize rank to [0, 1] range

            # Plot the Pareto front
            plt.scatter(fitness1, fitness2, color=color, label=f'Front {rank + 1}', linewidths=5)
            plt.plot(fitness1, fitness2, color=color)

        # Final plot settings
        save_path_i = "Pareto_all_fronts_last_generation.png"
        plt.xlabel('Payload lifting', fontsize=40)
        plt.ylabel('Walking', fontsize=40)
        plt.title('Pareto Fronts of Last Generation', fontsize=60)
        plt.grid(True)
        plt.legend(title="Pareto Fronts")

        # Set figure size and save the plot
        fig = plt.gcf()
        fig.set_size_inches((21, 16), forward=False)
        plt.savefig(SAVE_PATH + save_path_i)
        plt.close()

        print_main_hardware_charact(pareto_front)

