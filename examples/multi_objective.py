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
from matplotlib import cm  # Import color map

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
length_multiplier.limits = [0.5,2.0]
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
mpc_parameters.limits = np.array([[5,60],[20,100],[10,140],[10,140],[10,140],[10,140],[10,140],[10,140]])
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
    # print("achieve goal", 30*achieve_goal)
    # print("torque_mean", 0.5*torque_mean)
    # print("torque_diff", 100*torque_diff)
    # mujoco_instance.close_visualization()
    fitness_pl = weight_achieve_goal*achieve_goal + weight_torque_mean*torque_mean + weight_torque_diff*torque_diff
    if(fitness_pl>250):
        fitness_pl = 250
    return fitness_pl

def compute_fitness_walking(modifications_length, modifications_densities, motors_param,joint_name_list_updated,joint_active, mpc_chr, tsid_chr):
    # Modify the robot model and initialize
     # Set loop variables
    TIME_TH = 20
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
        # Reading robot state from simulator
        s, ds, tau = mujoco_instance.get_state()
        H_b = mujoco_instance.get_base()
        w_b = mujoco_instance.get_base_velocity()
        t = mujoco_instance.get_simulation_time()
        torque.append(np.linalg.norm(tau))
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
    ind1 = chrom_generator.check_chromosome_in_set(ind1)
    ind2 = chrom_generator.check_chromosome_in_set(ind2)
    return ind1, ind2

def mutate(ind):
    #TODO to be implemented in the chromosome generator 
    ind_new = chrom_generator.generate_chromosome()
    idx= random.randint(0, len(ind_new))
    # if(random.random()>0.5):
    ind[:idx] = ind_new[:idx]
    # else:
    #     ind[idx:] = ind_new[idx:]
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
            if random.random() < 0.9:
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

if __name__ == "__main__":
    final_population, all_generations = main()
    all_gen = pickle.load( open("all_generations.pkl" , "rb" ) )
    final_population = all_gen[-1]

    # path_pop = "result/generation1173.p"
    # final_population = pickle.load(open(path_pop, "rb"))
    # Extract the Pareto front
    # pareto_front = tools.ParetoFront()
    # pareto_front.update(final_population)

    # Print the solutions in the Pareto front
    # for ind in pareto_front:
    #     print(f"Individual: {ind}, Fitness: {ind.fitness.values}")
        # Plot the Pareto front
    # fitnesses = [ind.fitness.values for ind in pareto_front.items]
    # fitness1 = [f[0] for f in fitnesses]
    # fitness2 = [f[1] for f in fitnesses]
    # # print("size pareto front", len(pareto_front))
    # for item in pareto_front.items: 
    #     evaluate(item)

    ## PLOT ALL GENERATIONS 
    # cmap = cm.Blues  # Use the 'Blues' colormap
    # num_generations = 1150
    # for i in range(num_generations):
    #     plt.figure()
    #     path_pop = "result/generation"+ str(i)+".p"
    #     final_population = pickle.load(open(path_pop, "rb"))
    #     # Extract the Pareto front
    #     pareto_front = tools.ParetoFront()
    #     pareto_front.update(final_population)

    #     # Print the solutions in the Pareto front
    #     # for ind in pareto_front:
    #     #     print(f"Individual: {ind}, Fitness: {ind.fitness.values}")
    #         # Plot the Pareto front
    #     fitnesses = [ind.fitness.values for ind in pareto_front.items]
    #     fitness1 = [f[0] for f in fitnesses]
    #     fitness2 = [f[1] for f in fitnesses]
    #     # print("size pareto front", len(pareto_front))
    #     # for item in pareto_front.items: 
    #     #     evaluate(item)
    #     color = cmap(i / num_generations)  # Normalize index to [0, 1] range
    #     plt.scatter(fitness1, fitness2, c=color, linewidths=5)        
    #     # plt.plot(fitness1, fitness2, 'ro-')
    #     # plt.xscale("log")
    #     # plt.yscale("log")
    #     print(i)
    #     save_path_i = "Pareto"+str(i)+".png"
    #     plt.xlabel('Payload lifting', fontsize="40")
    #     plt.ylabel('Walking', fontsize="40")
    #     plt.title('Pareto Front', fontsize="60")
    #     plt.grid(True)
    #     fig = plt.gcf()
    #     print(i)
    #     fig.set_size_inches((21, 16), forward=False)
    #     plt.savefig(SAVE_PATH + save_path_i)
    #     plt.close()
    # # Step 1: Sort the population into Pareto fronts
    # pareto_fronts = tools.sortNondominated(final_population, len(final_population))
    # print("final populaton",len(final_population))
    # # Step 2: Plot the Pareto fronts
    # print('pareto front', len(pareto_fronts))
    # for i, front in enumerate(pareto_fronts):
    #     # Extract the objectives of individuals in this front
    #     front_objs = [ind.fitness.values for ind in front]
        
    #     # Unzip to separate the objectives
    #     front_x, front_y = zip(*front_objs)
    #     plt.figure()
    #     # Plot the front with different styles or colors
    #     plt.scatter(front_x, front_y, label=f"Front {i+1}", marker='o')

    # plt.title("Pareto Fronts")
    # plt.xlabel("Objective 1")
    # plt.ylabel("Objective 2")
    # plt.legend()
    # plt.grid(True)

    # fig = plt.gcf()
    # fig.set_size_inches((21, 16), forward=False)
    # plt.savefig(SAVE_PATH + "ParetoFronts.png")
    # plt.show()
    # plt.show()

