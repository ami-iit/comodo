import subprocess
import yarp
import time
from include.robotModel.robotModel import RobotModel
from lxml import etree
import numpy as np


class GazeboLauncher:
    def __init__(self, robotModel: RobotModel):
        self.process = None
        self.robot_model = robotModel
        self.offset_z = 0.15
        self.CLOCK_RPC_PORT_NAME = "/clock/rpc"
        self.CLIENT_PORT_NAME = "/simulator_syncronizer"

        ## Declaring the main commands to be used with gazebo clock rpc
        self.GET_GAZEBO_FREQUENCE = "getStepSize"
        self.RESET_SIMULATION_TIME = "resetSimulationTime"
        self.STEP_SIMULATION_AND_WAIT = "stepSimulationAndWait"
        self.GET_SIMULATION_TIME = "getSimulationTime"
        self.PAUSE_SIMULATION = "pauseSimulation"
        self.RESET_SIMULATION = "resetSimulation"
        self.launch_and_check_yarp_server()
        self.initialized_control_board = False

    def set_control_board_names(head_control_board:str, torso_control_board:str, left_arm_control_board:str, right_arm_control_board:str, left_leg_control_board:str, right_leg_control_board:str): 
        self.head_control_board = head_control_board
        self.torso_control_board = torso_control_board
        self.left_arm_control_board =left_arm_control_board
        self.right_arm_control_board = right_arm_control_board
        self.left_leg_control_board = left_leg_control_board
        self.right_leg_control_board = right_leg_control_board
        self.initialized_control_board = True 

    def write_clock_rpc_command(self, command):
        response = yarp.Bottle()
        response.clear()
        cmd = yarp.Bottle()
        cmd.clear()
        cmd.addString(str(command))
        self.client_rpc.write(cmd, response)
        return response.toString()

    def get_simulation_time(self):
        response = self.write_clock_rpc_command(self.GET_SIMULATION_TIME)
        return float(response)

    def launch_and_check_yarp_server(self):
        # Start the YARP server
        self.yarp_server_process = subprocess.Popen(["yarp", "server"])
        # Wait for the server to start
        print("Waiting for the server to start...")
        yarp.Network.init()
        check_yarp_port_name = "/check_yarp_server"
        # Create a port and set it as a server
        port = yarp.Port()
        port.open(check_yarp_port_name)

        while not (yarp.Network.exists(check_yarp_port_name)):
            yarp.Network.init()
            check_yarp_port_name = "/check_yarp_server"
            # Create a port and set it as a server
            port = yarp.Port()
            port.open(check_yarp_port_name)
            print("yarp server still not up ")
        print("yarp server up and running")
        port.close()

    def launch(self):
        input_port = yarp.Port()
        CHECK_PORT = "/check_gazebo"
        CLOCK_PORT = "/icubSim/head/state:o"
        input_port.open(CHECK_PORT)
        subprocess.Popen(["pkill", "gzserver"])
        subprocess.Popen(["pkill", "gzclient"])
        self.process = subprocess.Popen(
            ["gazebo","-slibgazebo_yarp_clock.so", self.robot_model.world_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        while not (yarp.Network.connect(CLOCK_PORT, CHECK_PORT)):
            time.sleep(0.05)
        input_port.close()
        self.client_rpc = yarp.RpcClient()
        # connect to the RPC port
        self.client_rpc.open(self.CLIENT_PORT_NAME)
        self.client_rpc.addOutput(self.CLOCK_RPC_PORT_NAME)
        # read the response from the RPC port
        self.SIMULATION_FREQUENCY = float(
            self.write_clock_rpc_command(self.GET_GAZEBO_FREQUENCE)
        )
        self.pause_simulation_and_reset_time()

    def pause_simulation_and_reset_time(self):
        # resetting simulation time
        self.write_clock_rpc_command(self.RESET_SIMULATION_TIME)
        self.write_clock_rpc_command(self.PAUSE_SIMULATION)

    def syncronize_simulator_control(self, controller_time_step):
        number_of_steps = int(controller_time_step / self.SIMULATION_FREQUENCY)
        for _ in range(number_of_steps):
            self.write_clock_rpc_command(self.STEP_SIMULATION_AND_WAIT)

    def perform_one_sim_step(self): 
        self.write_clock_rpc_command(self.STEP_SIMULATION_AND_WAIT)
        
    def get_initial_configuration_world(self, control_board_name: str, world_path=None):
        if world_path is None:
            world_path = self.robot_model.world_path
        tree = etree.parse(world_path)
        root = tree.getroot()
        world = root.find("world")
        model = world.find("model")
        for node in model.findall("plugin"):
            if node.get("name") == control_board_name:
                inConf = node.find("initialConfiguration")
                return np.fromstring(inConf.text, dtype=np.float32, sep=" ")

    def set_initial_configuration_world(
        self, root, control_board_name: str, new_value: np.array
    ):
        world = root.find("world")
        model = world.find("model")
        for node in model.findall("plugin"):
            if node.get("name") == control_board_name:
                inConf = node.find("initialConfiguration")
                new_value_str = str(new_value)
                new_value_str = new_value_str.replace("[", "")
                new_value_str = new_value_str.replace("]", "")
                inConf.text = new_value_str
        return root

    def set_pose_world(self, root, value: np.array):
        world = root.find("world")
        model = world.find("model")
        include = model.find("include")
        pose = include.find("pose")
        value[2] = value[2] + self.offset_z
        new_value_str = str(value)
        new_value_str = new_value_str.replace("[", "")
        new_value_str = new_value_str.replace("]", "")
        pose.text = new_value_str
        return root

    def write_world(self, root, world_path=None):
        if world_path is None:
            world_path = self.robot_model.world_path
        with open(world_path, "bw") as f:
            f.write(etree.tostring(root, pretty_print=True))

    def modify_world(self, s, xyz_rpy, world_path=None):
        if(not(self.initialized_control_board)): 
            raise Exception("[comodo::GazeboLauncher] You have not initialized the control boards name, I cannot modify the world")
       
        if world_path is None:
            world_path = self.robot_model.world_path

        # TODO for now we assume that all the entry in the world are in s 
        # self.constant_left_arm = self.get_initial_configuration_world(
        #     self.left_arm_control_board
        # )[4:]
        # self.constant_right_arm = self.get_initial_configuration_world(
        #     self.right_arm_control_board
        # )[4:]

        left_arm = self.robot_model.get_left_arm_from_joint_position(s)
        right_arm = self.robot_model.get_right_arm_from_joint_position(s)
        left_leg = self.robot_model.get_left_leg_from_joint_position(s)
        right_leg = self.robot_model.get_right_leg_from_joint_position(s)
        torso = self.robot_model.get_torso_from_joint_position(s)

        # left_arm = np.concatenate((left_arm_temp, self.constant_left_arm), axis=0)
        # right_arm = np.concatenate((right_arm_temp, self.constant_right_arm), axis=0)

        tree = etree.parse(world_path)
        root = tree.getroot()
        if(left_arm is not None):
            root = self.set_initial_configuration_world(
                root, self.left_arm_control_board, left_arm
            )
        if(right_arm is not None):
            root = self.set_initial_configuration_world(
                root, self.right_arm_control_board, right_arm
            )
        if(left_leg is not None):   
            root = self.set_initial_configuration_world(
                root, self.left_leg_control_board, left_leg
            )
        if(right_leg is not None):
            root = self.set_initial_configuration_world(
                root, self.right_leg_control_board, right_leg
            )
        if(torso is not None):
            root = self.set_initial_configuration_world(
            root, self.torso_control_board, torso
            )

        root = self.set_pose_world(root, xyz_rpy)
        return root

    def terminate(self):
        subprocess.Popen(["pkill", "gzserver"])
        subprocess.Popen(["pkill", "gzclient"])
        self.process.terminate()
        self.process.wait()

    def terminate_yarp_server(self):
        yarp.Network.fini()
        self.yarp_server_process.terminate()
        subprocess.Popen(["pkill", "yarp"])
        pass
