from comodo.abstractClasses.simulator import Simulator
from comodo.gazeboClassicSimulator.robotInterface import robotInterface
from comodo.gazeboClassicSimulator.gazeboLauncher import GazeboLauncher 
import copy

class GazeboClassicSimulator(Simulator):

    def __init__(self, gazebo_launcher:GazeboLauncher) -> None:
        self.torque_control_mode = False
        self.gazebo_launcher = gazebo_launcher
        self.torque_control_mode = False
        super().__init__() 
    
    def load_model(self, robot_model,  s, xyz_rpy):
        self.robot_model = robot_model
        
        self.robot_interface = robotInterface(
            self.robot_model.robot_name,
            "/local",
            self.robot_model.joint_name_list,
            self.robot_model.remote_control_board_list,
        )
        
        root = self.gazebo_launcher.modify_world( s, xyz_rpy)
        self.gazebo_launcher.write_world(root)
        self.gazebo_launcher.launch()
        ## Opening the robot interface 
        self.robot_interface.open()
    
    def get_simulation_frequency(self): 
        return copy.copy(self.gazebo_launcher.SIMULATION_FREQUENCY)
    
    def get_simulation_time(self): 
        return self.gazebo_launcher.get_simulation_time()

    def set_position_control_mode(self): 
        self.robot_interface.set_position_control_mode()
        self.torque_control_mode = False

    def set_torque_control_mode(self): 
        self.robot_interface.set_torque_control_mode()
        self.torque_control_mode = True
    
    def set_input(self, input):

        if(self.torque_control_mode):
            self.robot_interface.set_joints_torque(input)
        else:
            RuntimeWarning("Torque Control Mode not set")
    
    def step(self, n_step=1):
        for _ in range(n_step): 
            self.gazebo_launcher.perform_one_sim_step()
    
    def get_state(self):
        s = self.robot_interface.get_joints_position()
        ds = self.robot_interface.get_joints_velocity()
        tau = self.robot_interface.get_joints_torque()
        return s,ds,tau

    def close(self):
        self.gazebo_launcher.terminate()