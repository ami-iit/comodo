from comodo.robotModel.robotModel import RobotModel
import manifpy as manif
import matplotlib.pyplot as plt 
import numpy as np
from matplotlib.patches import Rectangle
import bipedal_locomotion_framework.bindings as blf
from datetime import timedelta

class FootPositionPlanner():
   
    def __init__(self, robot_model:RobotModel, dT, step_length):
        self.robot_model = robot_model
        self.define_kindyn_functions()
        self.step_length = step_length
        self.dT = dT

    def set_scaling_parameters(self, scaling, scalingPos, scalingPosY): 
        
        self.scaling = scaling
        self.scaling_pos = scalingPos
        self.scaling_pos_y = scalingPosY
    
    def define_kindyn_functions(self): 
        
        self.H_left_foot_fun = self.robot_model.forward_kinematics_fun(self.robot_model.left_foot_frame)
        self.H_right_foot_fun = self.robot_model.forward_kinematics_fun(self.robot_model.right_foot_frame)

    def update_initial_position(self, Hb,s): 
        self.define_kindyn_functions()
        self.H_left_foot_init = self.H_left_foot_fun(Hb,s)
        self.H_right_foot_init = self.H_right_foot_fun(Hb,s)

    def compute_feet_contact_position(self): 
        quaternion = [0.0, 0.0, 0.0, 1.0]
        # quaternion[3] = 1 
        ### Left Foot 
        self.contact_list_left_foot = blf.contacts.ContactList()
        # self.contact_list_left_foot.setDefaultName("LeftFoot")
        contact = blf.contacts.PlannedContact()
        leftPosition = np.zeros(3)
        leftPosition_casadi = np.array(self.H_left_foot_init[:3,3])
        leftPosition[0] = float(leftPosition_casadi[0])
        leftPosition[1] = float(leftPosition_casadi[1])
        leftPosition[2] = float(leftPosition_casadi[2])
        leftPosition[2] = 0.0
        # Note we are using the default name and type i.e. full contact
        contact.pose = manif.SE3(position=leftPosition, quaternion=quaternion)
        contact.activation_time = timedelta(seconds=0.0)
        contact.deactivation_time = timedelta(seconds=1.0*self.scaling)
        contact.name = "contactLeft1"
        self.contact_list_left_foot.add_contact(contact)

        leftPosition[0] += float(self.step_length * self.scaling_pos)
        # print(leftPosition.shape)
        contact.pose = manif.SE3(position = leftPosition, quaternion = quaternion)
        contact.activation_time = timedelta(seconds=2.0*self.scaling)
        contact.deactivation_time = timedelta(seconds=5.0*self.scaling)
        contact.name = "contactLeft2"
        self.contact_list_left_foot.add_contact(contact)

        leftPosition[0]+= self.step_length * self.scaling_pos
        leftPosition[2] = 0.0
        contact.pose = manif.SE3(position = leftPosition, quaternion = quaternion)
        contact.activation_time = timedelta(seconds=6.0*self.scaling)
        contact.deactivation_time = timedelta(seconds=9.0*self.scaling)
        contact.name = "contactLeft3"
        self.contact_list_left_foot.add_contact(contact)
        
        leftPosition[0]+= self.step_length * self.scaling_pos
        leftPosition[2] = 0.0
        contact.pose = manif.SE3(position = leftPosition, quaternion = quaternion)
        contact.activation_time = timedelta(seconds=10.0*self.scaling)
        contact.deactivation_time = timedelta(seconds=13.0*self.scaling)
        contact.name = "contactLeft4"
        self.contact_list_left_foot.add_contact(contact)
     
        leftPosition[0]+= self.step_length * self.scaling_pos
        leftPosition[2] = 0.0
        contact.pose = manif.SE3(position = leftPosition, quaternion = quaternion)
        contact.activation_time = timedelta(seconds=14.0*self.scaling)
        contact.deactivation_time = timedelta(seconds=25.0*self.scaling)
        contact.name = "contactLeft5"
        self.contact_list_left_foot.add_contact(contact)
        

        # leftPosition[1]-= 0.01 * self.scaling_pos_y # lateral step TODO check if keeping it as initial test
        # leftPosition[2] = 0.0
        # contact.pose = manif.SE3(position = leftPosition, quaternion = quaternion)
        # contact.activation_time = timedelta(seconds=18.0*self.scaling)
        # contact.deactivation_time = timedelta(seconds=21.0*self.scaling)
        # contact.name = "contactLeft6"
        # self.contact_list_left_foot.add_contact(contact)
        

        # leftPosition[1]-= 0.01 * self.scaling_pos_y # lateral step TODO check if keeping it as initial test
        # leftPosition[2] = 0.0
        # contact.pose = manif.SE3(position = leftPosition, quaternion = quaternion)
        # contact.activation_time = timedelta(seconds=22.0*self.scaling)
        # contact.deactivation_time = timedelta(seconds=25.0*self.scaling)
        # contact.name = "contactLeft7"
        # self.contact_list_left_foot.add_contact(contact)
        
        # leftPosition[1]-= 0.01 * self.scaling_pos_y # lateral step TODO check if keeping it as initial test
        # leftPosition[2] = 0.0
        # contact.pose = manif.SE3(position = leftPosition, quaternion = quaternion)
        # contact.activation_time = timedelta(seconds=26.0*self.scaling)
        # contact.deactivation_time = timedelta(seconds=29.0*self.scaling)
        # contact.name = "contactLeft8"
        # self.contact_list_left_foot.add_contact(contact)
        
        ### Right Foot 
        self.contact_list_right_foot = blf.contacts.ContactList()
        # self.contact_list_left_foot.setDefaultName("RightFoot")
        contact = blf.contacts.PlannedContact()
        rightPosition = np.zeros(3)
        rightPosition_casadi = np.array(self.H_right_foot_init[:3,3])
        rightPosition[0] = float(rightPosition_casadi[0])
        rightPosition[1] = float(rightPosition_casadi[1])
        rightPosition[2] = float(rightPosition_casadi[2])
        rightPosition[2] = 0.0
        # Note we are using the default name and type i.e. full contact
        contact.pose = manif.SE3(position = rightPosition, quaternion = quaternion)
        contact.activation_time = timedelta(seconds=0.0)
        contact.deactivation_time = timedelta(seconds=3.0*self.scaling)
        contact.name = "contactRight1"
        self.contact_list_right_foot.add_contact(contact)

        rightPosition[0] += self.step_length*self.scaling_pos
        contact.pose = manif.SE3(position = rightPosition, quaternion = quaternion)
        contact.activation_time = timedelta(seconds=4.0*self.scaling)
        contact.deactivation_time = timedelta(seconds=7.0*self.scaling)
        contact.name = "contactRight2"
        self.contact_list_right_foot.add_contact(contact)
        
        rightPosition[0] += self.step_length*self.scaling_pos
        contact.pose = manif.SE3(position = rightPosition, quaternion =quaternion)
        contact.activation_time = timedelta(seconds=8.0*self.scaling)
        contact.deactivation_time = timedelta(seconds=11.0*self.scaling)
        contact.name = "contactRight3"
        self.contact_list_right_foot.add_contact(contact)

        rightPosition[0] += self.step_length*self.scaling_pos
        contact.pose = manif.SE3(position = rightPosition, quaternion = quaternion)
        contact.activation_time = timedelta(seconds=12.0*self.scaling)
        contact.deactivation_time = timedelta(seconds=15.0*self.scaling)
        contact.name = "contactRight4"
        self.contact_list_right_foot.add_contact(contact)

        rightPosition[0] += self.step_length*self.scaling_pos
        contact.pose = manif.SE3(position = rightPosition, quaternion =quaternion)
        contact.activation_time = timedelta(seconds=16.0*self.scaling)
        contact.deactivation_time = timedelta(seconds=25.0*self.scaling)
        contact.name = "contactRight5"
        self.contact_list_right_foot.add_contact(contact)

        # rightPosition[1] -= 0.01*self.scaling_pos_y
        # contact.pose = manif.SE3(position = rightPosition, quaternion = quaternion)
        # contact.activation_time = timedelta(seconds=20.0*self.scaling)
        # contact.deactivation_time = timedelta(seconds=23.0*self.scaling)
        # contact.name = "contactRight6"
        # self.contact_list_right_foot.add_contact(contact)
        
        # rightPosition[1] -= 0.01*self.scaling_pos_y
        # contact.pose = manif.SE3(position = rightPosition, quaternion = quaternion)
        # contact.activation_time = timedelta(seconds=24.0*self.scaling)
        # contact.deactivation_time = timedelta(seconds=27.0*self.scaling)
        # contact.name = "contactRight7"
        # self.contact_list_right_foot.add_contact(contact)


        # rightPosition[1] -= 0.01*self.scaling_pos_y
        # contact.pose = manif.SE3(position = rightPosition, quaternion = quaternion)
        # contact.activation_time = timedelta(seconds=28.0*self.scaling)
        # contact.deactivation_time = timedelta(seconds=29.0*self.scaling)
        # contact.name = "contactRight8"
        # self.contact_list_right_foot.add_contact(contact)

    def get_contact_phase_list(self): 
        contact_list_map ={}
        contact_list_map.update({"left_foot":self.contact_list_left_foot})
        contact_list_map.update({"right_foot":self.contact_list_right_foot})
        contact_phase_list = blf.contacts.ContactPhaseList()
        contact_phase_list.set_lists(contact_list_map)
        return contact_phase_list

    def define_feet_position_numpy(self): 

        ### Left Foot 
        self.contact_list_left_foot = []
        leftPosition = np.array(self.H_left_foot_init[:3,3])
        contact = np.array(3)
        
        contact = leftPosition
        self.contact_list_left_foot.append(contact)

        leftPosition[0]+= 0.05 * self.scaling_pos
        contact = leftPosition
        self.contact_list_left_foot.append(contact)

        leftPosition[0]+= 0.1 * self.scaling_pos
        leftPosition[2] = 0.0
        contact = leftPosition
        self.contact_list_left_foot.append(contact)
        
        leftPosition[0]+= 0.1 * self.scaling_pos
        leftPosition[2] = 0.0
        contact = leftPosition
        self.contact_list_left_foot.append(contact)
     
        leftPosition[0]+= 0.1 * self.scaling_pos
        leftPosition[2] = 0.0
        contact = leftPosition
        self.contact_list_left_foot.append(contact)
        

        leftPosition[1]-= 0.1 * self.scaling_pos_y # lateral step TODO check if keeping it as initial test
        leftPosition[2] = 0.0
        contact = leftPosition
        self.contact_list_left_foot.append(contact)
        

        leftPosition[1]-= 0.1 * self.scaling_pos_y # lateral step TODO check if keeping it as initial test
        leftPosition[2] = 0.0
        contact = leftPosition
        self.contact_list_left_foot.append(contact)
        
        leftPosition[1]-= 0.1 * self.scaling_pos_y # lateral step TODO check if keeping it as initial test
        leftPosition[2] = 0.0
        contact = leftPosition
        self.contact_list_left_foot.append(contact)

        ### Right Foot 
        self.contact_list_right_foot = []
        rightPosition = np.array(self.H_right_foot_init[:3,3])
        
        contact = np.array(3)
        # Note we are using the default name and type i.e. full contact
        contact = rightPosition
        self.contact_list_right_foot.append(contact)

        rightPosition[0] += 0.1*self.scaling_pos
        contact = rightPosition
        self.contact_list_right_foot.append(contact)
        
        rightPosition[0] += 0.1*self.scaling_pos
        contact = rightPosition
        self.contact_list_right_foot.append(contact)

        rightPosition[0] += 0.1*self.scaling_pos
        contact = rightPosition
        self.contact_list_right_foot.append(contact)

        rightPosition[0] += 0.05*self.scaling_pos
        rightPosition[1] -= 0.01*self.scaling_pos_y
        contact = rightPosition
        self.contact_list_right_foot.append(contact)

        rightPosition[1] -= 0.01*self.scaling_pos_y
        contact = rightPosition
        self.contact_list_right_foot.append(contact)
        
        rightPosition[1] -= 0.01*self.scaling_pos_y
        contact = rightPosition
        self.contact_list_right_foot.append(contact)


        rightPosition[1] -= 0.01*self.scaling_pos_y
        contact = rightPosition
        self.contact_list_right_foot.append(contact)

    def plot_feet_position(self):         
        fig =plt.figure()
        ax = fig.add_subplot(111)
        for item in self.contact_list_left_foot:
            pos = item.pose.translation()
            # pos = item 
            plt.plot(pos[0], pos[1], marker='D', color = 'red')
            ax.add_patch(Rectangle((pos[0]-0.025,pos[1]-0.01), 0.05, 0.02, color = 'red', alpha = 0.2))
            ax.text(pos[0], pos[1], str(item.activation_time), style='italic')
        for item in self.contact_list_right_foot: 
            pos = item.pose.translation()
            # pos = item 
            plt.plot(pos[0], pos[1], marker ='D', color= 'blue')
            ax.add_patch(Rectangle((pos[0]-0.025,pos[1]-0.01), 0.05, 0.02,color = 'blue', alpha = 0.2))
            ax.text(pos[0], pos[1], str(item.activation_time), style='italic')
        plt.title("Feet Position", fontsize="60")
        plt.xlabel("x [m]", fontsize="40")
        plt.ylabel("y [m]", fontsize="40")
        plt.show()

    def initialize_foot_swing_planner(self): 

        self.parameters_handler = blf.parameters_handler.StdParametersHandler()
        self.parameters_handler.set_parameter_datetime("sampling_time", self.dT)
        self.parameters_handler.set_parameter_float("step_height", 0.1)
        self.parameters_handler.set_parameter_float("foot_apex_time", 0.5)
        self.parameters_handler.set_parameter_float("foot_landing_velocity", 0.0)
        self.parameters_handler.set_parameter_float("foot_landing_acceleration", 0.0)
        self.parameters_handler.set_parameter_float("foot_take_off_velocity", 0.0)
        self.parameters_handler.set_parameter_float("foot_take_off_acceleration", 0.0)

        self.planner_left_foot = blf.planners.SwingFootPlanner()
        self.planner_right_foot = blf.planners.SwingFootPlanner()
        self.planner_left_foot.initialize(handler=self.parameters_handler)
        self.planner_right_foot.initialize(handler=self.parameters_handler)
        self.planner_left_foot.set_contact_list(contact_list=self.contact_list_left_foot)
        self.planner_right_foot.set_contact_list(contact_list=self.contact_list_right_foot)
        # print("planner right foot",self.planner_right_foot.__dir__())
        # self.plot_feet_position()
   
    def advance_swing_foot_planner(self): 
        self.planner_left_foot.advance()
        self.planner_right_foot.advance()
        
    def get_references_swing_foot_planner(self):
        left_foot = self.planner_left_foot.get_output()
        right_foot = self.planner_right_foot.get_output()
        return left_foot, right_foot