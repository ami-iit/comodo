import logging
import math

import numpy as np
from odio_urdf import xml_to_odio
from pydrake.common.eigen_geometry import Quaternion
from pydrake.geometry import MeshcatVisualizer, StartMeshcat
from pydrake.math import RigidTransform
from pydrake.multibody.meshcat import ContactVisualizer, ContactVisualizerParams
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.analysis import Simulator, SimulatorStatus
from pydrake.systems.framework import (
    Context,
    DiagramBuilder,
    InputPort,
    InputPortIndex,
    OutputPortIndex,
    PortDataType,
    System,
)

from comodo.abstractClasses.simulator import Simulator as SimulatorAbstract
from comodo.drakeSimulator.utils import DrakeURDFHelper

# the style of wrapping the simulator is inspired by the approach by DrakeGymEnv:
# https://github.com/RobotLocomotion/drake/blob/master/bindings/pydrake/gym/_drake_gym_env.py#L177


class DrakeSimulator(SimulatorAbstract):
    def __init__(self) -> None:
        self.duh = DrakeURDFHelper()
        self.active_meshcat = False
        super().__init__()

    def load_model(self, robot_model, s, xyz_rpy, kv_motors=None, Im=None):
        # load the robot model construct the diagram and store the simulator
        self.robot_model = robot_model
        self.urdf_string = robot_model.urdf_string

        # convert the urdf string to be drake compatible
        self.duh.load_urdf(urdf_string=self.urdf_string)
        self.duh.remove_all_collisions()
        self.duh.fix_not_in_joint_list(self.robot_model.joint_name_list)
        self.duh.convert_xml_to_odio()
        self.duh.add_acutation_tags()
        self.urdf_string = self.duh.get_urdf_string()

        builder = DiagramBuilder()
        self.time_step = 1e-3
        plant, scene_graph = AddMultibodyPlantSceneGraph(
            builder, time_step=self.time_step
        )
        parser = Parser(plant, scene_graph)
        # TODO: Handle urdfs with mesh packages
        # parser.package_map().Add("ergoCub", mesh_path)
        robot_model_sim = parser.AddModels(
            file_contents=self.urdf_string,
            file_type="urdf",
        )[0]

        # TODO: Use Im and Km to add motor params here
        # ignoring those arguments for now

        # add the ground and the feet
        self.duh.add_ground_with_friction(plant)

        # configure feet collisions
        xMinMax = [-0.1, 0.1]
        yMinMax = [-0.05, 0.05]
        self.duh.add_soft_feet_collisions(plant, xMinMax=xMinMax, yMinMax=yMinMax)

        plant.Finalize()
        builder.ExportInput(plant.get_actuation_input_port(), "control_input")
        builder.ExportOutput(plant.get_state_output_port(), "state_output")

        # useful simulator details
        self.nq = plant.num_positions()
        self.nv = plant.num_velocities()
        self.na = plant.num_actuators()
        self.sim_joint_order = self.duh.get_sim_joint_order(plant, robot_model_sim)
        # check if the joint ordering is the same
        # logging.info(
        #     "Need joint mapping: ", not (sim_joint_order == robot_model.joint_name_list)
        # )

        # create useful variables
        self.qpos = np.zeros(self.nq)
        self.joint_torques = np.zeros(self.na)

        if self.visualize_robot_flag:
            MeshcatVisualizer.AddToBuilder(builder, scene_graph, self.meshcat)
            ContactVisualizer.AddToBuilder(
                builder,
                plant,
                self.meshcat,
                ContactVisualizerParams(
                    newtons_per_meter=1e3, newton_meters_per_meter=1e1
                ),
            )

        diagram = builder.Build()
        self.simulator = Simulator(diagram)

        # now perform the setup
        self._setup(s, xyz_rpy)

    def _setup(self, s, xyz_rpy):
        # similar to DrakeGymEnv we need to expose the required ports
        system = self.simulator.get_system()
        self.state_output_port = system.GetOutputPort("state_output")
        self.control_input_port = system.GetInputPort("control_input")

        self.context = self.simulator.get_mutable_context()
        self.context.SetTime(0)
        if not self.simulator.Initialize():
            logging.error("Drake simulator instance failed to initialize")

        self.simulator.get_system().SetDefaultContext(self.context)

        self.diagram = self.simulator.get_system()
        plant = self.diagram.GetSubsystemByName("plant")
        plant_context = self.diagram.GetMutableSubsystemContext(plant, self.context)

        self.set_base_pose_in_drake(xyz_rpy)
        self.set_joint_vector_in_drake(s)
        # set initial pose of the robot
        plant.SetPositions(plant_context, self.qpos)
        if self.visualize_robot_flag:
            self.diagram.ForcedPublish(self.context)

    def reset(self):
        logging.error("Drake simulator reset not implemented, recreate drake_instance")
        pass

    def set_visualize_robot_flag(self, visualize_robot):
        # pass meshcat to visualise the robot
        self.visualize_robot_flag = visualize_robot
        if self.visualize_robot_flag and not self.active_meshcat:
            self.meshcat = StartMeshcat()
            self.active_meshcat = True
        pass

    # does this need to simulator specific?
    def set_base_pose_in_drake(self, xyz_rpy):
        base_xyz_quat = np.zeros(7)
        # order -- quaternion+xyz
        base_xyz_quat[:4] = self.RPY_to_quat(xyz_rpy[3], xyz_rpy[4], xyz_rpy[5])
        base_xyz_quat[4:] = xyz_rpy[:3]
        self.qpos[:7] = base_xyz_quat

    # set initial positions of the joints
    def set_joint_vector_in_drake(self, pos):
        self.qpos[7:] = pos
        pass

    def set_input(self, input):
        # expose the acutation output port of the system to accept control
        # inputs from the TSID and CentroidalMPC and step drake simulator
        self.control_input_port.FixValue(self.context, input)
        pass

    def step(self, n_step=1, visualize=True):
        # TODO: does nothing currently
        # self.control_input_port.FixValue(self.context, self.joint_torques)
        self.simulator.AdvanceTo(self.context.get_time() + n_step * self.time_step)
        if visualize and self.visualize_robot_flag:
            self.diagram.ForcedPublish(self.context)
        pass

    def step_with_motors(self, n_step, torque):
        # implementation of motor level control inputs?
        pass

    def get_base(self):
        # this is for the state output return
        # quat + pos
        base_pos = self.state_output_port.Eval(self.context)[:7]
        # normalise unnormalised quaternion
        base_quat_wxyz = base_pos[:4]
        H_b = RigidTransform(
            Quaternion(base_quat_wxyz / (np.linalg.norm(base_quat_wxyz))), base_pos[4:]
        ).GetAsMatrix4()
        return H_b

    def get_base_velocity(self):
        # this is for the state output return
        base_vel = self.state_output_port.Eval(self.context)[self.nq : self.nq + 6]
        # order -- linar_vel + angular_vel
        return np.concatenate((base_vel[3:], base_vel[:3]))

    # TODO: Change from get_state to get_joint_state?
    def get_state(self):
        # this is for the state output return
        robot_state = self.state_output_port.Eval(self.context)
        robot_pos = robot_state[: self.nq]
        robot_vel = robot_state[self.nq : self.nq + self.nv]
        # return joint state to be coherent with the MuJoCo API
        return (
            robot_pos[7:],
            robot_vel[6:],
            self.joint_torques,
        )

    def close(self):
        pass

    def visualize_robot(self):
        self.viewer.render()

    def get_simulation_time(self):
        # get time from the simulator context
        return self.context.get_time()

    def get_simulation_frequency(self):
        # return the simulator timestep
        return self.time_step

    # we will need similar stuff but with manifpy
    def RPY_to_quat(self, roll, pitch, yaw):
        cr = math.cos(roll / 2)
        cp = math.cos(pitch / 2)
        cy = math.cos(yaw / 2)
        sr = math.sin(roll / 2)
        sp = math.sin(pitch / 2)
        sy = math.sin(yaw / 2)

        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy

        # Note: The order for drake is different as compared to mujoco
        return [qw, qx, qy, qz]

    def close_visualization(self):
        self.meshcat.Delete()
        logging.info(
            "Drake uses meshcat for visualization. Close browser tab to close \
            the visualisation."
        )
        self.reset()
        pass
