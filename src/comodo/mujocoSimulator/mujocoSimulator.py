from comodo.abstractClasses.simulator import Simulator
from comodo.robotModel.robotModel import RobotModel
from comodo.mujocoSimulator.callbacks import Callback
from comodo.mujocoSimulator.mjcontactinfo import MjContactInfo
from typing import Dict, List
import mujoco
import math
import numpy as np
import mujoco_viewer
import copy
import logging
import pathlib
import tempfile
import casadi as cs


class MujocoSimulator(Simulator):
    def __init__(self, logger: logging.Logger = None, log_dir: str = "") -> None:
        self.robot_model = None
        self.model = None
        self.desired_pos = None
        self.postion_control = False
        self.should_stop = False
        self.t = 0
        self.iter = 0
        self.contacts = []
        self._setup_logger(logger, log_dir)
        self.compute_misalignment_gravity_fun()
        super().__init__()

    def _setup_logger(self, logger: logging.Logger, log_dir) -> None:
        if logger is None:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            fmt = logging.Formatter(fmt=FMT)
            ch.setFormatter(fmt=fmt)
            self.logger.addHandler(ch)
        else:
            self.logger = logger

    def load_model(self, robot_model: RobotModel, s, xyz_rpy, kv_motors=None, Im=None, floor_opts: Dict = {}) -> None:
        """
        Loads the robot model into the MuJoCo simulator.
        
        Args:
            robot_model (RobotModel): The robot model to be loaded.
            s (array-like): The joint state vector.
            xyz_rpy (array-like): The base pose in terms of position (xyz) and orientation (rpy).
            kv_motors (array-like?): Motor velocity constants. Defaults to zeros if not provided.
            Im (array-like?): Motor inertia values. Defaults to zeros if not provided.
        Returns:
            None
        """

        self.robot_model = robot_model
        mujoco_xml = robot_model.get_mujoco_model(floor_opts=floor_opts, save_mjc_xml=False)
        try:
            self.model = mujoco.MjModel.from_xml_string(mujoco_xml)
        except Exception as e:
            with open("failed_mujoco.xml", "w") as f:
                f.write(mujoco_xml)
            self.logger.error(f"Failed to load the model: {e}. Dumped the model to failed_mujoco.xml")
            raise
        self.data = mujoco.MjData(self.model)
        self.create_mapping_vector_from_mujoco()
        self.create_mapping_vector_to_mujoco()
        mujoco.mj_forward(self.model, self.data)
        self.set_joint_vector_in_mujoco(s)
        self.set_base_pose_in_mujoco(xyz_rpy=xyz_rpy)
        mujoco.mj_forward(self.model, self.data)
        self.visualize_robot_flag = False
        self.should_stop = False

        self.Im = Im if not (Im is None) else np.zeros(self.robot_model.NDoF)
        self.kv_motors = (
            kv_motors if not (kv_motors is None) else np.zeros(self.robot_model.NDoF)
        )
        # self.H_left_foot = copy.deepcopy(self.robot_model.H_left_foot)
        # self.H_right_foot = copy.deepcopy(self.robot_model.H_right_foot)
        # self.H_left_foot_num = None
        # self.H_right_foot_num = None
        self.mass = self.robot_model.get_total_mass()

    def get_model_frame_pose2(self, frame_name: str) -> np.ndarray:
        frame_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, frame_name)
        return self.data.xpos[frame_id] 
    
    def get_model_frame_pose(self, frame_name: str) -> np.ndarray:
        base_pose = self.get_base()
        joint_values = self.get_state()[0]
        frame_pose = self.robot_model.forward_kinematics(frame_name, base_pose, joint_values)
        return frame_pose


    def get_contact_status(self) -> tuple:
        """
        Determines the contact status of the left and right feet.

        Returns:
            tuple: A tuple containing two boolean values:
                - left_foot_contact (bool): True if the left foot is in contact, False otherwise.
                - right_foot_contact (bool): True if the right foot is in contact, False otherwise.
        """

        left_wrench, rigth_wrench = self.get_feet_wrench()
        left_foot_contact = left_wrench[2] > 0.1 * self.mass
        right_foot_contact = rigth_wrench[2] > 0.1 * self.mass
        return left_foot_contact, right_foot_contact

    def set_visualize_robot_flag(self, visualize_robot) -> None:
        """
        Sets the flag to visualize the robot and initializes the viewer if the flag is set to True.

        Args:
            visualize_robot (bool): A flag indicating whether to visualize the robot.
        """

        self.visualize_robot_flag = visualize_robot
        if self.visualize_robot_flag:
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)

    def set_base_pose_in_mujoco(self, xyz_rpy) -> None:
        """
        Set the base pose in the MuJoCo simulator.

        Args:
            xyz_rpy (array-like): A 6-element array where the first three elements represent the 
                      position (x, y, z) and the last three elements represent the 
                      orientation in roll, pitch, and yaw (rpy) angles.
        Returns:
            None
        Notes:
            - This function converts the roll, pitch, and yaw angles to a quaternion and sets the 
            base position and orientation in the MuJoCo simulator's data structure.
        """

        base_xyz_quat = np.zeros(7)
        base_xyz_quat[:3] = xyz_rpy[:3]
        base_xyz_quat[3:] = self.RPY_to_quat(xyz_rpy[3], xyz_rpy[4], xyz_rpy[5])
        base_xyz_quat[2] = base_xyz_quat[2]
        self.data.qpos[:7] = base_xyz_quat

    def set_joint_vector_in_mujoco(self, pos) -> None:
        """
        Sets the joint positions in the MuJoCo simulation.
        This method converts the given joint positions from the robot's coordinate
        system to MuJoCo's coordinate system and updates the joint positions in the
        MuJoCo simulation accordingly.

        Args:
            pos (list or array-like): A list or array of joint positions in the robot's
                          coordinate system.
        Returns:
            None
        """
        
        pos_muj = self.convert_vector_to_mujoco(pos)
        indexes_joint = self.model.jnt_qposadr[1:]
        for i in range(self.robot_model.NDoF):
            self.data.qpos[indexes_joint[i]] = pos_muj[i]

    def set_input(self, input) -> None:
        """
        Sets the input for the MuJoCo simulator.
        This method converts the provided input vector to a format compatible 
        with MuJoCo and assigns it to the simulator's control data.

        Args:
            input (array-like): The input vector to be set for the simulator.
        Returns:
            None
        """
        
        input_muj = self.convert_vector_to_mujoco(input)
        self.data.ctrl = input_muj
        np.copyto(self.data.ctrl, input_muj)

    def set_position_input(self, pos) -> None:
        """
        Sets the desired position input for the simulator.
        This method converts the given position vector to the MuJoCo format and 
        sets it as the desired position. It also enables position control.
        
        Args:
            pos (list or np.ndarray): The position vector to be set as the desired position.
        """

        pos_muj = self.convert_vector_to_mujoco(pos)
        self.desired_pos = pos_muj
        self.postion_control = True

    def create_mapping_vector_to_mujoco(self) -> None:
        """
        Creates a mapping vector from the robot model's joint names to the Mujoco joint order.
        This function initializes the `to_mujoco` attribute as an empty list and populates it with
        indices that map each Mujoco joint to its corresponding index in the robot model's joint
        name list. If a Mujoco joint is not found in the joint name list, a ValueError is raised.

        Returns:
            None
        Raises:
            ValueError: If a Mujoco joint is not found in the robot model's joint name list.
        """

        # This function creates the to_mujoco map
        self.to_mujoco = []
        for mujoco_joint in self.robot_model.mujoco_joint_order:
            try:
                index = self.robot_model.joint_name_list.index(mujoco_joint)
                self.to_mujoco.append(index)
            except ValueError:
                raise ValueError(
                    f"Mujoco joint '{mujoco_joint}' not found in joint list."
                )

    def create_mapping_vector_from_mujoco(self) -> None:
        """
        Creates a mapping vector from the MuJoCo joint order to the robot model's joint order.
        This function initializes the `from_mujoco` attribute as an empty list and then populates it
        with indices that map each joint in the robot model's joint name list to its corresponding
        index in the MuJoCo joint order list.

        Returns:
            None
        Raises:
            ValueError: If a joint name in the robot model's joint name list is not found in the
                MuJoCo joint order list.
        """

        # This function creates the to_mujoco map
        self.from_mujoco = []
        for joint in self.robot_model.joint_name_list:
            try:
                index = self.robot_model.mujoco_joint_order.index(joint)
                self.from_mujoco.append(index)
            except ValueError:
                raise ValueError(
                    f"Joint name list  joint '{joint}' not found in mujoco list."
                )

    def convert_vector_to_mujoco(self, array_in) -> np.ndarray:
        """
        Converts a given vector to the MuJoCo format.
        This function takes an input array and reorders its elements according to the
        mapping defined in `self.to_mujoco` for the degrees of freedom (DoF) of the robot model.

        Args:
            array_in (array-like): The input array to be converted.
        Returns:
            np.ndarray: The converted array in MuJoCo format.
        """

        out_muj = np.asarray(
            [array_in[self.to_mujoco[item]] for item in range(self.robot_model.NDoF)]
        )
        return out_muj

    def convert_from_mujoco(self, array_muj):
        """
        Converts an array from MuJoCo format to a classic format.

        Args:
            array_muj (np.ndarray): The input array in MuJoCo format.
        Returns:
            np.ndarray: The converted array in classic format.
        """

        out_classic = np.asarray(
            [array_muj[self.from_mujoco[item]] for item in range(self.robot_model.NDoF)]
        )
        return out_classic

    def step(self, n_step=1, visualize=True) -> None:
        """
        Advances the simulation by a specified number of steps.

        Args:
            n_step (int?): The number of simulation steps to advance. Default is 1.
            visualize (bool?): If True, renders the simulation after stepping. Default is True.
        Notes:
            - If position control is enabled, the control input is computed using
            proportional-derivative (PD) control based on the desired position.
            - The control input is applied to the simulation, and the simulation
            is advanced by the specified number of steps.
            - If visualization is enabled, the simulation is rendered after stepping.
        """

        if self.postion_control:
            for _ in range(n_step):
                s, s_dot, tau = self.get_state(use_mujoco_convention=True)
                kp_muj = self.convert_vector_to_mujoco(
                    self.robot_model.kp_position_control
                )
                kd_muj = self.convert_vector_to_mujoco(
                    self.robot_model.kd_position_control
                )
                ctrl = kp_muj * (self.desired_pos - s) - kd_muj * s_dot
                self.data.ctrl = ctrl
                np.copyto(self.data.ctrl, ctrl)
                mujoco.mj_step(self.model, self.data)
                mujoco.mj_step1(self.model, self.data)
                mujoco.mj_forward(self.model, self.data)
        else:
            mujoco.mj_step(self.model, self.data, n_step)
            mujoco.mj_step1(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
  
        if self.visualize_robot_flag:
            self.viewer.render()

    def step_with_motors(self, n_step, torque) -> None:
        """
        Advances the simulation by a specified number of steps while applying motor torques.

        Args:
            n_step (int): The number of simulation steps to advance.
            torque (array-like): An array of torques to be applied to the motors. The length of the array should match the number of degrees of freedom (NDoF) of the robot.
        Returns:
            None
        """
        
        indexes_joint_acceleration = self.model.jnt_dofadr[1:]
        s_dot_dot = self.data.qacc[indexes_joint_acceleration[0] :]
        for _ in range(n_step):
            indexes_joint_acceleration = self.model.jnt_dofadr[1:]
            s_dot_dot = self.data.qacc[indexes_joint_acceleration[0] :]
            s_dot = self.data.qvel[indexes_joint_acceleration[0] :]
            input = np.asarray(
                [
                    self.Im[self.to_mujoco[item]] * s_dot_dot[item]
                    + self.kv_motors[self.to_mujoco[item]] * s_dot[item]
                    + torque[self.to_mujoco[item]]
                    for item in range(self.robot_model.NDoF)
                ]
            )

            self.set_input(input)
            self.step(n_step=1, visualize=False)
        if self.visualize_robot_flag:
            self.viewer.render()

    def compute_misalignment_gravity_fun(self) -> None:
        """
        Computes the misalignment gravity function and assigns it to the instance variable `error_mis`.
        This function creates a symbolic 4x4 matrix `H` and a symbolic variable `theta`. It calculates 
        `theta` as the dot product of the vector [0, 0, 1] and the first three elements of the third 
        column of `H`, minus 1. It then defines a CasADi function `error` that takes `H` as input and 
        returns `theta` as output. This function is assigned to the instance variable `error_mis`.
        
        Returns:
            None
        """
        
        H = cs.SX.sym("H", 4, 4)
        theta = cs.SX.sym("theta")
        theta = cs.dot([0, 0, 1], H[:3, 2]) - 1
        error = cs.Function("error", [H], [theta])
        self.error_mis = error

    def check_feet_status(self, s: np.ndarray, H_b: np.ndarray) -> tuple:
        """
        Checks the status of the robot's feet to determine if they are in contact with the ground and aligned properly.
        
        Args:
            s (np.ndarray): The state vector of the robot. Shape must be (NDoF,).
            H_b (np.ndarray): The homogeneous transformation matrix representing the base pose of the robot. Shape must be (4, 4).
        Returns:
            tuple:
                bool: True if both feet are in contact with the ground and properly aligned, False otherwise.
                float: The total misalignment error of both feet.
        """

        left_foot_pose = self.robot_model.H_left_foot(H_b, s)
        rigth_foot_pose = self.robot_model.H_right_foot(H_b, s)
        left_foot_z = left_foot_pose[2, 3]
        rigth_foot_z = rigth_foot_pose[2, 3]
        left_foot_contact = not (left_foot_z > 0.1)
        rigth_foot_contact = not (rigth_foot_z > 0.1)
        misalignment_left = self.error_mis(left_foot_pose)
        misalignment_rigth = self.error_mis(rigth_foot_pose)
        left_foot_condition = abs(left_foot_contact * misalignment_left)
        rigth_foot_condition = abs(rigth_foot_contact * misalignment_rigth)
        misalignment_error = left_foot_condition + rigth_foot_condition
        if (
            abs(left_foot_contact * misalignment_left) > 0.02
            or abs(rigth_foot_contact * misalignment_rigth) > 0.02
        ):
            return False, misalignment_error

        return True, misalignment_error

    def get_feet_wrench(self) -> tuple:
        """
        Computes the wrenches (forces and torques) applied to the left and right feet of the robot.
        This method calculates the resulting wrenches on the robot's feet based on the current state
        and contact forces. It iterates through all contacts, determines if the contact is with the 
        left or right foot, and accumulates the wrench for each foot.

        Returns:
            tuple: A tuple containing two numpy arrays:
                - left_foot_wrench (numpy.ndarray): A 6-element array representing the wrench on the left foot.
                - right_foot_wrench (numpy.ndarray): A 6-element array representing the wrench on the right foot.
        """

        left_foot_wrench = np.zeros(6)
        rigth_foot_wrench = np.zeros(6)
        s, s_dot, tau = self.get_state()
        H_b = self.get_base()
        self.H_left_foot_num = np.array(self.H_left_foot(H_b, s))
        self.H_right_foot_num = np.array(self.H_right_foot(H_b, s))
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            c_array = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, i, c_array)
            name_contact = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom[1])
            )
            w_H_contact = np.eye(4)
            w_H_contact[:3, :3] = contact.frame.reshape(3, 3)
            w_H_contact[:3, 3] = contact.pos
            if (
                name_contact == self.robot_model.right_foot_rear_ct
                or name_contact == self.robot_model.right_foot_front_ct
            ):
                RF_H_contact = np.linalg.inv(self.H_right_foot_num) @ w_H_contact
                wrench_RF = self.compute_resulting_wrench(RF_H_contact, c_array)
                rigth_foot_wrench[:] += wrench_RF.reshape(6)
            elif (
                name_contact == self.robot_model.left_foot_front_ct
                or name_contact == self.robot_model.left_foot_rear_ct
            ):
                LF_H_contact = np.linalg.inv(self.H_left_foot_num) @ w_H_contact
                wrench_LF = self.compute_resulting_wrench(LF_H_contact, c_array)
                left_foot_wrench[:] += wrench_LF.reshape(6)
        return (left_foot_wrench, rigth_foot_wrench)

    def compute_resulting_wrench(self, b_H_a, force_torque_a) -> np.ndarray:
        """
        Compute the resulting wrench (force and torque) in frame b given the wrench in frame a.

        Args:
            b_H_a (np.ndarray): A 4x4 homogeneous transformation matrix representing the pose of frame a relative to frame b.
            force_torque_a (np.ndarray): A 6-element array representing the force and torque in frame a.
        Returns:
            np.ndarray: A 6x1 array representing the force and torque in frame b.
        """

        p = b_H_a[:3, 3]
        R = b_H_a[:3, :3]
        adjoint_matrix = np.zeros([6, 6])
        adjoint_matrix[:3, :3] = R
        adjoint_matrix[3:, :3] = np.cross(p, R)
        adjoint_matrix[3:, 3:] = R
        force_torque_b = adjoint_matrix @ force_torque_a.reshape(6, 1)
        return force_torque_b

    # note that for mujoco the ordering is w,x,y,z
    def get_base(self) -> np.ndarray:
        """
        Computes the transformation matrix representing the base position and orientation 
        of the model in the simulation.

        Returns:
            np.ndarray: A 4x4 transformation matrix where the upper-left 3x3 submatrix 
                represents the rotation matrix derived from the quaternion, and 
                the upper-right 3x1 submatrix represents the translation vector 
                (position) of the base.
        """
        indexes_joint = self.model.jnt_qposadr[1:]

        # Extract quaternion components
        w, x, y, z = self.data.qpos[3 : indexes_joint[0]]

        # Calculate rotation matrix
        rot_mat = np.array(
            [
                [
                    1 - 2 * y * y - 2 * z * z,
                    2 * x * y - 2 * z * w,
                    2 * x * z + 2 * y * w,
                    0,
                ],
                [
                    2 * x * y + 2 * z * w,
                    1 - 2 * x * x - 2 * z * z,
                    2 * y * z - 2 * x * w,
                    0,
                ],
                [
                    2 * x * z - 2 * y * w,
                    2 * y * z + 2 * x * w,
                    1 - 2 * x * x - 2 * y * y,
                    0,
                ],
                [0, 0, 0, 1],
            ]
        )

        # Set up transformation matrix
        trans_mat = np.eye(4)
        trans_mat[:3, :3] = rot_mat[:3, :3]
        trans_mat[:3, 3] = self.data.qpos[:3]
        # Return transformation matrix
        return trans_mat

    def get_base_velocity(self) -> np.ndarray:
        """
        Retrieve the base velocity of the model.
        This method extracts the base velocity from the simulation data. It uses the joint degrees of freedom
        addresses to determine the relevant indices and returns the velocity of the base.

        Returns:
            numpy.ndarray: The base velocity of the model.
        """
        
        indexes_joint_velocities = self.model.jnt_dofadr[1:]
        return self.data.qvel[: indexes_joint_velocities[0]]

    def get_state(self, use_mujoco_convention=False) -> tuple:
        """
        Returns the state of the robot either in mujoco_convention or classic one.
        If the model has no joints, an empty state is returned either way.
        
        Args:
            use_mujoco_convention (bool): If True, the state is returned in mujoco_convention. If False, it is returned in classic convention.
        Returns:
            s_out (np.array): joint positions
            s_dot_out (np.array): joint velocities
            tau_out (np.array): joint torques
        """
        indexes_joint = self.model.jnt_qposadr[1:]
        indexes_joint_velocities = self.model.jnt_dofadr[1:]
        if len(indexes_joint) == 0:
            return np.array([]), np.array([]), np.array([])
        s = self.data.qpos[indexes_joint[0] :]
        s_dot = self.data.qvel[indexes_joint_velocities[0] :]
        tau = self.data.ctrl
        if use_mujoco_convention:
            return s, s_dot, tau
        s_out = self.convert_from_mujoco(s)
        s_dot_out = self.convert_from_mujoco(s_dot)
        tau_out = self.convert_from_mujoco(tau)
        return s_out, s_dot_out, tau_out

    def close(self) -> None:
        """
        Closes the simulator viewer if the visualization flag is set.
        This method checks if the `visualize_robot_flag` is True. If it is, it closes the viewer associated with the simulator.
        """

        if self.visualize_robot_flag:
            self.viewer.close()

    def visualize_robot(self) -> None:
        self.viewer.render()

    def get_simulation_time(self) -> float:
        """
        Retrieve the current simulation time.

        Returns:
            float: The current time of the simulation.
        """

        return self.data.time

    def get_simulation_frequency(self) -> float:
        return self.model.opt.timestep

    def RPY_to_quat(self, roll, pitch, yaw) -> list:
        """
        Convert roll, pitch, and yaw angles to a quaternion.
        The quaternion is returned as a list of four elements [qw, qx, qy, qz].
        
        Args:
            roll (float): The roll angle in radians.
            pitch (float): The pitch angle in radians.
            yaw (float): The yaw angle in radians.
        Returns:
            list: A list containing the quaternion [qw, qx, qy, qz].
        """


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

        return [qw, qx, qy, qz]

    def close_visualization(self) -> None:
        """
        Closes the visualization window if it is open.
        This method checks if the `visualize_robot_flag` is True. If it is, it closes the viewer associated with the simulator.
        """

        if self.visualize_robot_flag:
            self.viewer.close()

    def run(self, tf: float,  callbacks: List[Callback] = [], visualise: bool = False) -> None:
        """
        Run the simulation.
        This method runs the simulation until the `should_stop` flag is set to True.
        """

        self.reset()
        for callback in callbacks:
            callback.on_simulation_start()
            callback.set_simulator(self)
        self.set_visualize_robot_flag(visualise)
        while self.t < tf:
            self.step()

            contact = MjContactInfo(self.t, self.iter, self.data.contact)
            if not contact.is_none():
                self.contacts.append(contact)
            
            self.t = self.get_simulation_time()
            for callback in callbacks:
                d = {
                    "contact" : contact,
                }
                callback.on_simulation_step(self.t, self.iter, self.data, d)
            self.iter += 1
            if self.should_stop:
                break
        for callback in callbacks:
            callback.on_simulation_end()
         
    def reset(self) -> None:
        """
        Resets the simulator to the initial state.
        This method resets the simulator to the initial state by setting the control input to zero and 
        calling the `step` method with zero steps.
        """

        self.set_input(np.zeros(self.robot_model.NDoF))
        self.step(0)
        self.t = 0
        self.iter = 0