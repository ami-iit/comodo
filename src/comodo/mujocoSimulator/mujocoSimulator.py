from comodo.abstractClasses.simulator import Simulator
import mujoco
import math
import numpy as np
import mujoco_viewer


class MujocoSimulator(Simulator):
    def __init__(self) -> None:
        super().__init__()

    def load_model(self, robot_model, s, xyz_rpy, kv_motors=None, Im=None):
        self.robot_model = robot_model
        mujoco_xml = robot_model.get_mujoco_model()
        self.model = mujoco.MjModel.from_xml_string(mujoco_xml)
        # mujoco.mj_saveLastXML("/home/carlotta/iit_ws/element_hardware-intelligence/muj_mod.xml",self.model)
        # mujoco.MjModel.mj_saveModel()
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.set_joint_vector_in_mujoco(s)
        self.set_base_pose_in_mujoco(xyz_rpy=xyz_rpy)
        mujoco.mj_forward(self.model, self.data)
        self.visualize_robot_flag = False

        self.Im = Im if not (Im is None) else np.zeros(self.robot_model.NDoF)
        self.kv_motors = (
            kv_motors if not (kv_motors is None) else np.zeros(self.robot_model.NDoF)
        )

    def set_visualize_robot_flag(self, visualize_robot):
        self.visualize_robot_flag = visualize_robot
        if self.visualize_robot_flag:
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)

    def set_base_pose_in_mujoco(self, xyz_rpy):
        base_xyz_quat = np.zeros(7)
        base_xyz_quat[:3] = xyz_rpy[:3]
        base_xyz_quat[3:] = self.RPY_to_quat(xyz_rpy[3], xyz_rpy[4], xyz_rpy[5])
        base_xyz_quat[2] = base_xyz_quat[2]
        self.data.qpos[:7] = base_xyz_quat

    def set_joint_vector_in_mujoco(self, pos):
        indexes_joint = self.model.jnt_qposadr[1:]
        for i in range(self.robot_model.NDoF):
            self.data.qpos[indexes_joint[i]] = pos[i]

    def set_input(self, input):
        self.data.ctrl = input
        np.copyto(self.data.ctrl, input)

    def step(self, n_step=1, visualize=True):
        mujoco.mj_step(self.model, self.data, n_step)
        mujoco.mj_step1(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        if self.visualize_robot_flag and visualize:
            self.viewer.render()

    def step_with_motors(self, n_step, torque):

        indexes_joint_acceleration = self.model.jnt_dofadr[1:]
        s_dot_dot = self.data.qacc[indexes_joint_acceleration[0] :]
        for _ in range(n_step):

            indexes_joint_acceleration = self.model.jnt_dofadr[1:]
            s_dot_dot = self.data.qacc[indexes_joint_acceleration[0] :]
            s_dot = self.data.qvel[indexes_joint_acceleration[0] :]
            input = np.asarray(
                [
                    self.Im[item] * s_dot_dot[item]
                    + self.kv_motors[item] * s_dot[item]
                    + torque[item]
                    for item in range(self.robot_model.NDoF)
                ]
            )

            self.set_input(input)
            self.step(n_step=1, visualize=False)
        if self.visualize_robot_flag:
            self.viewer.render()

    # note that for mujoco the ordering is w,x,y,z
    def get_base(self):

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

    def get_base_velocity(self):
        indexes_joint_velocities = self.model.jnt_dofadr[1:]
        return self.data.qvel[: indexes_joint_velocities[0]]

    def get_state(self):
        indexes_joint = self.model.jnt_qposadr[1:]
        indexes_joint_velocities = self.model.jnt_dofadr[1:]
        s = self.data.qpos[indexes_joint[0] :]
        s_dot = self.data.qvel[indexes_joint_velocities[0] :]
        tau = self.data.ctrl
        return s, s_dot, tau

    def close(self):
        if self.visualize_robot_flag:
            self.viewer.close()

    def visualize_robot(self):
        self.viewer.render()

    def get_simulation_time(self):
        return self.data.time

    def get_simulation_frequency(self):
        return self.model.opt.timestep

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

        return [qw, qx, qy, qz]

    def close_visualization(self):
        if self.visualize_robot_flag:
            self.viewer.close()
