from comodo.abstractClasses.simulator import Simulator
import mujoco
import math
import numpy as np
import mujoco_viewer
import copy
import casadi as cs


class MujocoSimulator(Simulator):
    def __init__(self) -> None:
        self.desired_pos = None
        self.postion_control = False
        self.compute_misalignment_gravity_fun()
        super().__init__()

    def load_model(self, robot_model, s, xyz_rpy, kv_motors=None, Im=None):
        self.robot_model = robot_model
        
        mujoco_xml = robot_model.get_mujoco_model()
        self.model = mujoco.MjModel.from_xml_string(mujoco_xml)
        self.data = mujoco.MjData(self.model)
        self.create_mapping_vector_from_mujoco()
        self.create_mapping_vector_to_mujoco()
        mujoco.mj_forward(self.model, self.data)
        self.set_joint_vector_in_mujoco(s)
        self.set_base_pose_in_mujoco(xyz_rpy=xyz_rpy)
        mujoco.mj_forward(self.model, self.data)
        self.visualize_robot_flag = False

        self.Im = Im if not (Im is None) else np.zeros(self.robot_model.NDoF)
        self.kv_motors = (
            kv_motors if not (kv_motors is None) else np.zeros(self.robot_model.NDoF)
        )
        self.H_left_foot = copy.deepcopy(self.robot_model.H_left_foot)
        self.H_right_foot = copy.deepcopy(self.robot_model.H_right_foot)
        self.H_left_foot_num = None 
        self.H_right_foot_num = None 
        self.mass = self.robot_model.get_total_mass()


    def get_contact_status(self):
        left_wrench, rigth_wrench = self.get_feet_wrench()
        left_foot_contact = left_wrench[2] > 0.1*self.mass
        right_foot_contact = rigth_wrench[2] > 0.1*self.mass
        return left_foot_contact, right_foot_contact

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
        pos_muj = self.convert_vector_to_mujoco(pos)
        indexes_joint = self.model.jnt_qposadr[1:]
        for i in range(self.robot_model.NDoF):
            self.data.qpos[indexes_joint[i]] = pos_muj[i]

    def set_input(self, input):
        input_muj = self.convert_vector_to_mujoco(input)
        self.data.ctrl = input_muj
        np.copyto(self.data.ctrl, input_muj)

    def set_position_input(self, pos):
        pos_muj = self.convert_vector_to_mujoco(pos)
        self.desired_pos = pos_muj
        self.postion_control = True

    def create_mapping_vector_to_mujoco(self):
        # This function creates the to_mujoco map
        self.to_mujoco = []
        for mujoco_joint in self.robot_model.mujoco_joint_order:
            try:
                index = self.robot_model.joint_name_list.index(mujoco_joint)
                self.to_mujoco.append(index)
            except ValueError:
                raise ValueError(f"Mujoco joint '{mujoco_joint}' not found in joint list.")
        
    def create_mapping_vector_from_mujoco(self):
        # This function creates the to_mujoco map
        self.from_mujoco = []
        for joint in self.robot_model.joint_name_list:
            try:
                index = self.robot_model.mujoco_joint_order.index(joint)
                self.from_mujoco.append(index)
            except ValueError:
                raise ValueError(f"Joint name list  joint '{joint}' not found in mujoco list.")
        
    def convert_vector_to_mujoco (self, array_in): 
        out_muj = np.asarray([array_in[self.to_mujoco[item]] for item in range(self.robot_model.NDoF)]) 
        return out_muj
    
    def convert_from_mujoco(self, array_muj):
        out_classic = np.asarray([array_muj[self.from_mujoco[item]] for item in range(self.robot_model.NDoF)]) 
        return out_classic

    def step(self, n_step=1, visualize=True):
        if self.postion_control:
            for _ in range(n_step):
                s, s_dot, tau = self.get_state(use_mujoco_convention=True)
                kp_muj = self.convert_vector_to_mujoco(self.robot_model.kp_position_control)
                kd_muj = self.convert_vector_to_mujoco(self.robot_model.kd_position_control)
                ctrl = (
                    kp_muj * (self.desired_pos - s)
                    - kd_muj * s_dot
                )
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

    def step_with_motors(self, n_step, torque):
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

    def compute_misalignment_gravity_fun(self):
        H = cs.SX.sym("H", 4, 4)
        theta = cs.SX.sym("theta")
        theta = cs.dot([0, 0, 1], H[:3, 2]) - 1
        error = cs.Function("error", [H], [theta])
        self.error_mis = error

    def check_feet_status(self, s, H_b):
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

    def get_feet_wrench(self):
        left_foot_wrench = np.zeros(6)
        rigth_foot_wrench = np.zeros(6)
        s,s_dot, tau = self.get_state()
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
    
    def compute_resulting_wrench(self, b_H_a, force_torque_a):
        p = b_H_a[:3, 3]
        R = b_H_a[:3, :3]
        adjoint_matrix = np.zeros([6, 6])
        adjoint_matrix[:3, :3] = R
        adjoint_matrix[3:, :3] = np.cross(p, R)
        adjoint_matrix[3:, 3:] = R
        force_torque_b = adjoint_matrix @ force_torque_a.reshape(6, 1)
        return force_torque_b

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

    def get_state(self, use_mujoco_convention = False):
        indexes_joint = self.model.jnt_qposadr[1:]
        indexes_joint_velocities = self.model.jnt_dofadr[1:]
        s = self.data.qpos[indexes_joint[0] :]
        s_dot = self.data.qvel[indexes_joint_velocities[0] :]
        tau = self.data.ctrl
        if(use_mujoco_convention):
            return s,s_dot,tau 
        s_out = self.convert_from_mujoco(s)
        s_dot_out = self.convert_from_mujoco(s_dot)
        tau_out = self.convert_from_mujoco(tau)
        return s_out, s_dot_out, tau_out

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
