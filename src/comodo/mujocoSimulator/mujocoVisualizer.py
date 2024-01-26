from comodo.abstractClasses.simulator import Simulator
import mujoco
import math
import numpy as np
import mujoco_viewer


class MujocoVisualizer():
    
    def __init__(self) -> None:
        pass 
    def load_model(self, robot_model, s, xyz_rpy, kv_motors=None, Im=None):
        self.robot_model = robot_model
        mujoco_xml = robot_model.get_mujoco_model()
        self.model = mujoco.MjModel.from_xml_string(mujoco_xml)
        self.data = mujoco.MjData(self.model)
        self.set_joint_vector_in_mujoco(s)
        self.set_base_pose_in_mujoco(xyz_rpy=xyz_rpy)
        mujoco.mj_forward(self.model, self.data)
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

    def update_vis(self, s, xyz_rpy): 
        self.set_base_pose_in_mujoco(self,xyz_rpy)
        self.set_joint_vector_in_mujoco(s)
        mujoco.mj_forward(self.model, self.data)
        self.viewer.render()

    def get_feet_wrench(self):
        left_foot_rear_wrench = np.zeros(6)
        left_foot_front_wrench = np.zeros(6)
        rigth_foot_rear_wrench = np.zeros(6)
        rigth_foot_front_wrench = np.zeros(6)

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            c_array = np.zeros(6, dtype=np.float64)
            force_global = np.zeros(3, dtype=np.float64)
            torque_global = np.zeros(3, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, i, c_array)
            frame_i = np.transpose(contact.frame.reshape(3, 3))
            mujoco.mju_mulMatTVec(force_global[:3], frame_i, c_array[:3])
            mujoco.mju_mulMatTVec(torque_global[:3], frame_i, c_array[3:])
            name_contact = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom[1])
            )
            force_global[2] = -force_global[2]
            if name_contact == self.robot_model.right_foot_rear_ct:
                rigth_foot_rear_wrench = np.concatenate((force_global, torque_global))
            elif name_contact == self.robot_model.right_foot_front_ct:
                rigth_foot_front_wrench = np.concatenate((force_global, torque_global))
            elif name_contact == self.robot_model.left_foot_front_ct:
                left_foot_front_wrench = np.concatenate((force_global, torque_global))
            elif name_contact == self.robot_model.left_foot_rear_ct:
                left_foot_rear_wrench = np.concatenate((force_global, torque_global))
        return (
            left_foot_front_wrench,
            left_foot_rear_wrench,
            rigth_foot_front_wrench,
            rigth_foot_rear_wrench,
        )

    def close(self):
            self.viewer.close()

    def visualize_robot(self):
        self.viewer.render()

    def get_simulation_time(self):
        return self.data.time

    def get_simulation_frequency(self):
        return self.model.opt.timestep

    def close_visualization(self):
        if self.visualize_robot_flag:
            self.viewer.close()
