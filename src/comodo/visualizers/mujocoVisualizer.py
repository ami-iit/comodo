import mujoco
import math
import numpy as np
import mujoco_viewer
import scipy 

class MujocoVisualizer():
    
    def __init__(self) -> None:
        pass 
    
    def load_model(self, robot_model, s, H_b):
        self.robot_model = robot_model
        mujoco_xml = robot_model.get_mujoco_model()
        self.model = mujoco.MjModel.from_xml_string(mujoco_xml)
        self.data = mujoco.MjData(self.model)
        self.set_joint_vector_in_mujoco(s)
        xyz, quat = self.from_H_b_to_xyz_quat(H_b)
        self.set_base_pose_in_mujoco(xyz, quat)
        mujoco.mj_forward(self.model, self.data)
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)

    def set_base_pose_in_mujoco(self, xyz, quat):
        indexes_joint = self.model.jnt_qposadr[1:]
        # Extract quaternion components
        self.data.qpos[3 : indexes_joint[0]] = quat[:]
        self.data.qpos[:3] = xyz[:]

    def set_joint_vector_in_mujoco(self, pos):
        indexes_joint = self.model.jnt_qposadr[1:]
        for i in range(self.robot_model.NDoF):
            self.data.qpos[indexes_joint[i]] = pos[i]

    def update_vis(self, s, H_b): 
        xyz, quat = self.from_H_b_to_xyz_quat(H_b)
        self.set_base_pose_in_mujoco(xyz, quat)
        self.set_joint_vector_in_mujoco(s)
        mujoco.mj_forward(self.model, self.data)
        self.viewer.render()

    def from_H_b_to_xyz_quat(self, H_b):
        xyz = H_b[:3,3] 
        rot_mat = scipy.spatial.transform.Rotation.from_matrix(H_b[:3,:3])
        quat_xyzw = rot_mat.as_quat()
        # note that for mujoco the ordering is w,x,y,z
        quat_wxyz = np.asarray([quat_xyzw[3],quat_xyzw[0],quat_xyzw[1],quat_xyzw[2]])
        return xyz, quat_wxyz
    def close(self):
            self.viewer.close()

    def visualize_robot(self):
        self.viewer.render()