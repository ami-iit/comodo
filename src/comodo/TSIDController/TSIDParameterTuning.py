import numpy as np


class TSIDParameterTuning:
    def __init__(self, active_joint) -> None:
        self.CoM_Kp = 15.0
        self.CoM_Kd = 7.0
        postural_Kp_temp = (np.array([130,80,20,60,130,80,20,60,150,20,20,180,140,20,150,20,20,180,140,20])* 2.8)  # TODO symmetry
        self.postural_Kp = []
        self.active_joints = active_joint
        for idx,item in enumerate(active_joint):
            if(item):
                self.postural_Kp.append(postural_Kp_temp[idx])
        self.postural_weight = 10 * np.ones(len(self.postural_Kp))
        self.foot_tracking_task_kp_lin = 30.0
        self.foot_tracking_task_kd_lin = 7.0
        self.foot_tracking_task_kp_ang = 300.0
        self.foot_tracking_task_kd_ang = 10.0
        self.root_tracking_task_weight = 10 * np.ones(3)
        self.root_link_kp_ang = 20.0
        self.root_link_kd_ang = 10.0

    def set_postural_gain(self, arm, leg):
        postural_tot = np.concatenate([arm, arm, leg, leg])
        self.postural_Kp = np.concatenate(postural_tot)
        for idx,item in enumerate(self.active_joint):
            if(item):
                self.postural_Kp.append(postural_tot[idx])

    def set_foot_task(self, kp_lin, kp_ang,):
        self.foot_tracking_task_kp_lin = kp_lin
        self.foot_tracking_task_kd_lin = np.power(kp_lin, 1 / 2)
        self.foot_tracking_task_kp_ang = kp_ang
        self.foot_tracking_task_kd_ang = np.power(kp_ang,1/2)

    def set_weights(self, postural_weight, root_weight):
        self.postural_weight = postural_weight
        self.root_tracking_task_weight = root_weight

    def set_root_task(self, kp_ang:float):
        self.root_link_kp_ang = kp_ang
        self.root_link_kd_ang = np.power(kp_ang, 1/2)

    def set_com_task(self,kp_com:float):
        self.CoM_Kd = np.power(kp_com,1/2)
        self.CoM_Kp = kp_com

    def set_from_x_k(self, x_k):
        # self.set_postural_gain(x_k[:4], x_k[4:10])
        # self.set_foot_task(x_k[10], x_k[11])
        # self.set_root_task(x_k[12])
        # self.set_com_task(x_k[13])
        self.set_foot_task(x_k[0], x_k[1])
        self.set_root_task(x_k[2])
        self.set_com_task(x_k[3])
