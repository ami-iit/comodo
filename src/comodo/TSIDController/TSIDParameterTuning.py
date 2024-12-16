import numpy as np 

class TSIDParameterTuning():
    def __init__(self) -> None:
        self.CoM_Kp = 15.0
        self.CoM_Kd = 7.0
        self.postural_Kp = np.array([130,80,20,60,130,80,20,60,150,20,20,180,140,20,150,20,20,180,140,20])* 2.8 
        self.postural_kd = np.power(self.postural_Kp, 1 / 2) / 10
        self.postural_weight = 10*np.ones(len(self.postural_Kp))
        self.foot_tracking_task_kp_lin = 30.0
        self.foot_tracking_task_kd_lin = 7.0
        self.foot_tracking_task_kp_ang = 300.0
        self.foot_tracking_task_kd_ang = 10.0
        self.root_tracking_task_weight = 10*np.ones(3)
        self.root_link_kp_ang = 20.0
        self.root_link_kd_ang = 10.0

    def set_postural_gain(self, arm, leg, arm_kd, leg_kd):
        self.postural_Kp = np.concatenate([leg,leg,arm,arm])
        self.postural_kd = np.concatenate([leg_kd, leg_kd, arm_kd, arm_kd])
    def set_foot_task (self,kp_lin, kd_lin, kp_ang, kd_ang): 
        self.foot_tracking_task_kp_lin = kp_lin
        self.foot_tracking_task_kd_lin = kd_lin
        self.foot_tracking_task_kp_ang = kp_ang
        self.foot_tracking_task_kd_ang = kd_ang

    def set_weights (self, postural_weight, root_weight): 
        self.postural_weight = postural_weight
        self.root_tracking_task_weight = root_weight
 
    def set_root_task(self, kp_ang, kd_ang):
        self.root_link_kp_ang = kp_ang
        self.root_link_kd_ang = kd_ang
    
    def set_com_task(self,kp_com, kd_com): 
        self.CoM_Kd = kd_com
        self.CoM_Kp = kp_com
        
    def set_from_x_k(self,x_k):         
        self.set_postural_gain(x_k[:4], x_k[4:10], x_k[10:14], x_k[14:20])
        self.set_foot_task(x_k[20], x_k[21], x_k[22], x_k[23])
        self.set_root_task(x_k[24], x_k[25])
        self.set_com_task(x_k[26], x_k[27])