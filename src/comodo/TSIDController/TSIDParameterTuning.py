import numpy as np


class TSIDParameterTuning:
    def __init__(self) -> None:
        self.CoM_Kp = 15.0
        self.CoM_Kd = 7.0
        self.postural_Kp = (
            np.array(
                [
                    130,
                    80,
                    20,
                    60,
                    130,
                    80,
                    20,
                    60,
                    150,
                    20,
                    20,
                    180,
                    140,
                    20,
                    150,
                    20,
                    20,
                    180,
                    140,
                    20,
                ]
            )
            * 2.8
        )  # TODO symmetry
        self.postural_weight = 10 * np.ones(len(self.postural_Kp))
        self.foot_tracking_task_kp_lin = 30.0
        self.foot_tracking_task_kd_lin = 7.0
        self.foot_tracking_task_kp_ang = 300.0
        self.foot_tracking_task_kd_ang = 10.0
        self.root_tracking_task_weight = 10 * np.ones(3)
        self.root_link_kp_ang = 20.0
        self.root_link_kd_ang = 10.0

    def set_postural_gain(self, arm, leg):
        self.postural_Kp = np.concatenate([arm, arm, leg, leg])

    def set_foot_task(self, kp_lin, kd_lin, kp_ang, kd_ang):
        self.foot_tracking_task_kp_lin = kp_lin
        self.foot_tracking_task_kd_lin = kd_lin
        self.foot_tracking_task_kp_ang = kp_ang
        self.foot_tracking_task_kd_ang = kd_ang

    def set_weights(self, postural_weight, root_weight):
        self.postural_weight = postural_weight
        self.root_tracking_task_weight = root_weight

    def set_root_task(self, kp_ang, kd_ang):
        self.root_link_kp_ang = kp_ang
        self.root_link_kd_ang = kd_ang

    def set_com_task(self, kp_com, kd_com):
        self.CoM_Kd = kd_com
        self.CoM_Kp = kp_com

    def set_from_x_k(self, x_k):
        self.set_postural_gain(x_k[:4], x_k[4:10])
        self.set_foot_task(x_k[10], x_k[11], x_k[12], x_k[13])
        self.set_root_task(x_k[14], x_k[15])
        self.set_com_task(x_k[16], x_k[17])
