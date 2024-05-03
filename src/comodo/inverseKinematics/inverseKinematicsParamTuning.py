import numpy as np


class InverseKinematicsParamTuning:
    def __init__(self) -> None:
        self.zmp_gain = [0.4, 0.4]
        self.com_gain = [4.3, 4.3]
        self.foot_linear = 1.0
        self.foot_angular = 6.59
        self.com_linear = 2.59
        self.chest_angular = 1.4
        self.root_linear = 2.8
        self.kp_joint_tracking = 5.0
        self.weigth_joint = [
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
        ]

    def set_parameters(
        self,
        zmp_gain,
        com_gain,
        foot_linear,
        foot_angular,
        com_linear,
        chest_angular,
        root_linear,
    ):
        self.zmp_gain = [zmp_gain, zmp_gain]
        self.com_gain = [com_gain, com_gain]
        self.foot_linear = foot_linear
        self.foot_angular = foot_angular
        self.com_linear = com_linear
        self.chest_angular = chest_angular
        self.root_linear = root_linear

    def set_from_xk(self, x_k):
        # x_k[zmp_gain_1, zmp_gain_2, com_gain_1,com_gain_2, foot_linear, foot_angular, com_linear, chest_angular, root_linear]
        # dimension 9
        self.set_parameters(
            zmp_gain=x_k[0],
            com_gain=x_k[1],
            foot_linear=x_k[2],
            foot_angular=x_k[3],
            com_linear=x_k[4],
            chest_angular=x_k[5],
            root_linear=x_k[6],
        )
