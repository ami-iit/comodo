import numpy as np


class InverseKinematicParamTuning:
    def __init__(self) -> None:
        self.zmp_gain = [0.2, 0.2]
        self.com_gain = [2.0, 2.0]
        self.foot_linear = 3.0
        self.foot_angular = 9.0
        self.com_linear = 2.0
        self.chest_angular = 1.0
        self.root_linear = 2.0
        self.Kp_joint_tracking = 5.0
        self.weigth_joint = 1.0

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
        self.zmp_gain = zmp_gain
        self.com_gain = com_gain
        self.foot_linear = foot_linear
        self.foot_angular = foot_angular
        self.com_linear = com_linear
        self.chest_angular = chest_angular
        self.root_linear = root_linear

    def set_from_xk(self, x_k):
        self.set_parameters(
            zmp_gain=x_k[:2],
            com_gain=x_k[2:4],
            foot_linear=x_k[4],
            foot_angular=x_k[5],
            com_linear=x_k[6],
            chest_angular=x_k[7],
            root_linear=x_k[8],
        )
