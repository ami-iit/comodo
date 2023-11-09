import numpy as np


class MPCParameterTuning:
    def __init__(self) -> None:
        self.com_weight = np.asarray([100, 100, 1000])
        self.contact_position_weight = 1e3
        self.force_rate_change_weight = np.asarray([10.0, 10.0, 10.0])
        self.angular_momentum_weight = 1e5
        self.contact_force_symmetry_weight = 1.0

    def set_parameters(
        self,
        com_weight,
        contac_position,
        force_rate_change,
        angular_mom_weight,
        contact_force_symmetry_weight,
    ):
        self.com_weight = com_weight
        self.contact_position_weight = contac_position
        self.force_rate_change_weight = force_rate_change
        self.angular_momentum_weight = angular_mom_weight

    def set_from_xk(self, x_k):
        self.set_parameters(x_k[:3], x_k[3], x_k[4:7], x_k[7], x_k[8])
