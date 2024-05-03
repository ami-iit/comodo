import numpy as np


class MPCParameterTuning:
    def __init__(self) -> None:
        self.com_weight = np.asarray([2, 2, 80])
        self.contact_position_weight = 125
        self.force_rate_change_weight = np.asarray([130.0, 20.0, 10.0])
        self.angular_momentum_weight = 125
        self.contact_force_symmetry_weight = 35.0

    def set_parameters(
        self,
        com_weight,
        contac_position,
        force_rate_change,
        angular_mom_weight,
        contact_force_symmetry_weight,
    ):
        # Forcing the x and y component to be one equal to the other
        self.com_weight = np.asanyarray([com_weight[0], com_weight[0], com_weight[1]])
        self.contact_position_weight = contac_position
        self.force_rate_change_weight = force_rate_change
        self.angular_momentum_weight = angular_mom_weight
        self.contact_force_symmetry_weight = contact_force_symmetry_weight

    def set_from_xk(self, x_k):
        self.set_parameters(
            com_weight=x_k[:2],
            contac_position=x_k[2],
            force_rate_change=x_k[3:6],
            angular_mom_weight=x_k[6],
            contact_force_symmetry_weight=x_k[7],
        )
