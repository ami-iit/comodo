import numpy as np


class PayloadLiftingControllerParameters:
    def __init__(self) -> None:
        self.joints_Kp_parameters = (
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
            * 2.5
        )
        self.CoM_Kp = np.array([1.5, 1.5, 2, 0.025, 0.025, 0.025])
        self.CoM_Ki = np.array([50, 50, 100, 0, 0, 0])

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
