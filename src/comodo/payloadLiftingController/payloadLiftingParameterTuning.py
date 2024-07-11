import numpy as np


class PayloadLiftingControllerParameters:
    def __init__(self, joint_active) -> None:
        kp_param_temp = np.array([130,80,20,60,130,80,20,60,150,20,20,180,140,20,150,20,20,180,140,20])* 2.5
        self.joints_Kp_parameters = []
        for idx,item in enumerate(joint_active):
            if(item):
                self.joints_Kp_parameters.append(kp_param_temp[idx])            
        self.momentum_kp = np.array([1.5, 1.5, 2, 0.025, 0.025, 0.025])
        self.momentum_ki = np.array([50, 50, 100, 0, 0, 0])

    def set_parameters(
        self,
        momentum_kp,
        momentum_ki
    ):
        # Forcing the x and y component to be one equal to the other
        self.momentum_kp = np.asanyarray([momentum_kp[0], momentum_kp[0], momentum_kp[1],momentum_kp[2],momentum_kp[2],momentum_kp[3]])
        self.momentum_ki = np.asanyarray([momentum_ki[0], momentum_ki[0], momentum_ki[1],0.0,0.0,0.0])
 

    def set_from_xk(self, x_k):
        self.set_parameters(
            momentum_kp=x_k[:4],
            momentum_ki=x_k[4:],
         )
