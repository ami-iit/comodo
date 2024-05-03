import numpy as np

class HippoptWalkingParameterTuning:
    
    def __init__(self) -> None:
        self.step_length = 0.6
        self.time_step = 0.1
        self.horizon_length = 30
    
    def set_parameters(
        self,
        step_length, 
        time_step, 
        horizon_length
    ):
        self.step_length = step_length
        self.time_step = time_step
        self.horizon_length = horizon_length
