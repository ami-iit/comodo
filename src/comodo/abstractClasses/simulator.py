from abc import ABC, abstractmethod

class Simulator(ABC): 
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def load_model(self, robot_model,  s, xyz_rpy): 
        pass 
    
    @abstractmethod
    def set_input(self, input): 
        pass

    @abstractmethod
    def step(self, n_step=1): 
        pass 
    
    @abstractmethod
    def get_state(self): 
        pass 

    @abstractmethod
    def close(self):
        pass
