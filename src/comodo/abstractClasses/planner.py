from abc import ABC, abstractmethod

class Planner(ABC): 
    def __init__(self, robot_model):
        self.robot_model = robot_model
        super().__init__()
    
    @abstractmethod
    def plan_trajectory(self): 
        pass 
