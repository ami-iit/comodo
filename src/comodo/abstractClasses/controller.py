from abc import ABC, abstractmethod


class Controller(ABC):
    def __init__(self, frequency, robot_model) -> None:
        self.torque = None
        self.frequency = frequency
        self.robot_model = robot_model
        super().__init__()

    def set_state(self, s, s_dot, t):
        self.s = s
        self.s_dot = s_dot
        self.t = t

    def get_torque(self):
        return self.torque

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def get_fitness_parameters(self):
        pass
