from abc import ABC, abstractmethod
import mujoco

class Callback(ABC):
    def set_simulator(self, simulator):
        self.simulator = simulator

    @abstractmethod
    def on_simulation_start(self) -> None:
        pass

    @abstractmethod
    def on_simulation_step(self, t: float, data: mujoco.MjData) -> None:
        pass

    @abstractmethod
    def on_simulation_end(self) -> None:
        pass


class EarlyStoppingCallback(Callback):
    def __init__(self, condition) -> None:
        self.condition = condition

    def on_simulation_start(self) -> None:
        pass

    def on_simulation_step(self, t: float, data: mujoco.MjData) -> None:
        if self.condition(t, data):
            if self.simulator is not None:
                self.simulator.should_stop = True

    def on_simulation_end(self) -> None:
        pass
            


class ScoreCallback(Callback):
    def __init__(self, score_function) -> None:
        self.score = 0
        self.history = []
        self.score_function = score_function

    def on_simulation_start(self) -> None:
        self.score = 0
        self.history = []

    def on_simulation_step(self, t: float, data: mujoco.MjData) -> None:
        score = self.score_function(t, data)
        self.score += score
        self.history.append(score)

    def on_simulation_end(self) -> None:
        print(f"Final score: {self.score}")


class TrackerCallback(Callback):
    def __init__(self, tracked_variables: list, print_values: bool = False) -> None:
        self.tracked_variables = tracked_variables
        self.print_values = print_values

        self.t = []
        self.vals = {var: [] for var in tracked_variables}

    def on_simulation_start(self) -> None:
        pass

    def on_simulation_step(self, t: float, data: mujoco.MjData) -> None:
        self.t.append(t)
        for var in self.tracked_variables:
            try:
                val = eval(f"data.{var}")
                self.vals[var].append(val)
                if self.print_values:
                    print(f"{self.tracked_variables}: {val}")
            except:
                print(f"Error: {self.tracked_variables} not found in data")

    def on_simulation_end(self) -> None:
        pass

    def get_tracked_values(self):
        return self.t, self.vals