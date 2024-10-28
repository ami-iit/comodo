from abc import ABC, abstractmethod
from typing import List
from comodo.mujocoSimulator.mjcontactinfo import MjContactInfo
import types
import logging
import mujoco

class Callback(ABC):
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs
        self.simulator = None

    def set_simulator(self, simulator):
        self.simulator = simulator

    @abstractmethod
    def on_simulation_start(self) -> None:
        pass

    @abstractmethod
    def on_simulation_step(self, t: float, data: mujoco.MjData, opts: dict = None) -> None:
        pass

    @abstractmethod
    def on_simulation_end(self) -> None:
        pass


class EarlyStoppingCallback(Callback):
    def __init__(self, condition, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.score = 0
        self.condition = condition

    def on_simulation_start(self) -> None:
        pass

    def on_simulation_step(self, t: float, iter: int, data: mujoco.MjData, opts: dict = None) -> None:
        if self.condition(t, iter, data, opts, *self.args, **self.kwargs):
            if self.simulator is not None:
                self.simulator.should_stop = True

    def on_simulation_end(self) -> None:
        pass
            


class ScoreCallback(Callback):
    def __init__(self, score_function, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.score = 0
        self.history = []
        self.score_function = score_function

    def on_simulation_start(self) -> None:
        self.score = 0
        self.history = []

    def on_simulation_step(self, t: float, iter: int, data: mujoco.MjData, opts: dict = None) -> None:
        score = self.score_function(t, data, *self.args, **self.kwargs)
        self.score += score
        self.history.append(score)

    def on_simulation_end(self) -> None:
        pass


class TrackerCallback(Callback):
    def __init__(self, tracked_variables: list, print_values: bool | list = False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.score = 0
        self.tracked_variables = tracked_variables

        if isinstance(print_values, list):
            self.print_values = print_values
        elif isinstance(print_values, bool):
            self.print_values = [print_values for _ in range(len(tracked_variables))]
        else:
            raise ValueError(f"print_values should be a boolean or a list of booleans for masking, not {type(print_values)}")

        self.t = []
        self.vals = {}

    def on_simulation_start(self) -> None:
        pass

    def on_simulation_step(self, t: float, iter: int, data: mujoco.MjData, opts: dict = None) -> None:
        self.t.append(t)
        for var in self.tracked_variables:
            if isinstance(var, str):
                try:
                    val = eval(f"data.{var}")
                    if not var in self.vals:
                        self.vals[var] = []
                    self.vals[var].append(val)
                    if self.print_values[self.tracked_variables.index(var)]:
                        print(f"{var}: {val}")
                except:
                    print(f"Error: {self.tracked_variables} not found in data")
            elif isinstance(var, types.FunctionType):
                val = var(t, iter, data, opts, *self.args, **self.kwargs)
                if not var.__name__ in self.vals:
                    self.vals[var.__name__] = []
                if not isinstance(val, (int, float)):
                    return
                self.vals[var.__name__].append(val)
                if self.print_values[self.tracked_variables.index(var)]:
                    print(f"{var.__name__}: {val}")

    def on_simulation_end(self) -> None:
        pass

    def get_tracked_values(self):
        return self.t, self.vals
    


class ContactCallback(Callback):
    def __init__(self, tracked_bodies: List[str] = [], logger: logging.Logger = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tracked_bodies = tracked_bodies
        self.logger = logger
        self.last_contact = None

    def on_simulation_start(self) -> None:
        pass

    def on_simulation_step(self, t: float, iter: int, data: mujoco.MjData, opts: dict = None) -> None:
        if opts.get("contact", None) is not None:
            contact_info = opts["contact"]
            assert isinstance(contact_info, MjContactInfo), "Contact info is not an instance of MjContactInfo"
            if contact_info.is_none():
                return
            self.last_contact = contact_info
            if self.logger is not None:
                self.logger.debug(f"Contact detected at t={t}: {contact_info}")
            else:
                pass
                #print(f"Contact detected at t={t}: {contact_info}")


    def on_simulation_end(self) -> None:
        pass
