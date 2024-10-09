from __future__ import annotations

import logging
import math
import pathlib
from enum import Enum
from typing import Optional, Union

import jax.numpy as jnp
import jaxsim
import jaxsim.api as js
import numpy as np
import numpy.typing as npt
from jaxsim import VelRepr, integrators
from jaxsim.mujoco import MujocoVideoRecorder
from jaxsim.mujoco.loaders import UrdfToMjcf
from jaxsim.mujoco.model import MujocoModelHelper
from jaxsim.mujoco.visualizer import MujocoVisualizer
from jaxsim.rbda.contacts.relaxed_rigid import (
    RelaxedRigidContacts,
    RelaxedRigidContactsParams,
)
from jaxsim.rbda.contacts.rigid import RigidContacts, RigidContactsParams
from jaxsim.rbda.contacts.visco_elastic import ViscoElasticContacts

from comodo.abstractClasses.simulator import Simulator

# === Logger setup ===
logger = logging.getLogger("JaxsimSimulator")
logger.setLevel(logging.DEBUG)
# Remove default handlers if any
if logger.hasHandlers():
    logger.handlers.clear()
logger.propagate = False
# Console handler
console_handler = logging.StreamHandler()
formatter = logging.Formatter(
    "[%(asctime)-19s.%(msecs)03d] [%(levelname)-8s] [TID %(thread)-5d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

class ContactModelEnum(Enum):
    RIGID = "rigid"
    RELAXED_RIGID = "relaxed_rigid"
    VISCO_ELASTIC = "visco_elastic"
    ELASTIC = "elastic"

class JaxsimSimulator(Simulator):
    def __init__(self) -> None:
        # Flag to check if the simulator is initialized
        self._is_initialized: bool = False

        # Simulation data
        self._data: Optional[js.data.JaxSimModelData] = None

        # Simulation model
        self._model: Optional[js.model.JaxSimModel] = None

        # Time step for the simulation
        self._dt: Optional[float] = None

        # Joint torques
        self._tau: Optional[npt.ArrayLike] = None

        # Link contact forces
        self._link_contact_forces: Optional[npt.ArrayLike] = None

        # Contact model to use
        self._contact_model_type: ContactModelEnum = ContactModelEnum.RELAXED_RIGID

        # Index of the left foot link
        self._left_foot_link_idx: Optional[int] = None

        # Index of the right foot link
        self._right_foot_link_idx: Optional[int] = None

        # Index of the left foot sole frame
        self._left_footsole_frame_idx: Optional[int] = None

        # Index of the right foot sole frame
        self._right_footsole_frame_idx: Optional[int] = None

        # ==== Visualization attributes ====
        # Mode for visualization ('record', 'interactive', or None)
        self._visualization_mode: Optional[str] = None

        # Visualization object
        self._viz: Optional[MujocoVisualizer] = None

        # Recorder for video recording
        self._recorder: Optional[MujocoVideoRecorder] = None

        # Frames per second for visualization
        self._viz_fps: int = 10

        # Last rendered time in nanoseconds
        self._last_rendered_t_ns: float = 0.0

        # Optional Mujoco model helper
        self._mj_model_helper: Optional[MujocoModelHelper] = None

    def load_model(
        self,
        robot_model,
        s=None,
        xyz_rpy: npt.ArrayLike = None,
        terrain_params=None,
        visualization_mode=None,
    ) -> None:
        model = js.model.JaxSimModel.build_from_model_description(
            model_description=robot_model.urdf_string,
            model_name=robot_model.robot_name,
            # contact_model=RigidContacts(
            #     parameters=RigidContactsParams(mu=0.5, K=1.0e4, D=1.0e2)
            # ),
            # contact_model=RelaxedRigidContacts(
            #     parameters=RelaxedRigidContactsParams(
            #         mu=0.8,
            #         time_constant=0.005,
            #     )
            # ),
            contact_model=ViscoElasticContacts(),
        )
        model = js.model.reduce(
            model=model,
            considered_joints=robot_model.joint_name_list,
        )

        self._data = js.data.JaxSimModelData.build(
            model=model,
            velocity_representation=VelRepr.Mixed,
            base_position=jnp.array(xyz_rpy[:3]),
            base_quaternion=jnp.array(self.RPY_to_quat(*xyz_rpy[3:])),
            joint_positions=jnp.array(s),
            contacts_params=js.contact.estimate_good_soft_contacts_parameters(
                model=model,
                number_of_active_collidable_points_steady_state=16,
                max_penetration=0.002,
                damping_ratio=1.0,
                static_friction_coefficient=1.0,
            ),
        )

        # Un comment to use non visco-elastic models
        # self.integrator = integrators.fixed_step.RungeKutta4.build(
        #     dynamics=js.ode.wrap_system_dynamics_for_integration(
        #         model=model,
        #         data=self.data,
        #         system_dynamics=js.ode.system_dynamics,
        #     ),
        # )

        # self.integrator_state = self.integrator.init(
        #     x0=self.data.state, t0=0, dt=self.dt
        # )

        self._model = model

        # TODO: expose these names as parameters
        self._left_foot_link_idx = js.link.name_to_idx(
            model=self._model, link_name="l_ankle_2"
        )
        self._right_foot_link_idx = js.link.name_to_idx(
            model=self._model, link_name="r_ankle_2"
        )
        self._left_footsole_frame_idx = js.frame.name_to_idx(
            model=self._model, frame_name="l_sole"
        )
        self._right_footsole_frame_idx = js.frame.name_to_idx(
            model=self._model, frame_name="r_sole"
        )

        logging.info(f"Left foot link index: {self._left_foot_link_idx}")
        logging.info(f"Right foot link index: {self._right_foot_link_idx}")
        logging.info(f"Left foot sole frame index: {self._left_footsole_frame_idx}")
        logging.info(f"Right foot sole frame index: {self._right_footsole_frame_idx}")

        self._visualization_mode = visualization_mode

        if self._visualization_mode is not None:
            if self._visualization_mode not in ["record", "interactive"]:
                raise ValueError(
                    f"Invalid visualization mode: {self._visualization_mode}. "
                    f"Valid options are: 'record', 'interactive'"
                )

            mjcf_string, assets = UrdfToMjcf.convert(
                urdf=self._model.built_from,
            )

            self._mj_model_helper = MujocoModelHelper.build_from_xml(
                mjcf_description=mjcf_string, assets=assets
            )

            self._recorder = jaxsim.mujoco.MujocoVideoRecorder(
                model=self._mj_model_helper.model,
                data=self._mj_model_helper.data,
                fps=30,
            )
            logging.warning("recorder initialized")

    def get_feet_wrench(self) -> npt.ArrayLike:
        wrenches = self.get_link_contact_forces()

        left_foot = np.array(wrenches[self._left_foot_link_idx])
        right_foot = np.array(wrenches[self._right_foot_link_idx])
        return left_foot, right_foot

    def set_input(self, input: npt.ArrayLike) -> None:
        self._tau = jnp.array(input)

    def step(self, torques: np.ndarray = None, n_step: int = 1) -> None:
        if torques is None:
            torques = np.zeros(20)

        for _ in range(n_step):
            # Comment this to use non visco-elastic models
            self._data, _ = jaxsim.rbda.contacts.visco_elastic.step(
                model=self._model,
                data=self._data,
                dt=self._dt,
                joint_force_references=torques,
                link_forces=None,
            )

            # Uncomment this to use non visco-elastic models
            # self.data, self.integrator_state = js.model.step(
            #     model=self.model,
            #     data=self.data,
            #     dt=self.dt,
            #     integrator=self.integrator,
            #     integrator_state=self.integrator_state,
            #     joint_forces=torques,
            #     link_forces=None,  # f
            # )

            current_time_ns = np.array(object=self._data.time_ns).astype(int)

            match self._visualization_mode:
                case "record":
                    if current_time_ns - self._last_rendered_t_ns >= int(
                        1e9 / self._recorder.fps
                    ):
                        self.record_frame()
                        self._last_rendered_t_ns = current_time_ns

                case "interactive":
                    if current_time_ns - self._last_rendered_t_ns >= int(
                        1e9 / self._viz_fps
                    ):
                        self.render()
                        self._last_rendered_t_ns = current_time_ns
                case None:
                    pass

        # self.link_contact_forces = js.model.link_contact_forces(
        #     model=self.model,
        #     data=self.data,
        #     joint_force_references=torques,
        # )

    def get_base(self) -> npt.ArrayLike:
        return np.array(self._data.base_transform())

    def get_base_velocity(self) -> npt.ArrayLike:
        return np.array(self._data.base_velocity())

    def get_simulation_time(self) -> float:
        return self._data.time()

    def reset_simulation_time(self) -> None:
        self._data = self._data.replace(time_ns=jnp.array(0, dtype=jnp.uint64))
        assert self._data.time_ns == 0

    def get_state(self) -> Union[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:
        s = np.array(self._data.joint_positions())
        s_dot = np.array(self._data.joint_velocities())
        tau = np.array(self._tau)

        return s, s_dot, tau

    def get_link_contact_forces(self) -> npt.ArrayLike:
        return self._link_contact_forces

    def total_mass(self) -> float:
        return js.model.total_mass(self._model)

    def get_feet_positions(self) -> tuple[npt.ArrayLike, npt.ArrayLike]:
        W_p_lf = js.frame.transform(
            model=self._model,
            data=self._data,
            frame_index=self._left_footsole_frame_idx,
        )[0:3, 3]
        W_p_rf = js.frame.transform(
            model=self._model,
            data=self._data,
            frame_index=self._right_footsole_frame_idx,
        )[0:3, 3]

        return (W_p_lf, W_p_rf)

    def get_com_position(self) -> npt.ArrayLike:
        return js.com.com_position(self._model, self._data)

    def close(self) -> None:
        pass

    def RPY_to_quat(self, roll, pitch, yaw):
        cr = math.cos(roll / 2)
        cp = math.cos(pitch / 2)
        cy = math.cos(yaw / 2)
        sr = math.sin(roll / 2)
        sp = math.sin(pitch / 2)
        sy = math.sin(yaw / 2)

        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy

        return [qw, qx, qy, qz]

    def render(self):
        if not self._viz:
            mjcf_string, assets = UrdfToMjcf.convert(
                urdf=self._model.built_from,
            )

            self._mj_model_helper = MujocoModelHelper.build_from_xml(
                mjcf_description=mjcf_string, assets=assets
            )

            self._viz = MujocoVisualizer(
                model=self._mj_model_helper.model, data=self._mj_model_helper.data
            )
            self._handle = self._viz.open_viewer()

        self._mj_model_helper.set_base_position(
            position=self._data.base_position(),
        )
        self._mj_model_helper.set_base_orientation(
            orientation=self._data.base_orientation(),
        )
        self._mj_model_helper.set_joint_positions(
            positions=self._data.joint_positions(),
            joint_names=self._model.joint_names(),
        )
        self._viz.sync(viewer=self._handle)

    def record_frame(self):
        self._mj_model_helper.set_base_position(
            position=self._data.base_position(),
        )
        self._mj_model_helper.set_base_orientation(
            orientation=self._data.base_orientation(),
        )
        self._mj_model_helper.set_joint_positions(
            positions=self._data.joint_positions(),
            joint_names=self._model.joint_names(),
        )

        self._recorder.record_frame()

    def save_video(self, file_path: str | pathlib.Path):
        self._recorder.write_video(path=file_path)

    def set_terrain_parameters(self, terrain_params: npt.ArrayLike) -> None:
        terrain_params_dict = dict(
            zip(["max_penetration", "damping_ratio", "mu"], terrain_params)
        )

        logging.warning(f"Setting terrain parameters: {terrain_params_dict}")

        contact_params = js.contact.estimate_good_soft_contacts_parameters(
            model=self._model,
            number_of_active_collidable_points_steady_state=16,
            max_penetration=terrain_params_dict["max_penetration"],
            damping_ratio=terrain_params_dict["damping_ratio"],
            static_friction_coefficient=terrain_params_dict["mu"],
        )

        self._data = self._data.replace(contacts_params=contact_params)
