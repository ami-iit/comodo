from __future__ import annotations

import logging
import math
import pathlib
from enum import Enum
from typing import Any

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
from jaxsim.rbda.contacts.common import ContactsParams
from jaxsim.rbda.contacts.soft import SoftContacts
from jaxsim.rbda.contacts.relaxed_rigid import (
    RelaxedRigidContacts,
    RelaxedRigidContactsParams,
)
from jaxsim.rbda.contacts.rigid import RigidContacts, RigidContactsParams
from jaxsim.rbda.contacts.visco_elastic import ViscoElasticContacts

from comodo.abstractClasses.simulator import Simulator
from comodo.robotModel.robotModel import RobotModel

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
    SOFT = "soft"


class JaxsimSimulator(Simulator):
    def __init__(self) -> None:
        # Flag to check if the simulator is initialized
        self._is_initialized: bool = False

        # Simulation data
        self._data: js.data.JaxSimModelData | None = None

        # Simulation model
        self._model: js.model.JaxSimModel | None = None

        # Integrator (not used for visco-elastic contacts)
        self._integrator: integrators.common.Integrator | None = None

        # Integrator state for the simulation (not used for visco-elastic contacts)
        self._integrator_state: dict[str, Any] | None = None

        # Time step for the simulation
        self._dt: float | None = None

        # Joint torques
        self._tau: npt.ArrayLike | None = None

        # Link contact forces
        self._link_contact_forces: npt.ArrayLike | None = None

        # Contact model to use
        self._contact_model_type: ContactModelEnum = ContactModelEnum.RELAXED_RIGID

        # Index of the left foot link
        self._left_foot_link_idx: int | None = None

        # Index of the right foot link
        self._right_foot_link_idx: int | None = None

        # Index of the left foot sole frame
        self._left_footsole_frame_idx: int | None = None

        # Index of the right foot sole frame
        self._right_footsole_frame_idx: int | None = None

        # ==== Visualization attributes ====
        # Mode for visualization ('record', 'interactive', or None)
        self._visualization_mode: str | None = None

        # Visualization object
        self._viz: MujocoVisualizer | None = None

        # Recorder for video recording
        self._recorder: MujocoVideoRecorder | None = None

        # Frames per second for visualization
        self._viz_fps: int = 10

        # Last rendered time in nanoseconds
        self._last_rendered_t_ns: float = 0.0

        # Optional Mujoco model helper
        self._mj_model_helper: MujocoModelHelper | None = None

    # ==== Simulator interface methods ====

    def load_model(
        self,
        robot_model: RobotModel,
        *,
        dt: float = 0.001,
        xyz_rpy: npt.ArrayLike = np.zeros(6),
        contact_model_type: ContactModelEnum = ContactModelEnum.RELAXED_RIGID,
        s: npt.ArrayLike | None = None,
        contact_params: ContactsParams | None = None,
        left_foot_link_name: str | None = "l_ankle_2",
        right_foot_link_name: str | None = "r_ankle_2",
        left_foot_sole_frame_name: str | None = "l_sole",
        right_foot_sole_frame_name: str | None = "r_sole",
        visualization_mode: str | None = None,
    ) -> None:
        # ==== Initialize simulator model and data ====

        self._dt = dt
        self._contact_model_type = contact_model_type

        match contact_model_type:
            case ContactModelEnum.RIGID:
                contact_model = RigidContacts()
            case ContactModelEnum.RELAXED_RIGID:
                contact_model = RelaxedRigidContacts()
            case ContactModelEnum.VISCO_ELASTIC:
                contact_model = ViscoElasticContacts()
            case ContactModelEnum.SOFT:
                contact_model = SoftContacts()
            case _:
                raise ValueError(f"Invalid contact model type: {contact_model_type}")

        model = js.model.JaxSimModel.build_from_model_description(
            model_description=robot_model.urdf_string,
            model_name=robot_model.robot_name,
            contact_model=contact_model,
        )

        model = js.model.reduce(
            model=model,
            considered_joints=robot_model.joint_name_list,
        )

        self._model = model

        if contact_params is None:
            match contact_model_type:
                case ContactModelEnum.RIGID:
                    contact_params = RigidContactsParams.build(mu=0.5, K=1.0e4, D=1.0e2)
                case ContactModelEnum.RELAXED_RIGID:
                    contact_params = RelaxedRigidContactsParams.build(
                        mu=0.8, time_constant=0.005
                    )
                case ContactModelEnum.VISCO_ELASTIC | ContactModelEnum.SOFT:
                    contact_params = js.contact.estimate_good_soft_contacts_parameters(
                        model=self._model,
                        number_of_active_collidable_points_steady_state=16,
                        max_penetration=0.002,
                        damping_ratio=1.0,
                        static_friction_coefficient=1.0,
                    )
                case _:
                    raise ValueError(
                        f"Invalid contact model type: {contact_model_type}"
                    )

        s = s or np.zeros(self._model.dofs())

        self._data = js.data.JaxSimModelData.build(
            model=self._model,
            velocity_representation=VelRepr.Mixed,
            base_position=jnp.array(xyz_rpy[:3]),
            base_quaternion=jnp.array(self._RPY_to_quat(*xyz_rpy[3:])),
            joint_positions=jnp.array(s),
            contacts_params=contact_params,
        )

        if contact_model_type is not ContactModelEnum.VISCO_ELASTIC:
            self._integrator = integrators.fixed_step.RungeKutta4.build(
                dynamics=js.ode.wrap_system_dynamics_for_integration(
                    model=self._model,
                    data=self._data,
                    system_dynamics=js.ode.system_dynamics,
                )
            )

            self._integrator_state = self._integrator.init(
                x0=self._data.state, t0=0, dt=self._dt
            )

        # TODO: expose these names as parameters
        self._left_foot_link_idx = js.link.name_to_idx(
            model=self._model, link_name=left_foot_link_name
        )
        self._right_foot_link_idx = js.link.name_to_idx(
            model=self._model, link_name=right_foot_link_name
        )
        self._left_footsole_frame_idx = js.frame.name_to_idx(
            model=self._model, frame_name=left_foot_sole_frame_name
        )
        self._right_footsole_frame_idx = js.frame.name_to_idx(
            model=self._model, frame_name=right_foot_sole_frame_name
        )

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

            self._recorder = MujocoVideoRecorder(
                model=self._mj_model_helper.model,
                data=self._mj_model_helper.data,
                fps=30,
            )

        self._is_initialized = True

    def set_input(self, input: npt.ArrayLike) -> None:
        self._tau = jnp.array(input)

    def step(self, n_step: int = 1) -> None:
        assert (
            self._is_initialized
        ), "Simulator is not initialized, call load_model first."

        for _ in range(n_step):
            if self._contact_model_type is ContactModelEnum.VISCO_ELASTIC:
                self._data, _ = jaxsim.rbda.contacts.visco_elastic.step(
                    model=self._model,
                    data=self._data,
                    dt=self._dt,
                    joint_force_references=self._tau,
                    link_forces=None,
                )

            else:
                # All other contact models
                self._data, self._integrator_state = js.model.step(
                    model=self._model,
                    data=self._data,
                    dt=self.dt,
                    integrator=self._integrator,
                    integrator_state=self._integrator_state,
                    joint_forces=self._tau,
                    link_forces=None,
                )

                self._link_contact_forces = js.model.link_contact_forces(
                    model=self._model,
                    data=self._data,
                    joint_force_references=self._tau,
                )

            current_time_ns = self._data.time_ns.astype(int)

            match self._visualization_mode:
                case "record":
                    if current_time_ns - self._last_rendered_t_ns >= int(
                        1e9 / self._recorder.fps
                    ):
                        self._record_frame()
                        self._last_rendered_t_ns = current_time_ns

                case "interactive":
                    if current_time_ns - self._last_rendered_t_ns >= int(
                        1e9 / self._viz_fps
                    ):
                        self._render()
                        self._last_rendered_t_ns = current_time_ns
                case None:
                    pass

    def get_state(self) -> tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:
        assert (
            self._is_initialized
        ), "Simulator is not initialized, call load_model first."

        s = np.array(self._data.joint_positions())
        s_dot = np.array(self._data.joint_velocities())
        tau = np.array(self._tau)

        return s, s_dot, tau

    def close(self) -> None:
        pass

    # ==== Properties and public methods ====

    @property
    def feet_wrench(self) -> npt.ArrayLike:
        assert (
            self._is_initialized
        ), "Simulator is not initialized, call load_model first."
        wrenches = self.link_contact_forces

        left_foot = np.array(wrenches[self._left_foot_link_idx])
        right_foot = np.array(wrenches[self._right_foot_link_idx])
        return left_foot, right_foot

    @property
    def base_transform(self) -> npt.ArrayLike:
        assert (
            self._is_initialized
        ), "Simulator is not initialized, call load_model first."
        return np.array(self._data.base_transform())

    @property
    def base_velocity(self) -> npt.ArrayLike:
        assert (
            self._is_initialized
        ), "Simulator is not initialized, call load_model first."
        return np.array(self._data.base_velocity())

    @property
    def simulation_time(self) -> float:
        assert (
            self._is_initialized
        ), "Simulator is not initialized, call load_model first."
        return self._data.time()

    @property
    def link_contact_forces(self) -> npt.ArrayLike:
        assert (
            self._is_initialized
        ), "Simulator is not initialized, call load_model first."
        if self._contact_model_type is ContactModelEnum.VISCO_ELASTIC:
            raise ValueError(
                "Link contact forces are only available for non visco-elastic contact models."
            )
        return self._link_contact_forces

    @property
    def total_mass(self) -> float:
        assert (
            self._is_initialized
        ), "Simulator is not initialized, call load_model first."
        return js.model.total_mass(self._model)

    @property
    def feet_positions(self) -> tuple[npt.ArrayLike, npt.ArrayLike]:
        assert (
            self._is_initialized
        ), "Simulator is not initialized, call load_model first."
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

    @property
    def com_position(self) -> npt.ArrayLike:
        assert (
            self._is_initialized
        ), "Simulator is not initialized, call load_model first."
        return js.com.com_position(self._model, self._data)

    def reset_simulation_time(self) -> None:
        assert (
            self._is_initialized
        ), "Simulator is not initialized, call load_model first."
        self._data = self._data.replace(time_ns=jnp.array(0, dtype=jnp.uint64))
        assert self._data.time_ns == 0, "Failed to reset simulation time."

    def update_contact_model_parameters(self, params: ContactsParams) -> None:
        assert (
            self._is_initialized
        ), "Simulator is not initialized, call load_model first."

        self._data = self._data.replace(contacts_params=params)

    # ==== Private methods ====

    def _RPY_to_quat(self, roll, pitch, yaw):
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

    def _render(self):
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

    def _record_frame(self):
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

    def _save_video(self, file_path: str | pathlib.Path):
        self._recorder.write_video(path=file_path)
