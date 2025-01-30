from __future__ import annotations

import enum
import logging
import math
import pathlib
from typing import Any

import jax.numpy as jnp
import jaxsim
import jaxsim.api as js
import jaxsim.rbda.contacts
import numpy as np
import numpy.typing as npt
from jaxsim import VelRepr
from jaxsim.mujoco import MujocoVideoRecorder
from jaxsim.mujoco.loaders import UrdfToMjcf
from jaxsim.mujoco.model import MujocoModelHelper
from jaxsim.mujoco.visualizer import MujocoVisualizer

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


class JaxsimContactModelEnum(enum.IntEnum):
    RIGID = enum.auto()
    RELAXED_RIGID = enum.auto()
    VISCO_ELASTIC = enum.auto()
    SOFT = enum.auto()


class JaxsimSimulator(Simulator):

    def __init__(
        self,
        dt: float = 0.001,
        contact_model_type: JaxsimContactModelEnum = JaxsimContactModelEnum.RELAXED_RIGID,
    ) -> None:

        super().__init__()

        # Flag to check if the simulator is initialized
        self._is_initialized: bool = False

        # Simulation data
        self._data: js.data.JaxSimModelData | None = None

        # Simulation model
        self._model: js.model.JaxSimModel | None = None

        # Step aux dict.
        # This is used only for variable-step integrators.
        self._step_aux_dict: dict[str, Any] = {}

        # Time step for the simulation
        self._dt: float = dt

        # Simulation time (seconds)
        self._t: float = 0.0

        # Joint torques
        self._tau: npt.NDArray | None = None

        # Link contact forces
        self._link_contact_forces: npt.NDArray | None = None

        # Contact model to use
        self._contact_model_type: JaxsimContactModelEnum = contact_model_type

        # Index of the left foot link
        self._left_foot_link_idx: int | None = None

        # Index of the right foot link
        self._right_foot_link_idx: int | None = None

        # Index of the left foot sole frame
        self._left_footsole_frame_idx: int | None = None

        # Index of the right foot sole frame
        self._right_footsole_frame_idx: int | None = None

        # Mapping between user provided joint names and JaxSim joint names
        self._to_js: npt.NDArray | None = None

        # Mapping between JaxSim joint names and user provided joint names
        self._to_user: npt.NDArray | None = None

        # ==== Visualization attributes ====
        # Mode for visualization ('record', 'interactive', or None)
        self._visualization_mode: str | None = None

        # Visualization object
        self._viz: MujocoVisualizer | None = None

        # Recorder for video recording
        self._recorder: MujocoVideoRecorder | None = None

        # Frames per second for visualization
        self._viz_fps: int = 10

        # Last rendered time (seconds)
        self._last_rendered_t: float = 0.0

        # Optional Mujoco model helper
        self._mj_model_helper: MujocoModelHelper | None = None

    # ==== Simulator interface methods ====

    def load_model(
        self,
        robot_model: RobotModel,
        *,
        xyz_rpy: npt.ArrayLike = np.zeros(6),
        s: npt.ArrayLike | None = None,
        contact_params: jaxsim.rbda.contacts.ContactParamsTypes | None = None,
        left_foot_link_name: str | None = "l_ankle_2",
        right_foot_link_name: str | None = "r_ankle_2",
        left_foot_sole_frame_name: str | None = "l_sole",
        right_foot_sole_frame_name: str | None = "r_sole",
        visualization_mode: str | None = None,
    ) -> None:
        """
        Load and initialize the robot model for simulation.

        Args:
            robot_model: The robot model to be loaded.
            dt: The time step for the simulation in seconds.
            xyz_rpy: The initial position and orientation (roll, pitch, yaw) of the robot base.
            contact_model_type: The type of contact model to use.
            s: The initial joint positions.
            contact_params: The parameters for the contact model.
            left_foot_link_name: The name of the left foot link.
            right_foot_link_name: The name of the right foot link.
            left_foot_sole_frame_name: The name of the left foot sole frame.
            right_foot_sole_frame_name: The name of the right foot sole frame.
            visualization_mode: The mode for visualization, either "record" or "interactive".

        Raises:
            ValueError: If an invalid contact model type or visualization mode is provided.
        """
        # ==== Initialize simulator model and data ====

        self._t = 0.0

        match self._contact_model_type:
            case JaxsimContactModelEnum.RIGID:
                contact_model = jaxsim.rbda.contacts.RigidContacts.build()
            case JaxsimContactModelEnum.RELAXED_RIGID:
                contact_model = jaxsim.rbda.contacts.RelaxedRigidContacts.build()
            case JaxsimContactModelEnum.VISCO_ELASTIC:
                contact_model = jaxsim.rbda.contacts.ViscoElasticContacts.build()
            case JaxsimContactModelEnum.SOFT:
                contact_model = jaxsim.rbda.contacts.SoftContacts.build()
            case _:
                raise ValueError(
                    f"Invalid contact model type: {self._contact_model_type}"
                )

        model = js.model.JaxSimModel.build_from_model_description(
            model_description=robot_model.urdf_string,
            model_name=robot_model.robot_name,
            contact_model=contact_model,
            contact_params=contact_params,
            time_step=self._dt,
        )

        if contact_params is None:
            match self._contact_model_type:
                case JaxsimContactModelEnum.RIGID:
                    contact_params = jaxsim.rbda.contacts.RigidContactsParams.build(
                        mu=0.5, K=1.0e4, D=1.0e2
                    )
                case JaxsimContactModelEnum.RELAXED_RIGID:
                    contact_params = (
                        jaxsim.rbda.contacts.RelaxedRigidContactsParams.build()
                    )
                case JaxsimContactModelEnum.VISCO_ELASTIC | JaxsimContactModelEnum.SOFT:
                    contact_params = js.contact.estimate_good_contact_parameters(
                        model=model,
                        number_of_active_collidable_points_steady_state=8,
                        max_penetration=0.002_5,
                        damping_ratio=1.0,
                        static_friction_coefficient=1.0,
                    )
                case _:
                    raise ValueError(
                        f"Invalid contact model type: {self._contact_model_type}"
                    )

        with model.editable(validate=False) as model:
            model.contact_params = contact_params

        self._model = js.model.reduce(
            model=model,
            considered_joints=tuple(robot_model.joint_name_list),
        )

        # Find mapping between user provided joint name list and JaxSim one
        user_joint_names = robot_model.joint_name_list
        js_joint_names = self._model.joint_names()

        get_joint_map = lambda from_, to: np.array(list(map(from_.index, to)))
        self._to_js = get_joint_map(from_=user_joint_names, to=js_joint_names)
        self._to_user = get_joint_map(from_=js_joint_names, to=user_joint_names)

        s = np.zeros(self._model.dofs()) if s is None else np.array(s)[self._to_js]

        self._data = js.data.JaxSimModelData.build(
            model=self._model,
            velocity_representation=VelRepr.Mixed,
            base_position=jnp.array(xyz_rpy[:3]),
            base_quaternion=jnp.array(JaxsimSimulator._RPY_to_quat(*xyz_rpy[3:])),
            joint_positions=jnp.array(s),
        )

        # Initialize tau to zero
        self._tau = np.zeros(self._model.dofs())

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

        # Check that joint list from mujoco and jaxsim have the same elements (just ordered differently)
        assert np.array_equal(
            np.array(js_joint_names)[self._to_user], user_joint_names
        ), "Joint names mismatch"
        assert np.array_equal(
            np.array(user_joint_names)[self._to_js], js_joint_names
        ), "Joint names mismatch"

        # Configure visualization

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
                width=320 * 4,
                height=240 * 4,
            )

        self._is_initialized = True

    def set_input(self, input: npt.ArrayLike) -> None:
        """Set the robot input torques.

        Args:
            input: torques to apply to the robot joints.
        """

        self._tau = jnp.array(input)[self._to_js]

    def step(self, n_step: int = 1, *, dry_run=False) -> None:
        """Step the simulation forward by n_step steps.

        Args:
            n_step: number of steps to simulate.
            dry_run: If True, the simulation will not advance the state of the
                simulator and will not trigger any visualization.
        """

        assert (
            self._is_initialized
        ), "Simulator is not initialized, call load_model first."

        for _ in range(n_step):

            if self._contact_model_type is JaxsimContactModelEnum.VISCO_ELASTIC:

                self._data = jaxsim.rbda.contacts.visco_elastic.step(
                    model=self._model,
                    data=self._data,
                    link_forces=None,
                    joint_force_references=self._tau,
                )

            else:

                # All other contact models
                self._data = js.model.step(
                    model=self._model,
                    data=self._data,
                    joint_force_references=self._tau,
                )

            if not dry_run:
                # Advance simulation time
                self._t += self._dt

                # Render visualization
                match self._visualization_mode:
                    case "record":
                        if self._t - self._last_rendered_t >= (1 / self._recorder.fps):
                            self._record_frame()
                            self._last_rendered_t = self._t

                    case "interactive":
                        if self._t - self._last_rendered_t >= (1 / self._viz_fps):
                            self._render()
                            self._last_rendered_t = self._t
                    case None:
                        pass

        # Update link contact forces
        if not dry_run:

            with self._data.switch_velocity_representation(VelRepr.Mixed):

                self._link_contact_forces, _ = js.contact_model.link_contact_forces(
                    model=self._model,
                    data=self._data,
                    joint_torques=self._tau,
                )

    def get_state(self) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """
        Retrieve the current state of the simulator.

        Returns:
            A tuple containing:
                - s: Joint positions as a numpy array.
                - s_dot: Joint velocities as a numpy array.
                - tau: Joint torques as a numpy array.
        Raises:
            AssertionError: If the simulator is not initialized.
        """

        assert (
            self._is_initialized
        ), "Simulator is not initialized, call load_model first."

        s = np.array(self._data.joint_positions)[self._to_user]
        s_dot = np.array(self._data.joint_velocities)[self._to_user]
        tau = np.array(self._tau)[self._to_user]

        return s, s_dot, tau

    def close(self) -> None:
        """Close the simulator and release any resources. Doing nothing for now."""
        pass

    # ==== Properties and public methods ====

    @property
    def feet_wrench(self) -> tuple[npt.NDArray, npt.NDArray]:

        if self._link_contact_forces is None:
            raise ValueError(
                "Link contact forces are only available after calling the step method."
            )

        wrenches = self.link_contact_forces

        left_foot = np.array(wrenches[self._left_foot_link_idx])
        right_foot = np.array(wrenches[self._right_foot_link_idx])
        return left_foot, right_foot

    @property
    def base_transform(self) -> npt.NDArray:
        assert (
            self._is_initialized
        ), "Simulator is not initialized, call load_model first."
        return np.array(self._data.base_transform)

    @property
    def base_velocity(self) -> npt.NDArray:
        assert (
            self._is_initialized
        ), "Simulator is not initialized, call load_model first."
        with self._data.switch_velocity_representation(VelRepr.Mixed):
            return np.array(self._data.base_velocity())

    @property
    def simulation_time(self) -> float:
        assert (
            self._is_initialized
        ), "Simulator is not initialized, call load_model first."
        return self._t

    @property
    def link_contact_forces(self) -> npt.NDArray:

        if self._link_contact_forces is None:
            raise ValueError(
                "Link contact forces are only available after calling the step method."
            )

        return self._link_contact_forces

    @property
    def total_mass(self) -> float:
        assert (
            self._is_initialized
        ), "Simulator is not initialized, call load_model first."
        return float(js.model.total_mass(self._model))

    @property
    def feet_positions(self) -> tuple[npt.NDArray, npt.NDArray]:
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

        return np.array(W_p_lf), np.array(W_p_rf)

    @property
    def frame_names(self) -> list[str]:
        assert (
            self._is_initialized
        ), "Simulator is not initialized, call load_model first."
        return list(self._model.frame_names())

    @property
    def link_names(self) -> list[str]:
        assert (
            self._is_initialized
        ), "Simulator is not initialized, call load_model first."
        return list(self._model.link_names())

    @property
    def com_position(self) -> npt.NDArray:
        assert (
            self._is_initialized
        ), "Simulator is not initialized, call load_model first."
        return np.array(js.com.com_position(self._model, self._data))

    @property
    def contact_model_type(self) -> JaxsimContactModelEnum:
        return self._contact_model_type

    def reset_simulation_time(self) -> None:
        self._t = 0.0

    def update_contact_model_parameters(
        self, params: jaxsim.rbda.contacts.ContactParamsTypes
    ) -> None:
        assert (
            self._is_initialized
        ), "Simulator is not initialized, call load_model first."

        self._data = self._data.replace(contacts_params=params)

    # ==== Private methods ====

    @staticmethod
    def _RPY_to_quat(roll, pitch, yaw) -> list[float]:
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

    def _render(self) -> None:
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
            position=np.array(self._data.base_position),
        )
        self._mj_model_helper.set_base_orientation(
            orientation=np.array(self._data.base_orientation()),
        )
        self._mj_model_helper.set_joint_positions(
            positions=np.array(self._data.joint_positions),
            joint_names=self._model.joint_names(),
        )
        self._viz.sync(viewer=self._handle)

    def _record_frame(self) -> None:
        self._mj_model_helper.set_base_position(
            position=np.array(self._data.base_position),
        )
        self._mj_model_helper.set_base_orientation(
            orientation=np.array(self._data.base_orientation()),
        )
        self._mj_model_helper.set_joint_positions(
            positions=np.array(self._data.joint_positions),
            joint_names=self._model.joint_names(),
        )

        self._recorder.record_frame()

    def _save_video(self, file_path: str | pathlib.Path) -> None:
        self._recorder.write_video(path=file_path)
