# %% [markdown]
# # Walking in JaxSim
#
# This notebook demonstrates how to use comodo with JaxSim simulator to control the walking motion of a humanoid robot.
#
# Here's a list of acronyms used in the notebook:
# - `lf`, `rf`: left foot, right foot
# - `js`: JaxSim
# - `tsid`: Task Space Inverse Dynamics
# - `mpc`: Model Predictive Control
# - `sfp`: Swing Foot Planner
# - `mj`: Mujoco
# - `s`: joint positions
# - `ds`: joint velocities
# - `τ`: joint torques
# - `b`: base
# - `com`: center of mass
# - `dcom`: center of mass velocity

# %%
# ==== Imports ====
from __future__ import annotations

import datetime
import os
import pathlib
import tempfile
import time
import traceback
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

# Here we set some environment variables
# Flag to solve MUMPS hanging
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "0"
# os.environ["JAXSIM_COLLISION_USE_BOTTOM_ONLY"] = "0"
# os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax.numpy as jnp
import jaxsim.api as js
import matplotlib.pyplot as plt
import numpy as np
import rod
from rod.builder.primitives import BoxBuilder

from comodo.centroidalMPC.centroidalMPC import CentroidalMPC
from comodo.centroidalMPC.mpcParameterTuning import MPCParameterTuning
from comodo.jaxsimSimulator import JaxsimContactModelEnum, JaxsimSimulator
from comodo.robotModel.createUrdf import createUrdf
from comodo.robotModel.robotModel import RobotModel
from comodo.TSIDController.TSIDController import TSIDController
from comodo.TSIDController.TSIDParameterTuning import TSIDParameterTuning

# %%
# ==== Set simulation parameters ====

T = 10.0
js_dt = 0.001


# %%
# Load the box model
import jaxsim
from jaxsim import VelRepr

rod_sdf = rod.Sdf(
    version="1.10",
    model=BoxBuilder(x=0.2, y=0.2, z=0.2, mass=1.0, name="box")
    .build_model()
    .add_link()
    .add_inertial()
    .add_visual()
    .add_collision()
    .build(),
)
model = js.model.JaxSimModel.build_from_model_description(
    model_description=rod_sdf,
    time_step=js_dt,
    contact_params=jaxsim.rbda.contacts.RelaxedRigidContactsParams.build(mu=0.001),
)

# %%
import jax

BATCH_SIZE = 64
DOFS = 20
N_SIM_TIME_STEPS = int(T / js_dt)


print(f"Running on {jax.devices()}")


import os

import jaxlie
import jaxsim
# %%
import jaxsim.api as js

# xyz = np.array([0.0, 0.0, 0.7])
# rpy = np.array([20.0, 0.0, 0.0]) * np.pi / 180
# xyz_rpy_0 = np.concatenate([xyz, rpy])

# batched_data = jax.vmap(
#     lambda xyz_rpy: js.data.JaxSimModelData.build(
#         model=model,
#         base_position=xyz_rpy[0:3],
#         # joint_positions=s,
#         base_quaternion=jaxlie.SO3.from_rpy_radians(*xyz_rpy[3:]).wxyz,
#     ),
#     in_axes=(0),
# )(
#     jnp.repeat(xyz_rpy_0[None, ...], BATCH_SIZE, axis=0),
# )


# Generate random data with different initial conditions

# Create a random JAX key.
key = jax.random.PRNGKey(seed=0)

# Split subkeys for sampling random initial data.
key, *subkeys = jax.random.split(key=key, num=BATCH_SIZE + 1)

# Create the batched data by sampling the height from [0.5, 0.6] meters.
batched_data = jax.vmap(
    lambda key: js.data.random_model_data(
        model=model,
        key=key,
        base_pos_bounds=([0, 0, 0.7], [0, 0, 1.2]),
        base_vel_lin_bounds=(0, 0),
        base_vel_ang_bounds=(0, 0),
        velocity_representation=VelRepr.Mixed,
    )
)(jnp.vstack(subkeys))

print("W_p_B(t0)=\n", batched_data.base_position[0:10])


jit_step = jax.jit(
    jax.vmap(lambda model, data: js.model.step(model, data), in_axes=(None, 0))
)


# %%
# xyz_rpy_0

# %%
# Compile the simulation loop
_ = jit_step(model, batched_data)

import mediapy as media
# %%
from jaxsim.mujoco import ModelToMjcf, MujocoModelHelper, MujocoVideoRecorder

mjcf_string, assets = ModelToMjcf.convert(rod_sdf.model)

_mj_model_helper = MujocoModelHelper.build_from_xml(
    mjcf_description=mjcf_string, assets=assets
)


framerate = 30
_recorder = MujocoVideoRecorder(
    model=_mj_model_helper.model,
    data=_mj_model_helper.data,
    fps=framerate,
    width=320 * 4,
    height=240 * 4,
)


_mj_model_helper.set_base_position(
    position=np.array(batched_data.base_position[0]),
)
_mj_model_helper.set_base_orientation(
    orientation=np.array(batched_data.base_quaternion[0]),
)


# import mediapy as media

media.show_image(_recorder.render_frame(), width=640, height=480)

# %%
# Perform the batched simulation on GPU

batched_base_positions = np.zeros((N_SIM_TIME_STEPS, BATCH_SIZE, 3))
batched_base_quaternions = np.zeros((N_SIM_TIME_STEPS, BATCH_SIZE, 4))

now = time.perf_counter()

for idx in range(N_SIM_TIME_STEPS):
    ti = idx * js_dt
    print(f"t = {ti:.3f} / {N_SIM_TIME_STEPS * js_dt} s", end="\r")
    batched_data = jit_step(
        model,
        batched_data,
    )
    batched_base_positions[idx] = batched_data.base_position
    batched_base_quaternions[idx] = batched_data.base_quaternion

wall_time = time.perf_counter() - now

# %%

# %%

# Plot the z value of batched_base_positions using batch index 0
time_vec = np.arange(N_SIM_TIME_STEPS) * js_dt
plt.figure(figsize=(10, 6))
for batch_idx in range(BATCH_SIZE):
    plt.plot(
        time_vec,
        batched_base_positions[:, batch_idx, 2],
        label=f"Batch {batch_idx}, alpha=0.7",
    )
plt.xlabel("Time [s]")
plt.ylabel("Base Position Z [m]")
plt.title("Base Position Z over Time for All Batches")
plt.grid()
# plt.ylim(-0.5, 1.5)
# plt.legend()
plt.show()


# %%
rtf = T / wall_time * BATCH_SIZE * 100
print(f"RTF: {rtf:.1f}%")

# %%
import pickle

# Save batched_data to a pickle file
batched_data_log = {
    "base_positions": batched_base_positions,
    "base_quaternions": batched_base_quaternions,
    # "joint_positions": batched_joint_positions,
}
with open("batched_data.pkl", "wb") as f:
    pickle.dump(batched_data_log, f)

print("batched_data has been saved to batched_data.pkl")

# %%
batched_base_quaternions.shape

# %%
del _recorder

# %%
# Check the results from one of the batched simulations

_recorder = MujocoVideoRecorder(
    model=_mj_model_helper.model,
    data=_mj_model_helper.data,
    fps=framerate,
    width=320 * 4,
    height=240 * 4,
)
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def get_repo_root(
    current_path: Path = Path(os.path.abspath("jaxsim_walking.ipynb")).parent,
) -> Path:
    current_path = current_path.resolve()

    for parent in current_path.parents:
        if (parent / ".git").exists():
            return parent

    raise RuntimeError("No .git directory found, not a Git repository.")


def create_output_dir(directory: Path):
    # Create the directory if it doesn't exist
    directory.mkdir(parents=True, exist_ok=True)


# Usage
repo_root = get_repo_root()

# Define the results directory
results_dir = repo_root / "examples" / "results"

# Create the results directory if it doesn't exist
create_output_dir(results_dir)

batch_model_idx = 0
t_frame_prev = -1.0
try:
    for idx in range(N_SIM_TIME_STEPS):
        ti = idx * js_dt
        if (ti - t_frame_prev) > (1.0 / framerate):
            t_frame_prev = ti
            _mj_model_helper.set_base_position(
                position=batched_base_positions[idx, batch_model_idx]
            )
            # print(base_quat, jnp.linalg.norm(base_quat))
            _mj_model_helper.set_base_orientation(
                orientation=batched_base_quaternions[idx, batch_model_idx]
            )
            # _mj_model_helper.set_joint_positions(
            #     positions=joint_pos, joint_names=model.joint_names()
            # )
            _recorder.record_frame()
finally:
    _recorder.write_video(results_dir / pathlib.Path(current_time + "batched.mp4"))

# %%
# Visualize a grid of the batched simulations

from comodo.utils.iDynVis import iDynTreeVisualizer
from comodo.utils.visualize_grid import (
    generate_equally_spaced_coordinates,
    quaternion_to_matrix,
)

# Define the number of boxes and the grid size
number_of_boxes = BATCH_SIZE
lengths_table_in_meters = 5

# Initialize the visualizer
idyntree_viz = iDynTreeVisualizer()
idyntree_viz.prepare_visualization()
idyntree_viz.modify_camera(lengths_table_in_meters)

# Create the output directory for screenshots
output_dir = "./screenshots/"
os.makedirs(output_dir, exist_ok=True)

# Generate equally spaced coordinates for the grid
coordinates = generate_equally_spaced_coordinates(
    n=number_of_boxes, l=lengths_table_in_meters
)

import rod.urdf.exporter

# Convert the SDF to URDF.
urdf_string = rod.urdf.exporter.UrdfExporter(pretty=True, indent="    ").to_urdf_string(
    sdf=rod_sdf
)


# Add the models to the visualizer
name_of_boxes = []
for i in range(number_of_boxes):
    name_i = "box" + str(i)
    name_of_boxes.append(name_i)
    idyntree_viz.add_model(
        joint_name_list=[],  # Assuming the box has no joints
        base_link="base_link",  # Replace with the actual base link name
        urdf_string=urdf_string,
        model_name=name_i,
    )

# Iterate over each time step
t_frame_prev = -1.0
for time_idx in range(N_SIM_TIME_STEPS):
    ti = time_idx * js_dt
    if (ti - t_frame_prev) > (1.0 / framerate):
        t_frame_prev = ti
        for idx_box, name_i in enumerate(name_of_boxes):
            coordinates_i = coordinates[idx_box]
            position = batched_base_positions[time_idx, idx_box]
            orientation = np.array(
                batched_base_quaternions[time_idx, idx_box]
            ).squeeze()
            # Update the model in the visualizer
            idyntree_viz.update_model(
                np.array([]),  # No joint positions for the box
                jaxlie.SE3.from_rotation_and_translation(
                    translation=position,
                    rotation=jaxlie.SO3.from_quaternion_xyzw(orientation[[1, 2, 3, 0]]),
                ).as_matrix(),
                model_name=name_i,
                delta_x=coordinates_i[0],
                delta_y=coordinates_i[1],
            )

        # Render and save the frame
        idyntree_viz.viz.run()
        idyntree_viz.viz.draw()
        filename = f"screenshot_{time_idx:04d}.png"
        idyntree_viz.viz.drawToFile(output_dir + filename)
# %%
