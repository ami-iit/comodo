import copy
import os
import pathlib

import jax
import jax.numpy as jnp
import jaxsim
import jaxsim.api as js
import matplotlib.pyplot as plt
import numpy as np

from .iDynVis import iDynTreeVisualizer

os.environ["JAXSIM_DISABLE_EXCEPTIONS"] = "0"

import math


def generate_equally_spaced_coordinates(n, l):
    """
    Z
    Generates n equally spaced (x, y) coordinates within a chessboard.

    Args:
    - n: Number of coordinates to generate.
    - l: Length of the chessboard (meters).

    Returns:
    - A list of n tuples, where each tuple is (x, y) representing a coordinate.
    """
    # Calculate the number of rows and columns in the grid
    grid_size = math.ceil(math.sqrt(n))  # Square root gives closest grid dimensions
    step = l / grid_size  # Distance between points in the grid

    coordinates = []
    for i in range(grid_size):
        for j in range(grid_size):
            # Compute the coordinate
            x = round(i * step + step / 2, 5)  # Center of the grid cell in x
            y = round(j * step + step / 2, 5)  # Center of the grid cell in y
            coordinates.append((x, y))
            if len(coordinates) == n:  # Stop if we've generated enough points
                return coordinates
    return coordinates


def quaternion_to_matrix(position, quaternion):
    """
    Convert position and quaternion to a 4x4 transformation matrix.

    Parameters
    ----------
    - position: A list or array-like of length 3 [x, y, z].
    - quaternion: A list or array-like of length 4 [qx, qy, qz, qw].

    Returns
    -------
    - A 4x4 NumPy array representing the transformation matrix.
    """
    x, y, z = position
    qx, qy, qz, qw = quaternion

    # Compute the rotation matrix components from the quaternion
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz

    rotation_matrix = np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ]
    )

    # Create the transformation matrix
    transform_matrix = np.eye(4)
    transform_matrix[:3, 3] = [x, y, z]

    return transform_matrix


def get_position(data):
    joint_position = data.joint_positions()
    base_quaternion = data.base_orientation(dcm=False)
    W_p_B = data.base_position()
    generalzed_pose = jnp.hstack((W_p_B, base_quaternion, joint_position))
    return generalzed_pose


def create_new_data_jaxsim(data_old, new_velocity, new_position):
    # data_n = js.data.JaxSimModelData()
    data_n = data_old.reset_base_quaternion(new_position[3:7])
    data_n = data_n.reset_base_position(new_position[0:3])
    data_n = data_n.reset_joint_positions(new_position[7:])
    data_n = data_n.reset_joint_velocities(new_velocity[6:])
    data_n = data_n.reset_base_linear_velocity(new_velocity[0:3])
    data_n = data_n.reset_base_angular_velocity(new_velocity[3:6])
    return data_n


# jax.config.update("jax_default_device", jax.devices("cpu")[0])
# # Load the iCub mode
# model_path = (
#     "/home/carlotta/iit_ws/ergocub-software/urdf/ergoCub/robots/ergoCubSN001/model.urdf"
# )
# joints = (
#     "torso_pitch",
#     "torso_roll",
#     "torso_yaw",
#     "l_shoulder_pitch",
#     "l_shoulder_roll",
#     "l_shoulder_yaw",
#     "l_elbow",
#     "r_shoulder_pitch",
#     "r_shoulder_roll",
#     "r_shoulder_yaw",
#     "r_elbow",
#     "l_hip_pitch",
#     "l_hip_roll",
#     "l_hip_yaw",
#     "l_knee",
#     "l_ankle_pitch",
#     "l_ankle_roll",
#     "r_hip_pitch",
#     "r_hip_roll",
#     "r_hip_yaw",
#     "r_knee",
#     "r_ankle_pitch",
#     "r_ankle_roll",
# )
# ndof = len(joints)

# # Build and reduce the model
# model_description = pathlib.Path(model_path)
# full_model = js.model.JaxSimModel.build_from_model_description(
#     model_description=model_description,
#     time_step=0.0001,
#     is_urdf=True,
#     contact_model=jaxsim.rbda.contacts.RelaxedRigidContacts.build(),
# )
# model = js.model.reduce(model=full_model, considered_joints=joints)

# # Initialize data and simulation
# data = js.data.JaxSimModelData.zero(model=model).reset_base_position(
#     base_position=jnp.array([0.0, 0.0, 1.8])
# )
# data = data.update_cached(model)
# T = jnp.arange(start=0, stop=1.8, step=model.time_step)
# tau = jnp.zeros(ndof)


# position_semi_implicit = []
# lengths_table_in_meters = 10
# # Prepare the visualization


# model_name = "ergoCub"
# data_list = []
# # Create dummy vector
# for t in T:
#     print(t)
#     data = js.model.step(
#         model=model,
#         data=data,
#         joint_force_references=tau,
#     )
#     data_list.append(copy.deepcopy(data))

# final_data_vector = data_list  # The last vector
# # replicated_matrix = (final_data_vector, number_of_robots)
# number_of_robots = 100
# replicated_matrix = [final_data_vector for _ in range(number_of_robots)]

# idyntree_viz = iDynTreeVisualizer()
# idyntree_viz.prepare_visualization()
# idyntree_viz.modify_camera(lengths_table_in_meters)
# import os

# name_of_robots = []
# output_dir = "./screenshots/"
# name_of_robots = []
# for i in range(number_of_robots):
#     name_i = "ergoCub" + str(i)
#     name_of_robots.append(name_i)
# os.makedirs(output_dir, exist_ok=True)

# # Creates coordinates
# coordinates = generate_equally_spaced_coordinates(
#     n=number_of_robots, l=lengths_table_in_meters
# )
# for name_i in name_of_robots:
#     idyntree_viz.add_model(
#         joint_name_list=list(joints),
#         base_link="root_link",
#         urdf_path=model_path,
#         model_name=name_i,
#     )
# print(len(final_data_vector))
# for time_istant in range(len(final_data_vector)):
#     # print(time_istant)
#     for idx_robot, name_i in enumerate(name_of_robots):
#         coordinates_i = coordinates[idx_robot]
#         robot_i = replicated_matrix[idx_robot]
#         data_i = robot_i[time_istant]
#         idyntree_viz.update_model(
#             np.array(data_i.joint_positions),
#             np.array(data_i.base_transform),
#             model_name=name_i,
#             delta_x=coordinates_i[0],
#             delta_y=coordinates_i[1],
#         )
#     idyntree_viz.viz.run()
#     idyntree_viz.viz.draw()
#     idyntree_viz.viz.drawToFile(output_dir + "screenshot_" + str(time_istant) + ".png")
