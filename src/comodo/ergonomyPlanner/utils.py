import numpy as np
import casadi as cs
import math


def FromQuaternionToMatrix():
    # Quaternion variable
    H = cs.SX.eye(4)
    q = cs.SX.sym("q", 7)

    R = cs.SX.eye(3) + 2 * q[0] * cs.skew(q[1:4]) + 2 * cs.mpower(cs.skew(q[1:4]), 2)

    H[:3, :3] = R
    H[:3, 3] = q[4:7]
    H = cs.Function("H", [q], [H])
    return H


# returns casadi function computing the rotation error and the position error as error_rot, error_pos
def TransfMatrixError():
    H = cs.SX.sym("H", 4, 4)
    H_des = cs.SX.sym("H_des", 4, 4)
    error_rot = cs.SX.ones(3)
    error_pos = cs.SX.ones(3)
    R = H[:3, :3]
    R_des = H_des[:3, :3]
    p = H[:3, 3]
    p_des = H_des[:3, 3]

    Temp = cs.mtimes(R, cs.transpose(R_des))
    error_rot = SkewVee(Temp)
    error_pos = p - p_des
    error_rot = cs.Function("error_rot", [H, H_des], [error_rot])
    error_pos = cs.Function("error_pos", [H, H_des], [error_pos])
    return error_rot, error_pos


def zAxisAngle():
    H = cs.SX.sym("H", 4, 4)
    theta = cs.SX.sym("theta")
    theta = cs.dot([0, 0, 1], H[:3, 2]) - 1

    error = cs.Function("error", [H], [theta])
    return error


def SkewVee(X):
    X_skew = 0.5 * (X - cs.transpose(X))
    x = cs.vertcat(-X_skew[1, 2], X_skew[0, 2], -X_skew[0, 1])
    return x


def JointCenterError(joint_min, joint_max):
    joint_pos = cs.SX.sym("joint_pos")

    error_joint_center = cs.SX.sym("joint_error")
    delta_joint = (joint_max - joint_min) / 2
    error_joint_center = (joint_pos - (joint_min + delta_joint)) / (delta_joint)

    return cs.Function("jointError", [joint_pos], [error_joint_center])


def matrix_to_xyzrpy(matrix):
    xyz_rpy = np.zeros(6)
    xyz_rpy[0] = matrix[0, 3]
    xyz_rpy[1] = matrix[1, 3]
    xyz_rpy[2] = matrix[2, 3]

    # Extract rotation matrix
    R = matrix[:3, :3]

    # Get yaw, pitch, roll from rotation matrix
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        xyz_rpy[3] = np.arctan2(R[2, 1], R[2, 2])
        xyz_rpy[4] = np.arctan2(-R[2, 0], sy)
        xyz_rpy[5] = np.arctan2(R[1, 0], R[0, 0])
    else:
        xyz_rpy[3] = np.arctan2(-R[1, 2], R[1, 1])
        xyz_rpy[4] = np.arctan2(-R[2, 0], sy)
        xyz_rpy[5] = 0
    return xyz_rpy
