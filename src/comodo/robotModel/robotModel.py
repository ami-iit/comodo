from adam.casadi.computations import KinDynComputations
import numpy as np
import pathlib
from typing import Union
from urchin import URDF
from urchin import Joint
from urchin import Link
from typing import Dict
import mujoco
import tempfile
import xml.etree.ElementTree as ET
import idyntree.bindings as iDynTree
import casadi as cs
import copy
import resolve_robotics_uri_py as rru
from pathlib import Path


class RobotModel(KinDynComputations):
    def __init__(
        self,
        urdf_path: Union[str, pathlib.Path, pathlib.PurePath, pathlib.PurePosixPath],
        robot_name: str,
        joint_name_list: list,
        base_link: str = "root",
        left_foot: str = None,
        right_foot: str = None,
        torso: str = None,
        right_foot_rear_link_name: str = None,
        right_foot_front_link_name: str = None,
        left_foot_rear_link_name: str = None,
        left_foot_front_link_name: str = None,
        kp_pos_control: np.float32 = np.array(
            [
                35 * 70.0,
                35 * 70.0,
                35 * 40.0,
                35 * 100.0,
                35 * 100.0,
                35 * 100.0,
                35 * 70.0,
                35 * 70.0,
                35 * 40.0,
                35 * 100.0,
                35 * 100.0,
                35 * 100.0,
                20 * 5.745,
                20 * 5.745,
                20 * 5.745,
                20 * 1.745,
                20 * 5.745,
                20 * 5.745,
                20 * 5.745,
                20 * 1.745,
            ]
        ),
        kd_pos_control: np.float32 = np.array(
            [
                15 * 0.15,
                15 * 0.15,
                15 * 0.35,
                15 * 0.15,
                15 * 0.15,
                15 * 0.15,
                15 * 0.15,
                15 * 0.15,
                15 * 0.35,
                15 * 0.15,
                15 * 0.15,
                15 * 0.15,
                4 * 5.745,
                4 * 5.745,
                4 * 5.745,
                4 * 1.745,
                4 * 5.745,
                4 * 5.745,
                4 * 5.745,
                4 * 1.745,
            ]
        ),
    ) -> None:
        valid_path_types = (str, pathlib.Path, pathlib.PurePath, pathlib.PurePosixPath)
        if not isinstance(urdf_path, valid_path_types):
            raise TypeError(f"urdf_path must be a string or a pathlib object, but got {type(urdf_path)}")

        self.collision_keyword = "_collision"
        self.visual_keyword = "_visual"
        self.urdf_path = str(urdf_path)
        self.robot_name = robot_name
        self.joint_name_list = joint_name_list
        self.base_link = base_link
        self.left_foot_frame = left_foot
        self.right_foot_frame = right_foot
        # self.torso_link = torso
        # self.right_foot_rear_ct = right_foot_rear_link_name + self.collision_keyword
        # self.right_foot_front_ct = right_foot_front_link_name + self.collision_keyword
        # self.left_foot_rear_ct = left_foot_rear_link_name + self.collision_keyword
        # self.left_foot_front_ct = left_foot_front_link_name + self.collision_keyword

        self.remote_control_board_list = [
            "/" + self.robot_name + "/torso",
            "/" + self.robot_name + "/left_arm",
            "/" + self.robot_name + "/right_arm",
            "/" + self.robot_name + "/left_leg",
            "/" + self.robot_name + "/right_leg",
        ]

        self.kp_position_control = kp_pos_control
        self.kd_position_control = kd_pos_control
        self.ki_position_control = 10 * self.kd_position_control
        self.gravity = iDynTree.Vector3()
        self.gravity.zero()
        self.gravity.setVal(2, -9.81)
        self.H_b = iDynTree.Transform()
        super().__init__(urdf_path, self.joint_name_list, self.base_link)
        # self.H_left_foot = self.forward_kinematics_fun(self.left_foot_frame)
        # self.H_right_foot = self.forward_kinematics_fun(self.right_foot_frame)

    def override_control_boar_list(self, remote_control_board_list: list):
        self.remote_control_board_list = remote_control_board_list

    def set_foot_corner(self, corner_0, corner_1, corner_2, corner_3):
        self.corner_0 = corner_0
        self.corner_1 = corner_1
        self.corner_2 = corner_2
        self.corner_3 = corner_3

    def set_limbs_indexes(
        self,
        right_arm_indexes: list,
        left_arm_indexes: list,
        right_leg_indexes: list,
        left_leg_indexes: list,
        torso_indexes: list,
    ):
        self.right_arm_indexes = right_arm_indexes
        self.left_arm_indexes = left_arm_indexes
        self.right_leg_indexes = right_leg_indexes
        self.left_leg_indexes = left_leg_indexes
        self.torso_indexes = torso_indexes

    def get_idyntree_kyndyn(self):
        model_loader = iDynTree.ModelLoader()
        model_loader.loadReducedModelFromString(
            copy.deepcopy(self.urdf_path), self.joint_name_list
        )
        kindyn = iDynTree.KinDynComputations()
        kindyn.loadRobotModel(model_loader.model())
        return kindyn

    def compute_desired_position_walking(self):
        # desired_knee = -1.22
        desired_knee = -1.0
        shoulder_roll = 0.251
        elbow = 0.616

        p_opts = {}
        s_opts = {"linear_solver": "mumps"}
        self.solver = cs.Opti()
        self.solver.solver("ipopt", p_opts, s_opts)

        self.w_H_torso = self.forward_kinematics_fun(self.torso_link)
        self.s = self.solver.variable(self.NDoF)  # joint positions
        self.quat_pose_b = self.solver.variable(7)
        left_foot_pos = np.asarray([0.000267595, 0.0801685, 0.0])
        left_foot_rotation = np.eye(3)
        right_foot_pos = np.asarray([-0.000139057, -0.0803188, 0.0])
        right_foot_rotation = np.eye(3)
        root_link_rotation = np.eye(3)
        quat_to_transf = self.from_quaternion_to_matrix()
        H_b = quat_to_transf(self.quat_pose_b)
        H_left_foot = self.H_left_foot(H_b, self.s)
        quat_left_foot = self.rotation_matrix_to_quaternion(H_left_foot[:3, :3])
        H_right_foot = self.H_right_foot(H_b, self.s)
        quat_right_foot = self.rotation_matrix_to_quaternion(H_right_foot[:3, :3])
        H_torso = self.w_H_torso(H_b, self.s)
        quat_torso = self.rotation_matrix_to_quaternion(H_torso[:3, :3])
        reference_rotation = np.asarray([1.0, 0.0, 0.0, 0.0])
        for index, joint_name in enumerate(self.joint_name_list):
            if "knee" in joint_name:
                self.solver.subject_to(self.s[index] == desired_knee)
            if "shoulder_roll" in joint_name:
                self.solver.subject_to(self.s[index] == shoulder_roll)
            if "elbow" in joint_name:
                self.solver.subject_to(self.s[index] == elbow)

        self.solver.subject_to(H_left_foot[2, 3] == 0.0)
        self.solver.subject_to(H_right_foot[2, 3] == 0.0)
        self.solver.subject_to(quat_left_foot == reference_rotation)
        self.solver.subject_to(quat_right_foot == reference_rotation)
        self.solver.subject_to(quat_torso == reference_rotation)
        cost_function = cs.sumsqr(self.s)
        cost_function += cs.sumsqr(H_left_foot[:2, 3] - left_foot_pos[:2])
        cost_function += cs.sumsqr(H_right_foot[:2, 3] - right_foot_pos[:2])
        self.solver.minimize(cost_function)
        self.sol = self.solver.solve()
        s_return = np.array(self.sol.value(self.s))
        quat_base_opt = np.array(self.sol.value(self.quat_pose_b))
        H_b_return = quat_to_transf(quat_base_opt)
        xyz_rpy = self.matrix_to_rpy(H_b_return)
        return s_return, xyz_rpy, H_b_return

    def get_left_arm_from_joint_position(self, s):
        if self.left_arm_indexes is None:
            return None
        return np.array(s[self.left_arm_indexes[0] : self.left_arm_indexes[1]])

    def get_right_arm_from_joint_position(self, s):
        if self.right_arm_indexes is None:
            return None
        return np.array(s[self.right_arm_indexes[0] : self.right_arm_indexes[1]])

    def get_left_leg_from_joint_position(self, s):
        if self.left_leg_indexes is None:
            return None
        return np.array(s[self.left_leg_indexes[0] : self.left_leg_indexes[1]])

    def get_right_leg_from_joint_position(self, s):
        if self.right_leg_indexes is None:
            return None
        return np.array(s[self.right_leg_indexes[0] : self.right_leg_indexes[1]])

    def get_torso_from_joint_position(self, s):
        if self.torso_indexes is None:
            return None
        return np.array(s[self.torso_indexes[0] : self.torso_indexes[1]])

    def compute_base_pose_left_foot_in_contact(self, s):
        w_H_torso = self.forward_kinematics_fun(self.base_link)
        w_H_leftFoot = self.forward_kinematics_fun(self.left_foot_frame)

        w_H_torso_num = np.array(w_H_torso(np.eye(4), s))
        w_H_lefFoot_num = np.array(w_H_leftFoot(np.eye(4), s))
        w_H_init = np.linalg.inv(w_H_lefFoot_num) @ w_H_torso_num
        return w_H_init

    def compute_com_init(self):
        com = self.CoM_position_fun()
        return np.array(com(self.w_H_b_init, self.s_init))

    def set_initial_position(self, s_init, w_H_b_init, xyz_rpy_init):
        self.s_init = s_init
        self.w_H_b_init = w_H_b_init
        self.xyz_rpy_init = xyz_rpy_init

    def rotation_matrix_to_quaternion(self, R):
        # Ensure the matrix is a valid rotation matrix (orthogonal with determinant 1)
        trace = cs.trace(R)
        S = cs.sqrt(trace + 1.0) * 2
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S

        # Normalize the quaternion
        length = cs.sqrt(w**2 + x**2 + y**2 + z**2)
        w /= length
        x /= length
        y /= length
        z /= length

        return cs.vertcat(w, x, y, z)

    def get_mujoco_urdf_string(self) -> str:
        ## We first start by ET
        tempFileOut = tempfile.NamedTemporaryFile(mode="w+")
        tempFileOut.write(copy.deepcopy(self.urdf_path))

        parser = ET.XMLParser(encoding="utf-8")
        tree = ET.parse(self.urdf_path, parser=parser)
        root = tree.getroot()
        self.mujoco_joint_order = []
        # Declaring as fixed the not controlled joints
        for joint in root.findall(".//joint"):
            joint_name = joint.attrib.get("name")
            if joint_name not in self.joint_name_list:
                joint.set("type", "fixed")
            else:
                self.mujoco_joint_order.append(joint_name)

        new_link = ET.Element("link")
        new_link.set("name", "world")
        root.append(new_link)
        floating_joint = ET.Element("joint")
        floating_joint.set("name", "floating_joint")
        floating_joint.set("type", "floating")
        # Create parent element
        parent = ET.Element("parent")
        parent.set("link", "world")
        floating_joint.append(parent)

        # Create child element
        child = ET.Element("child")
        child.set("link", self.base_link)
        floating_joint.append(child)

        # Append the new joint element to the root
        root.append(floating_joint)

        ## Adding the name to the collision and link visual
        for link in root.findall(".//link"):
            link_name = link.attrib.get("name")
            # Check if a collision element with a name exists
            collision_elements = link.findall("./collision")
            if not any(
                collision.find("name") is not None for collision in collision_elements
            ):
                # If no collision element with a name exists, add one
                if collision_elements:
                    # If there are collision elements, append name to the first one
                    collision_elements[0].set(
                        "name", link_name + self.collision_keyword
                    )
            visual_elements = link.findall("./visual")
            if not any(visual.find("name") is not None for visual in visual_elements):
                # If no collision element with a name exists, add one
                if visual_elements:
                    visual_elements[0].set("name", link_name + self.visual_keyword)
        meshes = self.get_mesh_path(root)
        robot_el = None
        for elem in root.iter():
            if elem.tag == "robot":
                robot_el = elem
                break
        mujoco_el = ET.Element("mujoco")
        compiler_el = ET.Element("compiler")
        compiler_el.set("discardvisual", "false")
        if not (meshes is None):
            compiler_el.set("meshdir", str(meshes))
        mujoco_el.append(compiler_el)
        robot_el.append(mujoco_el)
        # Convert the XML tree to a string
        robot_urdf_string_original = ET.tostring(root, encoding="unicode")
        return robot_urdf_string_original

    def get_mujoco_model(self, floor_opts: Dict, save_mjc_xml: bool = False) -> mujoco.MjModel:
        valid_floor_opts = ["inclination_deg", "sliding_friction", "torsional_friction", "rolling_friction"]
        for key in floor_opts.keys():
            if key not in valid_floor_opts:
                raise ValueError(f"Invalid key {key} in floor_opts. Valid keys are {valid_floor_opts}")
            
        floor_inclination_deg = floor_opts.get("inclination_deg", [0, 0, 0])
        sliding_friction = floor_opts.get("sliding_friction", 1)
        torsional_friction = floor_opts.get("torsional_friction", 0.005)
        rolling_friction = floor_opts.get("rolling_friction", 0.0001)
    
        # Type & value checking
        try:
            floor_inclination = np.array(floor_inclination_deg)
            floor_inclination = np.deg2rad(floor_inclination)
        except:
            raise ValueError(f"floor's inclination_deg must be a sequence of 3 elements, but got {floor_inclination_deg} of type {type(floor_inclination_deg)}")

        for friction_coeff in ("sliding_friction", "torsional_friction", "rolling_friction"):
            if not isinstance(eval(friction_coeff), (int, float)):
                raise ValueError(f"{friction_coeff} must be a number (int, float), but got {type(eval(friction_coeff))}")
            if not eval(f"0 <= {friction_coeff} <= 1"):
                raise ValueError(f"{friction_coeff} must be in the range [0, 1], but got {eval(friction_coeff)}")
        
        # Get the URDF string
        urdf_string = self.get_mujoco_urdf_string()
        try:
            mujoco_model = mujoco.MjModel.from_xml_string(urdf_string)
        except Exception as e:
            with tempfile.NamedTemporaryFile(mode="w+", delete=False) as file:
                file.write(urdf_string)
            raise type(e)(f"Error while creating the Mujoco model from the URDF string (path={self.urdf_path}) ==> {e}. Urdf string dumped to {file.name}")
        path_temp_xml = tempfile.NamedTemporaryFile(mode="w+")
        mujoco.mj_saveLastXML(path_temp_xml.name, mujoco_model)

        # Adding the Motors
        tree = ET.parse(path_temp_xml)
        root = tree.getroot()

        mujoco_elem = None
        for elem in root.iter():
            if elem.tag == "mujoco":
                mujoco_elem = elem
                break
        actuator_entry = ET.Element("actuator")

        for name_joint in self.mujoco_joint_order:
            new_motor_entry = ET.Element("motor")
            new_motor_entry.set("name", name_joint)
            new_motor_entry.set("joint", name_joint)
            new_motor_entry.set(
                "gear", "1"
            )  # This can be changed to direclty give motor torques
            new_motor_entry.set(
                "ctrlrange", "-100 100"
            )  # Also this can be changed to give the exact motor limits
            actuator_entry.append(new_motor_entry)
        mujoco_elem.append(actuator_entry)

        # Adding various assets

        asset_entry = ET.Element("asset")
        # <texture type="skybox" builtin="gradient" rgb1=".3 .5 .7" rgb2="0 0 0" width="32" height="512"/>
        new_texture_entry = ET.Element("texture")
        new_texture_entry.set("type", "skybox")
        new_texture_entry.set("builtin", "gradient")
        new_texture_entry.set("rgb1", ".3 .5 .7")
        new_texture_entry.set("rgb2", "0 0 0")
        new_texture_entry.set("width", "32")
        new_texture_entry.set("height", "512")
        asset_entry.append(new_texture_entry)

        # <texture name="body" type="cube" builtin="flat" mark="cross" width="127" height="1278" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" markrgb="1 1 1" random="0.01"/>
        new_texture_entry = ET.Element("texture")
        new_texture_entry.set("name", "body")
        new_texture_entry.set("type", "cube")
        new_texture_entry.set("builtin", "flat")
        new_texture_entry.set("mark", "cross")
        new_texture_entry.set("rgb1", "0.8 0.6 0.4")
        new_texture_entry.set("rgb2", "0.8 0.6 0.4")
        new_texture_entry.set("width", "127")
        new_texture_entry.set("height", "1278")
        new_texture_entry.set("markrgb", "1 1 1")
        new_texture_entry.set("random", "0.01")
        asset_entry.append(new_texture_entry)

        # <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1=".1 .2 .3" rgb2=".2 .3 .4"/>
        new_texture_entry = ET.Element("texture")
        new_texture_entry.set("name", "grid")
        new_texture_entry.set("type", "2d")
        new_texture_entry.set("builtin", "checker")
        new_texture_entry.set("rgb1", ".1 .2 .3")
        new_texture_entry.set("rgb2", ".2 .3 .4")
        new_texture_entry.set("width", "512")
        new_texture_entry.set("height", "512")
        asset_entry.append(new_texture_entry)

        # <material name="body" texture="body" texuniform="true" rgba="0.8 0.6 .4 1"/>
        new_material_entry = ET.Element("material")
        new_material_entry.set("name", "body")
        new_material_entry.set("texture", "body")
        new_material_entry.set("texuniform", "true")
        new_material_entry.set("rgba", "0.8 0.6 .4 1")
        asset_entry.append(new_material_entry)

        # <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance=".2"/>
        new_material_entry = ET.Element("material")
        new_material_entry.set("name", "grid")
        new_material_entry.set("texture", "grid")
        new_material_entry.set("texrepeat", "1 1")
        new_material_entry.set("texuniform", "true")
        new_material_entry.set("reflectance", ".2")
        asset_entry.append(new_material_entry)

        mujoco_elem.append(asset_entry)

        ## Adding the floor
        #   <geom name="floor" size="0 0 .05" type="plane" material="grid" condim="3"/>
        world_elem = None
        for elem in root.iter():
            if elem.tag == "worldbody":
                world_elem = elem
                break
        floor = ET.Element("geom")
        floor.set("name", "floor")
        floor.set("size", "0 0 .04")
        floor.set("type", "plane")
        floor.set("material", "grid")
        floor.set("condim", "3")
        floor.set("euler", "{} {} {}".format(*floor_inclination))
        floor.set("friction", "{} {} {}".format(sliding_friction, torsional_friction, rolling_friction))
        world_elem.append(floor)
        new_xml = ET.tostring(tree.getroot(), encoding="unicode")

        if save_mjc_xml:
            with open("./mujoco_model.xml", "w") as f:
                f.write(new_xml)

        return new_xml

    def get_base_pose_from_contacts(self, s, contact_frames_pose: dict):
        kindyn = self.get_idyntree_kyndyn()
        s = iDynTree.VectorDynSize.FromPython(s)
        ds = iDynTree.VectorDynSize.FromPython(np.zeros(self.NDoF))
        self.w_b = iDynTree.Twist()
        self.w_b.zero()

        kindyn.setRobotState(s, ds, self.gravity)

        w_p_b = np.zeros(3)
        w_H_b = np.eye(4)

        for key, value in contact_frames_pose.items():
            w_H_b_i = (
                value
                @ kindyn.getRelativeTransform(key, kindyn.getFloatingBase())
                .asHomogeneousTransform()
                .toNumPy()
            )
            w_p_b = w_p_b + w_H_b_i[0:3, 3]

        w_H_b[0:3, 3] = w_p_b / len(contact_frames_pose)
        # for the time being for the orientation we are just using the orientation of the last contact
        w_H_b[0:3, 0:3] = w_H_b_i[0:3, 0:3]

        return w_H_b

    def get_base_velocity_from_contacts(self, H_b, s, ds, contact_frames_list: list):
        kindyn = self.get_idyntree_kyndyn()
        s = iDynTree.VectorDynSize.FromPython(s)
        ds = iDynTree.VectorDynSize.FromPython(ds)
        w_b = iDynTree.Twist()
        w_b.zero()
        self.H_b.fromHomogeneousTransform(iDynTree.Matrix4x4.FromPython(H_b))
        kindyn.setRobotState(H_b, s, w_b, ds, self.gravity)
        Jc_multi_contacts = self.get_frames_jacobian(kindyn, contact_frames_list)
        a = Jc_multi_contacts[:, 0:6]
        b = -Jc_multi_contacts[:, 6:]
        w_b = np.linalg.lstsq(
            Jc_multi_contacts[:, 0:6],
            -Jc_multi_contacts[:, 6:] @ ds.toNumPy(),
            rcond=-1,
        )[0]
        return w_b

    def get_frames_jacobian(self, kindyn, frames_list: list):
        Jc_frames = np.zeros([6 * len(frames_list), 6 + self.NDoF])

        for idx, frame_name in enumerate(frames_list):
            Jc = iDynTree.MatrixDynSize(6, 6 + self.NDoF)
            kindyn.getFrameFreeFloatingJacobian(frame_name, Jc)

            Jc_frames[idx * 6 : (idx * 6 + 6), :] = Jc.toNumPy()

        return Jc_frames

    def from_quaternion_to_matrix(self):
        f_opts = (dict(jit=False, jit_options=dict(flags="-Ofast")),)
        # Quaternion variable
        H = cs.SX.eye(4)
        q = cs.SX.sym("q", 7)
        R = (
            cs.SX.eye(3)
            + 2 * q[0] * cs.skew(q[1:4])
            + 2 * cs.mpower(cs.skew(q[1:4]), 2)
        )

        H[:3, :3] = R
        H[:3, 3] = q[4:7]
        H = cs.Function("H", [q], [H])
        return H

    def compute_step_lenght(self, s, H_b):
        theta_max = 0.7853981633974483
        relative_transf = self.w_H_torso(H_b, s) @ np.linalg.inv(
            self.H_left_foot(H_b, s)
        )
        leg_bended_lenght = relative_transf[2, 3]
        step_lenght_max = np.sin(theta_max) * leg_bended_lenght
        percentage = 0.2
        return percentage * step_lenght_max

    def matrix_to_rpy(self, matrix):
        """Converts a rotation matrix to roll, pitch, and yaw angles in radians.

        Args:
            matrix (numpy.ndarray): 4x4 rotation matrix.

        Returns:
            tuple: Tuple containing the roll, pitch, and yaw angles in radians.
        """
        # assert matrix.shape == (3, 3), "Input matrix must be a 3x3 rotation matrix."

        # Extract rotation angles from the rotation matrix
        xyz = np.asarray([matrix[0, 3], matrix[1, 3], matrix[2, 3]])
        # xyz = matrix[:3,3]
        yaw = np.arctan2(matrix[1, 0], matrix[0, 0])
        pitch = np.arctan2(
            -matrix[2, 0], np.sqrt(matrix[2, 1] ** 2 + matrix[2, 2] ** 2)
        )
        roll = np.arctan2(matrix[2, 1], matrix[2, 2])
        xyz_rpy = np.zeros(6)
        xyz_rpy[0] = matrix[0, 3]
        xyz_rpy[1] = matrix[1, 3]
        xyz_rpy[2] = matrix[2, 3]
        xyz_rpy[3] = roll
        xyz_rpy[4] = pitch
        xyz_rpy[5] = yaw
        return xyz_rpy

    def get_centroidal_momentum_jacobian(self):
        Jcm = iDynTree.MatrixDynSize(6, 6 + self.ndof)
        self.kindyn.getCentroidalTotalMomentumJacobian(Jcm)
        return Jcm.toNumPy()

    def get_centroidal_momentum(self):
        Jcm = self.get_centroidal_momentum_jacobian()
        nu = np.concatenate((self.w_b.toNumPy(), self.ds.toNumPy()))
        return Jcm @ nu

    def get_mesh_path(self, robot_urdf: ET.Element) -> Path:
        """
        Get the mesh path from the robot urdf.

        Args:
            robot_urdf (ET.Element): The robot urdf.

        Returns:
            Path: The mesh path.
        """
        # find the mesh path
        mesh = robot_urdf.find(".//mesh")
        if mesh is None:
            return None
        path = mesh.attrib["filename"]
        mesh_path = rru.resolve_robotics_uri(path).parent

        return mesh_path
