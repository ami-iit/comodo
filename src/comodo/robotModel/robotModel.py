from adam.casadi.computations import KinDynComputations
import numpy as np
from urchin import URDF
from urchin import Joint
from urchin import Link
import mujoco
import tempfile
import xml.etree.ElementTree as ET
import idyntree.bindings as iDynTree
import casadi as cs
import copy
import xml.etree.ElementTree as ET


class RobotModel(KinDynComputations):
    def __init__(
        self,
        urdfstring: str,
        robot_name: str,
        joint_name_list: list,
        base_link: str = "root_link",
        left_foot: str = "l_sole",
        right_foot: str = "r_sole",
        torso: str = "chest",
    ) -> None:
        self.urdf_string = urdfstring
        self.robot_name = robot_name
        self.joint_name_list = joint_name_list
        self.base_link = base_link
        self.left_foot_frame = left_foot
        self.right_foot_frame = right_foot
        self.torso_link = torso

        self.remote_control_board_list = [
            "/" + self.robot_name + "/torso",
            "/" + self.robot_name + "/left_arm",
            "/" + self.robot_name + "/right_arm",
            "/" + self.robot_name + "/left_leg",
            "/" + self.robot_name + "/right_leg",
        ]

        # self.mujoco_lines_urdf = '<mujoco> <compiler discardvisual="false"/> </mujoco>'
        self.gravity = iDynTree.Vector3()
        self.gravity.zero()
        self.gravity.setVal(2, -9.81)
        self.H_b = iDynTree.Transform()
        path_temp_xml = tempfile.NamedTemporaryFile(mode="w+")
        path_temp_xml.write(urdfstring)
        super().__init__(path_temp_xml.name, self.joint_name_list, self.base_link)

    def override_control_boar_list(self, remote_control_board_list: list):
        self.remote_control_board_list = remote_control_board_list

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
            copy.deepcopy(self.urdf_string), self.joint_name_list
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

        self.H_left_foot = self.forward_kinematics_fun(self.left_foot_frame)
        self.H_right_foot = self.forward_kinematics_fun(self.right_foot_frame)
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

        # self.solver.subject_to(cs.abs(self.s[11])> 0.05 )
        # self.solver.subject_to(self.s[11]< -0.05)
        # self.solver.subject_to(cs.abs(self.s[17])>0.05 )
        # self.solver.subject_to(self.s[17]< -0.05)
        self.solver.subject_to(self.s[17] == desired_knee)
        self.solver.subject_to(self.s[11] == desired_knee)
        self.solver.subject_to(self.s[3] == elbow)
        self.solver.subject_to(self.s[7] == elbow)
        self.solver.subject_to(self.s[1] == shoulder_roll)
        self.solver.subject_to(self.s[5] == shoulder_roll)
        self.solver.subject_to(self.s[9] == self.s[15])
        # self.solver.subject_to(cs.norm_2(self.quat_pose_b[:4]) == 1.0)
        self.solver.subject_to(H_left_foot[2, 3] == 0.0)
        self.solver.subject_to(H_right_foot[2, 3] == 0.0)
        self.solver.subject_to(quat_left_foot == reference_rotation)
        self.solver.subject_to(quat_right_foot == reference_rotation)
        self.solver.subject_to(quat_torso == reference_rotation)
        cost_function = cs.sumsqr(self.s)
        cost_function += cs.sumsqr(H_left_foot[:2, 3] - left_foot_pos[:2])
        cost_function += cs.sumsqr(H_right_foot[:2, 3] - right_foot_pos[:2])
        # cost_function += cs.sumsqr(self.s[11] - desired_knee)
        # cost_function += cs.sumsqr(self.s[17] - desired_knee)
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

    def rotation_matrix_to_quaternion(self, R):
        # Ensure the matrix is a valid rotation matrix (orthogonal with determinant 1)
        trace = cs.trace(R)
        S = cs.sqrt(trace + 1.0) * 2
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S

        # Normalize the quaternion
        length = cs.sqrt(w ** 2 + x ** 2 + y ** 2 + z ** 2)
        w /= length
        x /= length
        y /= length
        z /= length

        return cs.vertcat(w, x, y, z)

    def get_mujoco_urdf_string(self):
        tempFileOut = tempfile.NamedTemporaryFile(mode="w+")
        tempFileOut.write(copy.deepcopy(self.urdf_string))
        robot = URDF.load(tempFileOut.name)

        ## Adding floating joint to have floating base system
        for item in robot.joints:
            if item.name not in (self.joint_name_list):
                item.joint_type = "fixed"
        world_joint = Joint("floating_joint", "floating", "world", self.base_link)
        world_link = Link("world", None, None, None)
        robot._links.append(world_link)
        robot._joints.append(world_joint)
        temp_urdf = tempfile.NamedTemporaryFile(mode="w+")
        robot.save(temp_urdf.name)
        tree = ET.parse(temp_urdf.name)
        root = tree.getroot()
        robot_el = None
        for elem in root.iter():
            if elem.tag == "robot":
                robot_el = elem
                break
        ## Adding compiler discard visaul false for mujoco rendering
        mujoco_el = ET.Element("mujoco")
        compiler_el = ET.Element("compiler")
        compiler_el.set("discardvisual", "false")
        mujoco_el.append(compiler_el)
        robot_el.append(mujoco_el)
        # Convert the XML tree to a string
        robot_urdf_string_original = ET.tostring(root)
        # urdf_string_temp  = temp_urdf.read()
        return robot_urdf_string_original

    def get_mujoco_model(self):
        urdf_string = self.get_mujoco_urdf_string()
        # urdf_string = urdf_mujoco_file.read()
        mujoco_model = mujoco.MjModel.from_xml_string(urdf_string)

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

        for name_joint in self.joint_name_list:
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
        floor.set("size", "0 0 .05")
        floor.set("type", "plane")
        floor.set("material", "grid")
        floor.set("condim", "3")
        world_elem.append(floor)
        new_xml = ET.tostring(tree.getroot(), encoding="unicode")
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
