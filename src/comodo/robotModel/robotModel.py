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
import resolve_robotics_uri_py as rru
from pathlib import Path


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
        left_hand: str = "l_hand",
        rigth_hand: str = "r_hand",
        right_foot_rear_link_name: str = "r_foot_rear",
        right_foot_front_link_name: str = "r_foot_front",
        left_foot_rear_link_name: str = "l_foot_rear",
        left_foot_front_link_name: str = "l_foot_front",
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
        self.collision_keyword = "_collision"
        self.visual_keyword = "_visual"
        self.urdf_string = urdfstring
        self.robot_name = robot_name
        self.joint_name_list = joint_name_list
        self.base_link = base_link
        self.left_foot_frame = left_foot
        self.right_foot_frame = right_foot
        self.rigth_hand = rigth_hand
        self.left_hand = left_hand
        self.torso_link = torso
        self.right_foot_rear_ct = right_foot_rear_link_name + self.collision_keyword
        self.right_foot_front_ct = right_foot_front_link_name + self.collision_keyword
        self.left_foot_rear_ct = left_foot_rear_link_name + self.collision_keyword
        self.left_foot_front_ct = left_foot_front_link_name + self.collision_keyword
        self.is_with_box = False
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
        path_temp_xml = tempfile.NamedTemporaryFile(mode="w+")
        path_temp_xml.write(urdfstring)
        super().__init__(path_temp_xml.name, self.joint_name_list, self.base_link)
        self.H_left_foot = self.forward_kinematics_fun(self.left_foot_frame)
        self.H_right_foot = self.forward_kinematics_fun(self.right_foot_frame)

    def override_control_boar_list(self, remote_control_board_list: list):
        self.remote_control_board_list = remote_control_board_list

    def set_with_box(self,is_with_box): 
        self.is_with_box  = is_with_box
    
    def get_initial_configuration(self):
        # TODO change to be generica 
        base_position_init = np.array([-0.0489, 0, 0.65])
        base_orientationQuat_init = np.array([0, 0, 0, 1])
        base_orientationQuat_position_init = np.concatenate(
            (base_orientationQuat_init, base_position_init), axis=0
        )

        s_init = {
            "torso_pitch": -3,
            "torso_roll": 0,
            "torso_yaw": 0,
            "l_shoulder_pitch": -35.97,
            "l_shoulder_roll": 29.97,
            "l_shoulder_yaw": 0.006,
            "l_elbow": 50,
            "r_shoulder_pitch": -35.97,
            "r_shoulder_roll": 29.97,
            "r_shoulder_yaw": 0.006,
            "r_elbow": 50,
            "l_hip_pitch": 12,
            "l_hip_roll": 5,
            "l_hip_yaw": 0,
            "l_knee": -10,
            "l_ankle_pitch": -1.6,
            "l_ankle_roll": -5,
            "r_hip_pitch": 12,
            "r_hip_roll": 5,
            "r_hip_yaw": 0,
            "r_knee": -10,
            "r_ankle_pitch": -1.6,
            "r_ankle_roll": -5,
        }

        return [s_init, base_orientationQuat_position_init]

    def get_joint_limits(self):
        # TODO change to be generic 
        joints_limits = {
            "torso_pitch": [-0.3141592653589793, 0.7853981633974483],
            "torso_roll": [-0.4014257279586958, 0.4014257279586958],
            "torso_yaw": [-0.7504915783575618, 0.7504915783575618],
            "l_shoulder_pitch": [-1.53588974175501, 0.22689280275926285],
            "l_shoulder_roll": [0.20943951023931956, 2.8448866807507573],
            "l_shoulder_yaw": [-0.8726646259971648, 1.3962634015954636],
            "l_elbow": [0.3, 1.3089969389957472],
            "r_shoulder_pitch": [-1.53588974175501, 0.22689280275926285],
            "r_shoulder_roll": [0.20943951023931956, 2.8448866807507573],
            "r_shoulder_yaw": [-0.8726646259971648, 1.3962634015954636],
            "r_elbow": [0.3, 1.3089969389957472],
            "l_hip_pitch": [-0.7853981633974483, 2.007128639793479],
            "l_hip_roll": [-0.17453292519943295, 2.007128639793479],
            "l_hip_yaw": [-1.3962634015954636, 1.3962634015954636],
            "l_knee": [-1.2, -0.4],
            "l_ankle_pitch": [-0.7853981633974483, 0.7853981633974483],
            "l_ankle_roll": [-0.4363323129985824, 0.4363323129985824],
            "r_hip_pitch": [-0.7853981633974483, 2.007128639793479],
            "r_hip_roll": [-0.17453292519943295, 2.007128639793479],
            "r_hip_yaw": [-1.3962634015954636, 1.3962634015954636],
            "r_knee": [-1.2, -0.4],
            "r_ankle_pitch": [-0.7853981633974483, 0.7853981633974483],
            "r_ankle_roll": [-0.4363323129985824, 0.4363323129985824],
        }
        return joints_limits

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
        
        desired_knee = -1.0
        shoulder_roll = 0.251
        elbow = 0.616
        # Initialize empty lists for indexes
        knee_indexes = []
        shoulder_roll_indexes = []
        elbow_indexes = []
        hip_roll = []
        # Populate the lists with indexes if the keywords are found
        for i, joint in enumerate(self.joint_name_list):
            if "knee" in joint:
                knee_indexes.append(i)
            if "shoulder_roll" in joint:
                shoulder_roll_indexes.append(i)
            if "elbow" in joint:
                elbow_indexes.append(i)
            if "hip_roll" in joint: 
                hip_roll.append(i)

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
        if(knee_indexes):
            self.solver.subject_to(self.s[knee_indexes[0]] == desired_knee)
            self.solver.subject_to(self.s[knee_indexes[1]] == desired_knee)
        if(elbow_indexes):
            self.solver.subject_to(self.s[elbow_indexes[0]] == elbow)
            self.solver.subject_to(self.s[elbow_indexes[1]] == elbow)
        if(shoulder_roll_indexes):
            self.solver.subject_to(self.s[shoulder_roll_indexes[0]] == shoulder_roll)
            self.solver.subject_to(self.s[shoulder_roll_indexes[1]] == shoulder_roll)
        if(hip_roll):
            self.solver.subject_to(self.s[hip_roll[0]] == self.s[hip_roll[1]])
        self.solver.subject_to(H_left_foot[2, 3] == 0.0)
        self.solver.subject_to(H_right_foot[2, 3] == 0.0)
        self.solver.subject_to(quat_left_foot == reference_rotation)
        self.solver.subject_to(quat_right_foot == reference_rotation)
        self.solver.subject_to(quat_torso == reference_rotation)
        cost_function = cs.sumsqr(self.s)
        cost_function += cs.sumsqr(H_left_foot[:2, 3] - left_foot_pos[:2])
        cost_function += cs.sumsqr(H_right_foot[:2, 3] - right_foot_pos[:2])
        self.solver.minimize(cost_function)
        try:
            self.sol = self.solver.solve()
        except:
            return False, None, None, None 
        self.sol = self.solver.solve()
        s_return = np.array(self.sol.value(self.s))
        quat_base_opt = np.array(self.sol.value(self.quat_pose_b))
        H_b_return = quat_to_transf(quat_base_opt)
        xyz_rpy = self.matrix_to_rpy(H_b_return)
        return True, s_return, xyz_rpy, H_b_return

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
        ## We first start by ET
        # tempFileOut = tempfile.NamedTemporaryFile(mode="w+")
        # tempFileOut.write(copy.deepcopy(self.urdf_string))
        root = ET.fromstring(self.urdf_string)
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

        # Add box joint for payload lifting 
        # box_link,box_joint= self.add_box()
        # root.append(box_link)
        # root.append(box_joint)
        ## Adding the name to the collision and link visual
        for link in root.findall(".//link"):
            link_name = link.attrib.get("name")
            # print(link_name)
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
        # # Convert the XML tree to a string
        robot_urdf_string_original = ET.tostring(root)
        return robot_urdf_string_original

    def get_mujoco_model(self):
        urdf_string = self.get_mujoco_urdf_string()

        mujoco_model = mujoco.MjModel.from_xml_string(urdf_string)
        path_temp_xml = tempfile.NamedTemporaryFile(mode="w+")
        mujoco.mj_saveLastXML(path_temp_xml.name, mujoco_model)

        # Adding the Motors
        tree = ET.parse(path_temp_xml)
        root = tree.getroot()
        # Find all body elements
        bodies = root.findall('.//body')
        if self.is_with_box: self.add_box(root)
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
        ## TODO only for payload lifting 

        if self.is_with_box:
            equality_const = self.add_equality_constraint_box()
            mujoco_elem.append(equality_const)
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

    def add_box(self, mujoco_xml): 
        # Find the body element with the name "l_elbow_1"
        l_elbow_1_body = mujoco_xml.find('.//body[@name="l_elbow_1"]')

        if l_elbow_1_body is not None:
            # Create the new body element
            new_body = ET.Element('body')
            new_body.set('name', 'box_link_left')
            new_body.set('pos', '-0.01 -0.20 -0.34')

            # Create the new inertial element
            new_inertial = ET.Element('inertial')
            new_inertial.set('pos', '0 0 0')
            new_inertial.set('quat', '0.0 0.0 0.0 1.0')
            new_inertial.set('mass', '5')
            new_inertial.set('diaginertia', '0.0833 0.0418 0.0418')
            new_body.append(new_inertial)

            # Create the new geom element
            new_geom = ET.Element('geom')
            new_geom.set('name', 'base_link_collision')
            new_geom.set('size', '0.15 0.20 0.0425')
            new_geom.set('type', 'box')
            new_body.append(new_geom)

            # Append the new body element inside the existing "l_elbow_1" body element
            l_elbow_1_body.append(new_body)
    
    def add_equality_constraint_box(self): 
        # Create the equality element
        equality = ET.Element('equality')

        # Create the weld element
        weld = ET.Element('weld')
        weld.set('body1', 'r_elbow_1')
        weld.set('body2', 'box_link_left')
        weld.set('relpose', ' 0.02 0.2 -0.32 1 0 0 0 ')
        equality.append(weld)
        return equality

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
