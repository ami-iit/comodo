from comodo.abstractClasses.controller import Controller
from comodo.inverseKinematics.inverseKinematicsParamTuning import (
    InverseKinematicsParamTuning,
)
import bipedal_locomotion_framework as blf
import numpy as np
import idyntree.bindings as iDynTree
import manifpy
import copy
import datetime
from scipy.spatial.transform import Rotation


class InverseKinematics(Controller):
    def __init__(self, frequency, robot_model) -> None:
        self.gravity = iDynTree.Vector3()
        self.gravity.zero()
        self.gravity.setVal(2, -blf.math.StandardAccelerationOfGravitation)
        self.kindyn = robot_model.get_idyntree_kyndyn()
        self.com_fun = robot_model.CoM_position_fun()
        super().__init__(frequency, robot_model)

    def define_tasks(self, parameters: InverseKinematicsParamTuning):
        self.robot_velocity_name = "robotVelocity"

        # # Set the parameters ik
        qp_ik_param_handler = blf.parameters_handler.StdParametersHandler()
        qp_ik_param_handler.set_parameter_string(
            name="robot_velocity_variable_name", value=self.robot_velocity_name
        )

        # Initialize the QP inverse kinematics
        self.solver = blf.ik.QPInverseKinematics()
        self.solver.initialize(handler=qp_ik_param_handler)

        qp_ik_var_handler = blf.system.VariablesHandler()
        qp_ik_var_handler.add_variable("robotVelocity", self.robot_model.NDoF + 6)

        # ## COM
        # Set the parameters
        com_param_handler = blf.parameters_handler.StdParametersHandler()
        com_param_handler.set_parameter_string(
            name="robot_velocity_variable_name", value=self.robot_velocity_name
        )
        com_param_handler.set_parameter_float(
            name="kp_linear", value=parameters.com_linear
        )
        com_param_handler.set_parameter_vector_bool(
            name="mask", value=[True, True, False]
        )
        # # Initialize the task
        com_task = blf.ik.CoMTask()
        com_task.set_kin_dyn(self.kindyn)
        com_task.initialize(param_handler=com_param_handler)
        com_var_handler = blf.system.VariablesHandler()
        com_var_handler.add_variable(
            self.robot_velocity_name, self.robot_model.NDoF + 6
        )
        com_task.set_variables_handler(variables_handler=com_var_handler)
        self.solver.add_task(task=com_task, task_name="COM", priority=0)

        # ## Joint tracking
        # # Set the parameters
        joint_tracking_param_handler = blf.parameters_handler.StdParametersHandler()
        joint_tracking_param_handler.set_parameter_string(
            name="robot_velocity_variable_name", value=self.robot_velocity_name
        )
        joint_tracking_param_handler.set_parameter_vector_float(
            name="kp",
            value=parameters.kp_joint_tracking * np.ones(self.robot_model.NDoF),
        )

        # # Initialize the task
        joint_tracking_task = blf.ik.JointTrackingTask()
        joint_tracking_task.set_kin_dyn(self.kindyn)
        joint_tracking_task.initialize(param_handler=joint_tracking_param_handler)
        joint_tracking_var_handler = blf.system.VariablesHandler()
        joint_tracking_var_handler.add_variable(
            self.robot_velocity_name, self.robot_model.NDoF + 6
        )
        if isinstance(parameters.weigth_joint, list):
            # TODO check the size
            weigth_joint = parameters.weigth_joint
        else:
            weigth_joint = parameters.weigth_joint * np.ones(self.robot_model.NDoF)
        self.solver.add_task(
            task=joint_tracking_task,
            task_name="JOINT_REGULARIZATION",
            priority=1,
            weight_provider=blf.bindings.system.ConstantWeightProvider(weigth_joint),
        )

        # # Set the parameters CHEST
        chest_handler = blf.parameters_handler.StdParametersHandler()
        chest_handler.set_parameter_string(
            name="robot_velocity_variable_name", value=self.robot_velocity_name
        )
        chest_handler.set_parameter_string(name="frame_name", value="chest")
        chest_handler.set_parameter_float(
            name="kp_angular", value=parameters.chest_angular
        )
        # # Initialize the task
        chest_task = blf.ik.SO3Task()
        chest_task.set_kin_dyn(self.kindyn)
        chest_task.initialize(param_handler=chest_handler)
        chest_var_handler = blf.system.VariablesHandler()
        chest_var_handler.add_variable(
            self.robot_velocity_name, self.robot_model.NDoF + 6
        )
        chest_task.set_variables_handler(variables_handler=chest_var_handler)
        self.solver.add_task(
            task=chest_task,
            task_name="CHEST",
            priority=1,
            weight_provider=blf.bindings.system.ConstantWeightProvider(
                [10.0, 10.0, 25.0]
            ),
        )

        # # Set the parameters ROOT
        root_handler = blf.parameters_handler.StdParametersHandler()
        root_handler.set_parameter_string(
            name="robot_velocity_variable_name", value=self.robot_velocity_name
        )
        root_handler.set_parameter_string(name="frame_name", value="root_link")
        root_handler.set_parameter_float(name="kp_linear", value=parameters.root_linear)
        root_handler.set_parameter_vector_bool(name="mask", value=[False, False, True])

        # # Initialize the task
        root_task = blf.ik.R3Task()
        root_task.set_kin_dyn(self.kindyn)
        root_task.initialize(param_handler=root_handler)
        root_var_handler = blf.system.VariablesHandler()
        root_var_handler.add_variable(
            self.robot_velocity_name, self.robot_model.NDoF + 6
        )
        root_task.set_variables_handler(variables_handler=root_var_handler)
        self.solver.add_task(task=root_task, task_name="ROOT_TASK", priority=0)

        # ## SE3 LEFT FOOT
        # # Set the parameters
        left_foot_param_handler = blf.parameters_handler.StdParametersHandler()
        left_foot_param_handler.set_parameter_string(
            name="robot_velocity_variable_name", value=self.robot_velocity_name
        )
        left_foot_param_handler.set_parameter_string(name="frame_name", value="l_sole")
        left_foot_param_handler.set_parameter_float(
            name="kp_linear", value=parameters.foot_linear
        )
        left_foot_param_handler.set_parameter_float(
            name="kp_angular", value=parameters.foot_angular
        )

        # # Initialize the task
        left_foot_taks = blf.ik.SE3Task()
        left_foot_taks.set_kin_dyn(self.kindyn)
        left_foot_taks.initialize(param_handler=left_foot_param_handler)
        left_foot_var_handler = blf.system.VariablesHandler()
        left_foot_var_handler.add_variable(
            self.robot_velocity_name, self.robot_model.NDoF + 6
        )
        left_foot_taks.set_variables_handler(variables_handler=left_foot_var_handler)
        self.solver.add_task(task=left_foot_taks, task_name="LEFT_FOOT", priority=0)

        # ## SE3 RIGTH FOOT
        # # Set the parameters
        rigth_foot_param_handler = blf.parameters_handler.StdParametersHandler()
        rigth_foot_param_handler.set_parameter_string(
            name="robot_velocity_variable_name", value=self.robot_velocity_name
        )
        rigth_foot_param_handler.set_parameter_string(name="frame_name", value="r_sole")
        rigth_foot_param_handler.set_parameter_float(
            name="kp_linear", value=parameters.foot_linear
        )
        rigth_foot_param_handler.set_parameter_float(
            name="kp_angular", value=parameters.foot_angular
        )

        # # Initialize the task
        rigth_foot_taks = blf.ik.SE3Task()
        rigth_foot_taks.set_kin_dyn(self.kindyn)
        rigth_foot_taks.initialize(param_handler=rigth_foot_param_handler)
        rigth_foot_var_handler = blf.system.VariablesHandler()
        rigth_foot_var_handler.add_variable(
            self.robot_velocity_name, self.robot_model.NDoF + 6
        )
        rigth_foot_taks.set_variables_handler(variables_handler=rigth_foot_var_handler)
        self.solver.add_task(task=rigth_foot_taks, task_name="RIGHT_FOOT", priority=0)

        self.solver.finalize(qp_ik_var_handler)
        self.define_zmp_controller(parameters=parameters)

    def define_zmp_controller(self, parameters: InverseKinematicsParamTuning):
        self.zmp_controller = blf.simplified_model_controllers.CoMZMPController()
        controller_param_handler = blf.parameters_handler.StdParametersHandler()
        controller_param_handler.set_parameter_vector_float(
            "zmp_gain", parameters.zmp_gain
        )
        controller_param_handler.set_parameter_vector_float(
            "com_gain", parameters.com_gain
        )
        self.zmp_controller.initialize(controller_param_handler)

    def define_integrator(self):
        self.system = blf.continuous_dynamical_system.FloatingBaseSystemKinematics()
        manif_rot = blf.conversions.to_manif_rot(self.H_b[:3, :3])
        self.system.set_state(
            (
                self.H_b[:3, 3],
                manif_rot,
                self.s,
            )
        )

        self.integrator = (
            blf.continuous_dynamical_system.FloatingBaseSystemKinematicsForwardEulerIntegrator()
        )
        self.integrator.set_dynamical_system(self.system)
        self.integrator.set_integration_step(self.frequency)

    def run(self):
        controller_succeded = self.solver.advance()
        return controller_succeded

    def pose_to_matrix(self, position, quaternion):
        """
        Convert position (xyz) and quaternion (wxyz) to a transformation matrix.

        Parameters:
        - position (list or numpy array): 3D position [x, y, z]
        - quaternion (list or numpy array): Quaternion [w, x, y, z]

        Returns:
        - transformation_matrix (numpy array): 4x4 transformation matrix
        """
        
        # Create a 3x3 rotation matrix from the quaternion
        rotation_matrix = Rotation.from_quat(quaternion.coeffs()).as_matrix()

        # Create a 4x4 transformation matrix
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = position

        return transformation_matrix

    def get_output(self):
        # GET IK RESULTS
        base_velocity = self.solver.get_output().base_velocity
        joint_vel = self.solver.get_output().joint_velocity
        self.system.set_control_input((base_velocity.coeffs(), joint_vel))
        self.integrator.integrate(
            datetime.timedelta(seconds=0), datetime.timedelta(seconds=self.frequency)
        )
        base_position, base_rotation, joint_position = self.integrator.get_solution()
        return joint_position

    def update_state(self):
        base_velocity = self.solver.get_output().base_velocity
        joint_vel = self.solver.get_output().joint_velocity
        base_position, base_rotation, joint_position = self.integrator.get_solution()
        H_b = self.pose_to_matrix(base_position, base_rotation)
        self.kindyn.setRobotState(
            H_b, joint_position, base_velocity.coeffs(), joint_vel, self.gravity
        )

    def update_task_references_mpc(
        self,
        com,
        dcom,
        ddcom,
        left_foot_desired,
        right_foot_desired,
        s_desired,
        wrenches_left,
        wrenches_right,
        H_omega,
    ):
        self.solver.get_task("LEFT_FOOT").set_set_point(
            left_foot_desired.transform,
            manifpy.SE3Tangent(left_foot_desired.mixed_velocity),
        )
        self.solver.get_task("RIGHT_FOOT").set_set_point(
            right_foot_desired.transform,
            manifpy.SE3Tangent(right_foot_desired.mixed_velocity),
        )
        self.solver.get_task("JOINT_REGULARIZATION").set_set_point(
            s_desired, np.zeros(self.robot_model.NDoF)
        )
        zmp_des = self.compute_zmp_wrench(wrenches_left, wrenches_right)
        com_vel_control = np.zeros(3)
        # # define the input
        controller_input = blf.simplified_model_controllers.CoMZMPControllerInput()
        controller_input.desired_CoM_position = com[:2]
        controller_input.desired_CoM_velocity = dcom[:2]
        controller_input.desired_ZMP_position = zmp_des
        controller_input.CoM_position = self.com[:2]
        controller_input.ZMP_position = self.zmp
        controller_input.angle = 0

        self.zmp_controller.set_input(controller_input)
        self.zmp_controller.advance()
        com_vel_control[:2] = self.zmp_controller.get_output()
        com_vel_control[2] = dcom[2]
        com_new = self.com[:, 0] + com_vel_control * self.frequency
        com_vel_control = dcom
        self.solver.get_task("COM").set_set_point(com, com_vel_control)

    def set_state_with_base(self, s, s_dot, H_b, w_b, t):
        self.s = s
        self.s_dot = s_dot
        self.t = t
        self.w_b = w_b
        self.H_b = H_b
        self.kindyn.setRobotState(self.H_b, self.s, self.w_b, self.s_dot, self.gravity)

    def set_desired_base_orientation(self):
        manif_rot = blf.conversions.to_manif_rot(self.H_b[:3, :3])
        self.solver.get_task("CHEST").set_set_point(
            manif_rot, manifpy.SO3Tangent.Zero()
        )
        self.solver.get_task("ROOT_TASK").set_set_point(self.H_b[:3, 3], np.zeros(3))

    def set_state(self, s, s_dot, t):
        self.s = s
        self.s_dot = s_dot
        self.t = t
        contact_frames_pose = {
            self.robot_model.left_foot_frame: np.eye(4),
            self.robot_model.right_foot_frame: np.eye(4),
        }
        contact_frames_list = [
            self.robot_model.left_foot_frame,
            self.robot_model.right_foot_frame,
        ]
        self.H_b = self.robot_model.get_base_pose_from_contacts(
            self.s, contact_frames_pose
        )
        self.w_b = iDynTree.Twist()
        self.w_b = self.robot_model.get_base_velocity_from_contacts(
            self.H_b, self.s, self.s_dot, contact_frames_list
        )
        self.kindyn.setRobotState(self.H_b, self.s, self.w_b, self.s_dot, self.gravity)

    def update_com(self, H_b, s):
        self.com = np.array(self.com_fun(H_b, s))

    ##Def abstract methods
    def get_fitness_parameters(self):
        print("to be implemented")
        pass

    def compute_zmp(
        self,
        left_wrench: np.array,
        right_wrench: np.array,
    ) -> np.array:
        """Auxiliary function to retrieve the zero-moment point from the feet wrenches."""

        # Compute local zmps (one per foot) from the foot wrenches
        LF_r_zmp_L = [-left_wrench[4] / left_wrench[2], left_wrench[3] / left_wrench[2]]
        RF_r_zmp_R = [
            -right_wrench[4] / right_wrench[2],
            right_wrench[3] / right_wrench[2],
        ]

        # Express the local zmps in homogeneous coordinates
        LF_r_zmp_L_homogenous = np.array([LF_r_zmp_L[0], LF_r_zmp_L[1], 0, 1])
        RF_r_zmp_R_homogenous = np.array([RF_r_zmp_R[0], RF_r_zmp_R[1], 0, 1])

        # Retrieve the global transform of the feet frames
        W_H_LF = (
            self.kindyn.getWorldTransform(self.robot_model.left_foot_frame)
            .asHomogeneousTransform()
            .toNumPy()
        )
        W_H_RF = (
            self.kindyn.getWorldTransform(self.robot_model.right_foot_frame)
            .asHomogeneousTransform()
            .toNumPy()
        )

        # Express the local zmps (one per foot) in a common reference frame (i.e. the world frame)
        W_r_zmp_L_hom = W_H_LF @ LF_r_zmp_L_homogenous
        W_r_zmp_L = W_r_zmp_L_hom[0:2]
        W_r_zmp_R_hom = W_H_RF @ RF_r_zmp_R_homogenous
        W_r_zmp_R = W_r_zmp_R_hom[0:2]

        # Compute the global zmp as a weighted mean of the local zmps (one per foot)
        # expressed in a common reference frame (i.e. the world frame)
        if np.linalg.norm(left_wrench) < 1e-6:
            # if the left foot is not in contact, the zmp is the rigth
            W_r_zmp_global = W_r_zmp_R
        elif np.linalg.norm(right_wrench) < 1e-6:
            W_r_zmp_global = W_r_zmp_L
        else:
            W_r_zmp_global = W_r_zmp_L * (
                left_wrench[2] / (left_wrench[2] + right_wrench[2])
            ) + W_r_zmp_R * (right_wrench[2] / (left_wrench[2] + right_wrench[2]))

        self.zmp = W_r_zmp_global
        return W_r_zmp_global

    def compute_zmp_wrench(self, left_force, right_force) -> np.array:
        left_wrench = np.zeros(6)
        left_wrench[:3] = left_force
        right_wrench = np.zeros(6)
        right_wrench[:3] = right_force
        """Auxiliary function to retrieve the zero-moment point from the feet wrenches."""

        # Compute local zmps (one per foot) from the foot wrenches
        LF_r_zmp_L = [-left_wrench[4] / left_wrench[2], left_wrench[3] / left_wrench[2]]
        RF_r_zmp_R = [
            -right_wrench[4] / right_wrench[2],
            right_wrench[3] / right_wrench[2],
        ]

        # Express the local zmps in homogeneous coordinates
        LF_r_zmp_L_homogenous = np.array([LF_r_zmp_L[0], LF_r_zmp_L[1], 0, 1])
        RF_r_zmp_R_homogenous = np.array([RF_r_zmp_R[0], RF_r_zmp_R[1], 0, 1])

        # Retrieve the global transform of the feet frames
        W_H_LF = (
            self.kindyn.getWorldTransform(self.robot_model.left_foot_frame)
            .asHomogeneousTransform()
            .toNumPy()
        )
        W_H_RF = (
            self.kindyn.getWorldTransform(self.robot_model.right_foot_frame)
            .asHomogeneousTransform()
            .toNumPy()
        )

        # Express the local zmps (one per foot) in a common reference frame (i.e. the world frame)
        W_r_zmp_L_hom = W_H_LF @ LF_r_zmp_L_homogenous
        W_r_zmp_L = W_r_zmp_L_hom[0:2]
        W_r_zmp_R_hom = W_H_RF @ RF_r_zmp_R_homogenous
        W_r_zmp_R = W_r_zmp_R_hom[0:2]

        # Compute the global zmp as a weighted mean of the local zmps (one per foot)
        # expressed in a common reference frame (i.e. the world frame)
        if np.linalg.norm(left_wrench) < 1e-6:
            # if the left foot is not in contact, the zmp is the rigth
            W_r_zmp_global = W_r_zmp_R
        elif np.linalg.norm(right_wrench) < 1e-6:
            W_r_zmp_global = W_r_zmp_L
        else:
            W_r_zmp_global = W_r_zmp_L * (
                left_wrench[2] / (left_wrench[2] + right_wrench[2])
            ) + W_r_zmp_R * (right_wrench[2] / (left_wrench[2] + right_wrench[2]))

        return W_r_zmp_global
