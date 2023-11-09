from comodo.abstractClasses.controller import Controller
from comodo.TSIDController.TSIDParameterTuning import TSIDParameterTuning
import bipedal_locomotion_framework as blf
import numpy as np
import idyntree.bindings as iDynTree
import manifpy


class TSIDController(Controller):
    def __init__(self, frequency, robot_model) -> None:

        blf.text_logging.set_verbosity(blf.text_logging.Verbosity.Debug)
        self.controller = blf.tsid.QPTSID()
        self.gravity = iDynTree.Vector3()
        self.gravity.zero()
        self.gravity.setVal(2, -blf.math.StandardAccelerationOfGravitation)
        self.kindyn = robot_model.get_idyntree_kyndyn()
        self.robot_acceleration_variable_name = "robotAcceleration"
        self.variable_name_left_contact = "lf_wrench"
        self.variable_name_right_contact = "rf_wrench"
        self.joint_torques_variable_name = "joint_torques"
        self.max_number_contacts = 2
        self.number_fails = 0
        self.MAX_NUMBER_FAILS = 4
        super().__init__(frequency, robot_model)

    def define_kyndyn(self):
        self.H_left_foot_fun = self.robot_model.forward_kinematics_fun(
            self.robot_model.left_foot_frame
        )
        self.H_right_foot_fun = self.robot_model.forward_kinematics_fun(
            self.robot_model.right_foot_frame
        )

    def define_varible_handler(self):
        self.var_handler = blf.system.VariablesHandler()
        self.var_handler.add_variable(
            self.robot_acceleration_variable_name, self.robot_model.NDoF + 6
        )
        self.var_handler.add_variable(
            self.joint_torques_variable_name, self.robot_model.NDoF
        )
        self.var_handler.add_variable(self.variable_name_left_contact, 6)
        self.var_handler.add_variable(self.variable_name_right_contact, 6)

    def define_tasks(self, tsid_parameters: TSIDParameterTuning):

        self.define_kyndyn()
        self.controller_variable_handler = blf.parameters_handler.StdParametersHandler()
        self.controller_variable_handler.set_parameter_string(
            name="robot_acceleration_variable_name",
            value=self.robot_acceleration_variable_name,
        )
        self.controller_variable_handler.set_parameter_string(
            name="joint_torques_variable_name", value=self.joint_torques_variable_name
        )
        vector_string_contact = [
            self.variable_name_left_contact,
            self.variable_name_right_contact,
        ]
        self.controller_variable_handler.set_parameter_vector_string(
            name="contact_wrench_variables_name", value=vector_string_contact
        )
        self.controller.initialize(self.controller_variable_handler)
        self.define_varible_handler()

        contact_group_left = blf.parameters_handler.StdParametersHandler()
        contact_group_left.set_parameter_string(
            name="variable_name", value=self.variable_name_left_contact
        )
        contact_group_left.set_parameter_string(
            name="frame_name", value=self.robot_model.left_foot_frame
        )

        contact_group_right = blf.parameters_handler.StdParametersHandler()
        contact_group_right.set_parameter_string(
            name="variable_name", value=self.variable_name_right_contact
        )
        contact_group_right.set_parameter_string(
            name="frame_name", value=self.robot_model.right_foot_frame
        )

        ## CoM task --> Task aiming at tracking desired CoM trajectory CONSTRAINT
        self.CoM_Task = blf.tsid.CoMTask()
        self.CoM_task_name = "CoM_task"
        self.CoM_task_priority = 0
        self.CoM_param_handler = blf.parameters_handler.StdParametersHandler()
        self.CoM_param_handler.set_parameter_string(
            name="robot_acceleration_variable_name",
            value=self.robot_acceleration_variable_name,
        )
        self.CoM_param_handler.set_parameter_float(
            name="kp_linear", value=tsid_parameters.CoM_Kp
        )
        self.CoM_param_handler.set_parameter_float(
            name="kd_linear", value=tsid_parameters.CoM_Kd
        )
        self.CoM_Task.set_kin_dyn(self.kindyn)
        self.CoM_Task.initialize(param_handler=self.CoM_param_handler)

        postural_Kp = tsid_parameters.postural_Kp
        # postural_Kp = np.ones(self.robot_model.NDoF)
        Kd_postural = 0 * np.power(postural_Kp, 1 / 2)
        ## Joint regularziation task --> Task aiming at tracking desired joint trajectory COST
        self.joint_regularization_task = blf.tsid.JointTrackingTask()
        self.joint_regularization_param_handler = (
            blf.parameters_handler.StdParametersHandler()
        )
        self.joint_regularization_param_handler.set_parameter_string(
            name="robot_acceleration_variable_name",
            value=self.robot_acceleration_variable_name,
        )
        self.joint_regularization_param_handler.set_parameter_vector_float(
            name="kp", value=postural_Kp
        )
        self.joint_regularization_param_handler.set_parameter_vector_float(
            name="kd", value=Kd_postural
        )
        self.joint_regularization_task_name = "joint_regularization_task"
        self.joint_regularization_task_priority = 1
        self.joint_regularization_task_weight = tsid_parameters.postural_weight
        self.joint_regularization_task.set_kin_dyn(self.kindyn)
        self.joint_regularization_task.initialize(
            param_handler=self.joint_regularization_param_handler
        )

        ## Left wrench task --> Task aiming at ensuring left foot wrench feasibility
        self.left_foot_wrench_task = blf.tsid.FeasibleContactWrenchTask()
        self.left_foot_wrench_task_name = "left_foot_wrench_task"
        self.left_foot_wrench_priority = 0
        self.left_foot_param_handler = blf.parameters_handler.StdParametersHandler()
        self.left_foot_param_handler.set_parameter_string(
            name="variable_name", value=self.variable_name_left_contact
        )
        self.left_foot_param_handler.set_parameter_string(
            name="frame_name", value=self.robot_model.left_foot_frame
        )
        self.left_foot_param_handler.set_parameter_int(name="number_of_slices", value=2)
        self.left_foot_param_handler.set_parameter_float(
            name="static_friction_coefficient", value=1.0
        )
        self.left_foot_param_handler.set_parameter_vector_float(
            name="foot_limits_x", value=[-0.12, 0.12]
        )
        self.left_foot_param_handler.set_parameter_vector_float(
            name="foot_limits_y", value=[-0.05, 0.05]
        )
        self.left_foot_wrench_task.set_kin_dyn(self.kindyn)
        self.left_foot_wrench_task.initialize(
            param_handler=self.left_foot_param_handler
        )

        ## Right wrench task --> Task aiming at ensuring right foot wrench feasibility
        self.right_foot_wrench_task = blf.tsid.FeasibleContactWrenchTask()
        self.right_foot_wrench_task_name = "right_foot_wrench_task"
        self.right_foot_wrench_priority = 0
        self.right_foot_param_handler = blf.parameters_handler.StdParametersHandler()
        self.right_foot_param_handler.set_parameter_string(
            name="variable_name", value=self.variable_name_right_contact
        )
        self.right_foot_param_handler.set_parameter_string(
            name="frame_name", value=self.robot_model.right_foot_frame
        )
        self.right_foot_param_handler.set_parameter_int(
            name="number_of_slices", value=2
        )
        self.right_foot_param_handler.set_parameter_float(
            name="static_friction_coefficient", value=1.0
        )
        self.right_foot_param_handler.set_parameter_vector_float(
            name="foot_limits_x", value=[-0.12, 0.12]
        )
        self.right_foot_param_handler.set_parameter_vector_float(
            name="foot_limits_y", value=[-0.05, 0.05]
        )
        self.right_foot_wrench_task.set_kin_dyn(self.kindyn)
        self.right_foot_wrench_task.initialize(
            param_handler=self.right_foot_param_handler
        )

        ## Base"" Dynamics Task --> Base dynamic constraint
        self.base_dynamic_task = blf.tsid.BaseDynamicsTask()
        self.base_dynamic_task_name = "base_dynamic_task"
        self.base_dynamic_task_priority = 0
        self.base_dynamic_param_handler = blf.parameters_handler.StdParametersHandler()
        self.base_dynamic_param_handler.set_parameter_string(
            name="robot_acceleration_variable_name",
            value=self.robot_acceleration_variable_name,
        )
        self.base_dynamic_param_handler.set_parameter_int(
            name="max_number_of_contacts", value=self.max_number_contacts
        )
        self.base_dynamic_param_handler.set_group("CONTACT_0", contact_group_left)
        self.base_dynamic_param_handler.set_group("CONTACT_1", contact_group_right)
        self.base_dynamic_task.set_kin_dyn(self.kindyn)
        self.base_dynamic_task.initialize(param_handler=self.base_dynamic_param_handler)

        ## Joint Dynamics Task --> Base dynamic constraint
        self.joint_dynamic_task = blf.tsid.JointDynamicsTask()
        self.joint_dynamic_task_name = "joint_dynamic_task"
        self.joint_dynamic_task_priority = 0
        self.joint_dynamic_param_handler = blf.parameters_handler.StdParametersHandler()
        self.joint_dynamic_param_handler.set_parameter_string(
            name="robot_acceleration_variable_name",
            value=self.robot_acceleration_variable_name,
        )
        self.joint_dynamic_param_handler.set_parameter_string(
            name="joint_torques_variable_name", value=self.joint_torques_variable_name
        )
        self.joint_dynamic_param_handler.set_parameter_int(
            name="max_number_of_contacts", value=self.max_number_contacts
        )
        self.joint_dynamic_param_handler.set_group("CONTACT_0", contact_group_left)
        self.joint_dynamic_param_handler.set_group("CONTACT_1", contact_group_right)
        self.joint_dynamic_task.set_kin_dyn(self.kindyn)
        self.joint_dynamic_task.initialize(
            param_handler=self.joint_dynamic_param_handler
        )

        ## Left Foot Task
        self.left_foot_tracking_task = blf.tsid.SE3Task()
        self.left_foot_tracking_task_name = "left_foot_tracking_task"
        self.left_foot_tracking_task_priority = 0
        self.left_foot_tracking_task_param_handler = (
            blf.parameters_handler.StdParametersHandler()
        )
        self.left_foot_tracking_task_param_handler.set_parameter_string(
            name="robot_acceleration_variable_name",
            value=self.robot_acceleration_variable_name,
        )
        self.left_foot_tracking_task_param_handler.set_parameter_string(
            name="frame_name", value=self.robot_model.left_foot_frame
        )
        self.left_foot_tracking_task_param_handler.set_parameter_float(
            name="kp_linear", value=tsid_parameters.foot_tracking_task_kp_lin
        )
        self.left_foot_tracking_task_param_handler.set_parameter_float(
            name="kd_linear", value=tsid_parameters.foot_tracking_task_kd_lin
        )
        self.left_foot_tracking_task_param_handler.set_parameter_float(
            name="kp_angular", value=tsid_parameters.foot_tracking_task_kp_ang
        )
        self.left_foot_tracking_task_param_handler.set_parameter_float(
            name="kd_angular", value=tsid_parameters.foot_tracking_task_kd_ang
        )
        self.left_foot_tracking_task.set_kin_dyn(self.kindyn)
        self.left_foot_tracking_task.initialize(
            param_handler=self.left_foot_tracking_task_param_handler
        )

        ## Right Foot Task
        self.right_foot_tracking_task = blf.tsid.SE3Task()
        self.right_foot_tracking_task_name = "right_foot_tracking_task"
        self.right_foot_tracking_task_priority = 0
        self.right_foot_tracking_task_param_handler = (
            blf.parameters_handler.StdParametersHandler()
        )
        self.right_foot_tracking_task_param_handler.set_parameter_string(
            name="robot_acceleration_variable_name",
            value=self.robot_acceleration_variable_name,
        )
        self.right_foot_tracking_task_param_handler.set_parameter_string(
            name="frame_name", value=self.robot_model.right_foot_frame
        )
        self.right_foot_tracking_task_param_handler.set_parameter_float(
            name="kp_linear", value=tsid_parameters.foot_tracking_task_kp_lin
        )
        self.right_foot_tracking_task_param_handler.set_parameter_float(
            name="kd_linear", value=tsid_parameters.foot_tracking_task_kd_lin
        )
        self.right_foot_tracking_task_param_handler.set_parameter_float(
            name="kp_angular", value=tsid_parameters.foot_tracking_task_kp_ang
        )
        self.right_foot_tracking_task_param_handler.set_parameter_float(
            name="kd_angular", value=tsid_parameters.foot_tracking_task_kd_ang
        )
        self.right_foot_tracking_task.set_kin_dyn(self.kindyn)
        self.right_foot_tracking_task.initialize(
            param_handler=self.right_foot_tracking_task_param_handler
        )

        ## Root link task
        self.root_link_task = blf.tsid.SO3Task()
        self.root_link_task_name = "root_link_task"
        self.root_link_task_priority = 1
        self.root_link_task_weigth = tsid_parameters.root_tracking_task_weight
        self.root_link_task_param_handler = (
            blf.parameters_handler.StdParametersHandler()
        )
        self.root_link_task_param_handler.set_parameter_string(
            name="frame_name", value=self.robot_model.torso_link
        )
        self.root_link_task_param_handler.set_parameter_float(
            name="kp_angular", value=tsid_parameters.root_link_kp_ang
        )
        self.root_link_task_param_handler.set_parameter_float(
            name="kd_angular", value=tsid_parameters.root_link_kd_ang
        )
        self.root_link_task_param_handler.set_parameter_string(
            name="robot_acceleration_variable_name",
            value=self.robot_acceleration_variable_name,
        )
        self.root_link_task.set_kin_dyn(self.kindyn)
        self.root_link_task.initialize(param_handler=self.root_link_task_param_handler)

        ## Add tasks to the controller
        self.controller.add_task(
            self.CoM_Task, self.CoM_task_name, self.CoM_task_priority
        )
        self.controller.add_task(
            self.joint_regularization_task,
            self.joint_regularization_task_name,
            self.joint_regularization_task_priority,
            self.joint_regularization_task_weight,
        )
        self.controller.add_task(
            self.left_foot_wrench_task,
            self.left_foot_wrench_task_name,
            self.left_foot_wrench_priority,
        )
        self.controller.add_task(
            self.right_foot_wrench_task,
            self.right_foot_wrench_task_name,
            self.right_foot_wrench_priority,
        )
        self.controller.add_task(
            self.joint_dynamic_task,
            self.joint_dynamic_task_name,
            self.joint_dynamic_task_priority,
        )
        self.controller.add_task(
            self.base_dynamic_task,
            self.base_dynamic_task_name,
            self.base_dynamic_task_priority,
        )
        self.controller.add_task(
            self.left_foot_tracking_task,
            self.left_foot_tracking_task_name,
            self.left_foot_tracking_task_priority,
        )
        self.controller.add_task(
            self.right_foot_tracking_task,
            self.right_foot_tracking_task_name,
            self.right_foot_tracking_task_priority,
        )
        self.controller.add_task(
            self.root_link_task,
            self.root_link_task_name,
            self.root_link_task_priority,
            self.root_link_task_weigth,
        )
        self.controller.finalize(self.var_handler)

    def run(self):
        # TODO  understand which foot is in contact from the desired one, for now both of them are assumed to be in contact
        controller_succeded = self.controller.advance()
        self.torque = self.controller.get_output().joint_torques
        if controller_succeded:
            self.number_fails = 0
            controller_succeded_out = True
        else:
            self.number_fails = self.number_fails + 1
            controller_succeded_out = True
            self.torque = np.zeros(self.robot_model.NDoF)
            if self.number_fails > self.MAX_NUMBER_FAILS:
                controller_succeded_out = False

        return controller_succeded_out

    def get_fitness_parameters(self):
        print("WIP")

    def update_contacts(self, left_contact: bool, right_contact: bool):
        activate_left_foot_tracking_task = blf.tsid.SE3Task.Enable
        activate_right_foot_tracking_task = blf.tsid.SE3Task.Enable

        if left_contact:
            activate_left_foot_tracking_task = blf.tsid.SE3Task.Disable
        if right_contact:
            activate_right_foot_tracking_task = blf.tsid.SE3Task.Disable

        self.controller.get_task(self.left_foot_wrench_task_name).set_contact_active(
            left_contact
        )
        self.controller.get_task(self.right_foot_wrench_task_name).set_contact_active(
            right_contact
        )
        self.controller.get_task(
            self.right_foot_tracking_task_name
        ).set_task_controller_mode(activate_right_foot_tracking_task)
        self.controller.get_task(
            self.left_foot_tracking_task_name
        ).set_task_controller_mode(activate_left_foot_tracking_task)

    def compute_com_position(self):
        self.COM = iDynTree.Position()
        self.COM = self.kindyn.getCenterOfMassPosition()
        self.com_vel = iDynTree.Twist()
        self.com_vel = self.kindyn.getCenterOfMassVelocity()

    def update_desired_tasks(
        self,
        CoM_star,
        CoM_dot_star,
        CoM_dot_dot_star,
        L,
        L_Dot,
        wrench_left,
        wrench_right,
        s_desired,
    ):
        self.CoM_Task.set_set_point(CoM_star, CoM_dot_star, CoM_dot_dot_star)
        self.left_foot_regularization_task.set_set_point(wrench_left)
        self.right_foot_regularization_task.set_set_point(wrench_right)
        self.joint_regularization_task.set_set_point(s_desired)
        manif_rot = blf.conversions.to_manif_rot(self.H_b[:3, :3])
        self.root_link_task.set_set_point(
            manif_rot, manifpy.SO3Tangent.Zero(), manifpy.SO3Tangent.Zero()
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
    ):

        self.CoM_Task.set_set_point(com, dcom, ddcom)
        self.left_foot_tracking_task.set_set_point(
            left_foot_desired.transform,
            manifpy.SE3Tangent(left_foot_desired.mixed_velocity),
            manifpy.SE3Tangent(left_foot_desired.mixed_acceleration),
        )
        self.right_foot_tracking_task.set_set_point(
            right_foot_desired.transform,
            manifpy.SE3Tangent(right_foot_desired.mixed_velocity),
            manifpy.SE3Tangent(right_foot_desired.mixed_acceleration),
        )
        self.update_contacts(
            left_contact=left_foot_desired.is_in_contact,
            right_contact=right_foot_desired.is_in_contact,
        )
        self.joint_regularization_task.set_set_point(s_desired)
        # self.angular_momentum_task.set_set_point(np.zeros(3), np.zeros(3))
        wrench_desired_left = np.zeros(6)
        wrench_desired_right = np.zeros(6)
        # mass = self.robot_model.get_total_mass()
        # wrench_desired[2] = -(mass*9.81/2)
        # wrench_desiredp[]
        wrench_desired_left[:3] = wrenches_left
        wrench_desired_right[:3] = wrenches_right
        # self.left_foot_regularization_task.set_set_point(wrench_desired_left)
        # self.right_foot_regularization_task.set_set_point(wrench_desired_right)

    def update_com_task(self):

        angle = 0.2 * self.t
        CoM_des = self.COM
        # Calculate the x and y coordinates of the position vector using the radius and angle
        # print(angle)
        x = 0.0  # 0.004 * np.cos(angle)
        y = 0.004 * np.sin(angle)
        # CoM_des[0] += x
        # CoM_des[1] += y
        # print("CoM des", CoM_des)
        # print("CoM Measured",self.kindyn.getCenterOfMassPosition())
        # print("x",x)
        # print("y",y)
        self.CoM_Task.set_set_point(CoM_des.toNumPy(), np.zeros(3), np.zeros(3))

    def set_state_with_base(self, s, s_dot, H_b, w_b, t):
        self.s = s
        self.s_dot = s_dot
        self.t = t
        self.w_b = w_b
        self.H_b = H_b
        self.kindyn.setRobotState(self.H_b, self.s, self.w_b, self.s_dot, self.gravity)

    def set_desired_base_orientation(self):
        manif_rot = blf.conversions.to_manif_rot(self.H_b[:3, :3])
        self.root_link_task.set_set_point(
            manif_rot, manifpy.SO3Tangent.Zero(), manifpy.SO3Tangent.Zero()
        )
        self.angular_momentum_task.set_set_point(np.zeros(3), np.zeros(3))

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
