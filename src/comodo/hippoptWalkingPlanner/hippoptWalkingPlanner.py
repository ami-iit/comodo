from comodo.abstractClasses.planner import Planner
from comodo.hippoptWalkingPlanner.hippoptWalkingPlannerParameterTuning import HippoptWalkingParameterTuning
import logging

import casadi as cs
import idyntree.bindings as idyntree
import liecasadi
import numpy as np

import hippopt
import hippopt.robot_planning as hp_rp
import hippopt.turnkey_planners.humanoid_kinodynamic.planner as walking_planner
import hippopt.turnkey_planners.humanoid_kinodynamic.settings as walking_settings
import hippopt.turnkey_planners.humanoid_kinodynamic.variables as walking_variables
import hippopt.turnkey_planners.humanoid_pose_finder.planner as pose_finder
from hippopt.robot_planning.variables.contacts import ContactPointDescriptor
class HippoptWalkingPlanner(Planner):
    def __init__(self, robot_model) -> None:
        self.urdf_path = robot_model.urdf_string
        super().__init__(robot_model)

    def plan_trajectory(self):
        output = self.planner.solve()

        self.humanoid_states = [s.to_humanoid_state() for s in output.values.system]
        self.left_contact_points = [s.contact_points.left for s in self.humanoid_states]
        self.right_contact_points = [s.contact_points.right for s in self.humanoid_states]

    def from_robot_model_to_contact_descriptor(self, foot_frame): 
        return [
            ContactPointDescriptor(
                input_foot_frame=foot_frame,
                input_position_in_foot_frame=self.robot_model.corner_0,
            ),
            ContactPointDescriptor(
                input_foot_frame=foot_frame,
                input_position_in_foot_frame=self.robot_model.corner_1,
            ),
            ContactPointDescriptor(
                input_foot_frame=foot_frame,
                input_position_in_foot_frame=self.robot_model.corner_2,
            ),
            ContactPointDescriptor(
                input_foot_frame=foot_frame,
                input_position_in_foot_frame=self.robot_model.corner_3,
            ),
        ]
    def get_planner_settings(self, paramters:HippoptWalkingParameterTuning) -> walking_settings.Settings:
        settings = walking_settings.Settings()
        settings.robot_urdf = str(self.urdf_path)
        settings.joints_name_list = self.robot_model.joint_name_list
        number_of_joints = len(settings.joints_name_list)
        idyntree_model_loader = idyntree.ModelLoader()
        idyntree_model_loader.loadReducedModelFromString(
            self.robot_model.urdf_string, settings.joints_name_list
        )
        idyntree_model = idyntree_model_loader.model()
        settings.root_link = self.robot_model.base_link
        settings.horizon_length = paramters.horizon_length
        settings.time_step = paramters.step_length
        settings.contact_points = hp_rp.FeetContactPointDescriptors()
        settings.contact_points.left = self.from_robot_model_to_contact_descriptor(self.robot_model.left_foot_frame)
        settings.contact_points.right = self.from_robot_model_to_contact_descriptor(self.robot_model.right_foot_frame)
        
        print(settings.contact_points.right)
        settings.planar_dcc_height_multiplier = 10.0
        settings.dcc_gain = 40.0
        settings.dcc_epsilon = 0.005
        settings.static_friction = 0.3
        settings.maximum_velocity_control = [2.0, 2.0, 5.0]
        settings.maximum_force_derivative = [500.0, 500.0, 500.0]
        settings.maximum_angular_momentum = 5.0
        settings.minimum_com_height = 0.3
        settings.minimum_feet_lateral_distance = 0.1
        settings.maximum_feet_relative_height = 0.05
        settings.maximum_joint_positions = cs.inf * np.ones(number_of_joints)
        settings.minimum_joint_positions = -cs.inf * np.ones(number_of_joints)
        for i in range(number_of_joints):
            joint = idyntree_model.getJoint(i)
            if joint.hasPosLimits():
                settings.maximum_joint_positions[i] = joint.getMaxPosLimit(i)
                settings.minimum_joint_positions[i] = joint.getMinPosLimit(i)
        settings.maximum_joint_velocities = np.ones(number_of_joints) * 2.0
        settings.minimum_joint_velocities = np.ones(number_of_joints) * -2.0
        settings.joint_regularization_cost_weights = np.ones(number_of_joints)
        settings.joint_regularization_cost_weights[:3] = 0.1  # torso
        settings.joint_regularization_cost_weights[3:11] = 10.0  # arms
        settings.joint_regularization_cost_weights[11:] = 1.0  # legs
        settings.contacts_centroid_cost_multiplier = 0.0
        settings.com_linear_velocity_cost_weights = [10.0, 0.1, 1.0]
        settings.com_linear_velocity_cost_multiplier = 1.0
        settings.desired_frame_quaternion_cost_frame_name = self.robot_model.torso_link
        settings.desired_frame_quaternion_cost_multiplier = 200.0
        settings.base_quaternion_cost_multiplier = 50.0
        settings.base_quaternion_velocity_cost_multiplier = 0.001
        settings.joint_regularization_cost_multiplier = 10.0
        settings.force_regularization_cost_multiplier = 10.0
        settings.foot_yaw_regularization_cost_multiplier = 2000.0
        settings.swing_foot_height_cost_multiplier = 1000.0
        settings.contact_velocity_control_cost_multiplier = 5.0
        settings.contact_force_control_cost_multiplier = 0.0001
        settings.final_state_expression_type = hippopt.ExpressionType.subject_to
        settings.periodicity_expression_type = hippopt.ExpressionType.subject_to
        settings.casadi_function_options = {"cse": True}
        settings.casadi_opti_options = {"expand": True, "detect_simple_bounds": True}
        settings.casadi_solver_options = {
            "max_iter": 4000,
            "linear_solver": "MA27",
            "alpha_for_y": "dual-and-full",
            "fast_step_computation": "yes",
            "hessian_approximation": "limited-memory",
            "tol": 1e-3,
            "dual_inf_tol": 1000.0,
            "compl_inf_tol": 1e-2,
            "constr_viol_tol": 1e-4,
            "acceptable_tol": 10,
            "acceptable_iter": 2,
            "acceptable_compl_inf_tol": 1000.0,
            "warm_start_bound_frac": 1e-2,
            "warm_start_bound_push": 1e-2,
            "warm_start_mult_bound_push": 1e-2,
            "warm_start_slack_bound_frac": 1e-2,
            "warm_start_slack_bound_push": 1e-2,
            "warm_start_init_point": "yes",
            "required_infeasibility_reduction": 0.8,
            "perturb_dec_fact": 0.1,
            "max_hessian_perturbation": 100.0,
            "acceptable_obj_change_tol": 1e0,
        }

        return settings

    def get_pose_finder_settings(
        self,
        input_settings: walking_settings.Settings,
    ) -> pose_finder.Settings:
        number_of_joints = len(input_settings.joints_name_list)
        settings = pose_finder.Settings()
        settings.robot_urdf = input_settings.robot_urdf
        settings.joints_name_list = input_settings.joints_name_list

        settings.root_link = input_settings.root_link
        settings.desired_frame_quaternion_cost_frame_name = (
            input_settings.desired_frame_quaternion_cost_frame_name
        )

        settings.contact_points = input_settings.contact_points

        settings.relaxed_complementarity_epsilon = 0.0001
        settings.static_friction = input_settings.static_friction

        settings.maximum_joint_positions = input_settings.maximum_joint_positions
        settings.minimum_joint_positions = input_settings.minimum_joint_positions

        settings.joint_regularization_cost_weights = np.ones(number_of_joints)
        settings.joint_regularization_cost_weights[:3] = 0.1  # torso
        settings.joint_regularization_cost_weights[3:11] = 10.0  # arms
        settings.joint_regularization_cost_weights[11:] = 1.0  # legs

        settings.base_quaternion_cost_multiplier = 50.0
        settings.desired_frame_quaternion_cost_multiplier = 100.0
        settings.joint_regularization_cost_multiplier = 0.1
        settings.force_regularization_cost_multiplier = 0.2
        settings.com_regularization_cost_multiplier = 10.0
        settings.average_force_regularization_cost_multiplier = 10.0
        settings.point_position_regularization_cost_multiplier = 100.0
        settings.casadi_function_options = input_settings.casadi_function_options
        settings.casadi_opti_options = input_settings.casadi_opti_options
        settings.casadi_solver_options = {}

        return settings

    def get_visualizer_settings(
        self, input_settings: walking_settings.Settings
    ) -> hp_rp.HumanoidStateVisualizerSettings:
        output_viz_settings = hp_rp.HumanoidStateVisualizerSettings()
        output_viz_settings.robot_model = self.planner.get_adam_model()
        output_viz_settings.considered_joints = input_settings.joints_name_list
        output_viz_settings.contact_points = input_settings.contact_points
        output_viz_settings.terrain = input_settings.terrain
        output_viz_settings.working_folder = "./"
        return output_viz_settings

    def compute_state(
        self,
        input_settings: walking_settings.Settings,
        pf_input: pose_finder.Planner,
        desired_com_position: np.ndarray,
        desired_left_foot_pose: liecasadi.SE3,
        desired_right_foot_pose: liecasadi.SE3,
    ) -> hp_rp.HumanoidState:
        desired_joints = self.robot_model.s_init
        assert len(input_settings.joints_name_list) == len(desired_joints)

        pf_ref = pose_finder.References(
            contact_point_descriptors=self.pf_settings.contact_points,
            number_of_joints=len(desired_joints),
        )

        pf_ref.state.com = desired_com_position
        pf_ref.state.contact_points.left = (
            hp_rp.FootContactState.from_parent_frame_transform(
                descriptor=input_settings.contact_points.left,
                transform=desired_left_foot_pose,
            )
        )
        pf_ref.state.contact_points.right = (
            hp_rp.FootContactState.from_parent_frame_transform(
                descriptor=input_settings.contact_points.right,
                transform=desired_right_foot_pose,
            )
        )

        pf_ref.state.kinematics.base.quaternion_xyzw = (
            liecasadi.SO3.Identity().as_quat().coeffs()
        )

        pf_ref.frame_quaternion_xyzw = liecasadi.SO3.Identity().as_quat().coeffs()

        pf_ref.state.kinematics.joints.positions = desired_joints

        pf_input.set_references(pf_ref)

        output_pf = pf_input.solve()
        return output_pf.values.state

    def compute_initial_state(
        self,
        input_settings: walking_settings.Settings,
        pf_input: pose_finder.Planner,
        contact_guess: hp_rp.FeetContactPhasesDescriptor,
    ) -> walking_variables.ExtendedHumanoidState:
        desired_left_foot_pose = contact_guess.left[0].transform
        desired_right_foot_pose = contact_guess.right[0].transform
        desired_com_position = (
            desired_left_foot_pose.translation() + desired_right_foot_pose.translation()
        ) / 2.0
        desired_com_position[2] = self.robot_model.compute_com_init()[2] # TODO
        output_pf = self.compute_state(
            input_settings=input_settings,
            pf_input=pf_input,
            desired_com_position=desired_com_position,
            desired_left_foot_pose=desired_left_foot_pose,
            desired_right_foot_pose=desired_right_foot_pose,
        )

        output_state = walking_variables.ExtendedHumanoidState()
        output_state.contact_points = output_pf.contact_points
        output_state.kinematics = output_pf.kinematics
        output_state.com = output_pf.com

        output_state.centroidal_momentum = np.zeros((6, 1))

        return output_state

    def compute_middle_state(
        self,
        input_settings: walking_settings.Settings,
        pf_input: pose_finder.Planner,
        contact_guess: hp_rp.FeetContactPhasesDescriptor,
    ) -> hp_rp.HumanoidState:
        desired_left_foot_pose = contact_guess.left[1].transform
        desired_right_foot_pose = contact_guess.right[0].transform
        desired_com_position = (
            desired_left_foot_pose.translation() + desired_right_foot_pose.translation()
        ) / 2.0
        desired_com_position[2] = self.robot_model.compute_com_init()[2]  # TODO
        return self.compute_state(
            input_settings=input_settings,
            pf_input=pf_input,
            desired_com_position=desired_com_position,
            desired_left_foot_pose=desired_left_foot_pose,
            desired_right_foot_pose=desired_right_foot_pose,
        )

    def compute_final_state(
        self,
        input_settings: walking_settings.Settings,
        pf_input: pose_finder.Planner,
        contact_guess: hp_rp.FeetContactPhasesDescriptor,
    ) -> hp_rp.HumanoidState:
        desired_left_foot_pose = contact_guess.left[1].transform
        desired_right_foot_pose = contact_guess.right[1].transform
        desired_com_position = (
            desired_left_foot_pose.translation() + desired_right_foot_pose.translation()
        ) / 2.0
        desired_com_position[2] = self.robot_model.compute_com_init()[2]  # TODO
        return self.compute_state(
            input_settings=input_settings,
            pf_input=pf_input,
            desired_com_position=desired_com_position,
            desired_left_foot_pose=desired_left_foot_pose,
            desired_right_foot_pose=desired_right_foot_pose,
        )

    def get_references(
        self,
        input_settings: walking_settings.Settings,
        desired_states: list[hp_rp.HumanoidState],
    ) -> list[walking_variables.References]:
        output_list = []

        for i in range(input_settings.horizon_length):
            output_reference = walking_variables.References(
                number_of_joints=len(input_settings.joints_name_list),
                number_of_points_left=len(input_settings.contact_points.left),
                number_of_points_right=len(input_settings.contact_points.right),
            )

            output_reference.contacts_centroid_cost_weights = [100, 100, 10]
            output_reference.contacts_centroid = [0.3, 0.0, 0.0]
            output_reference.joint_regularization = desired_states[
                i
            ].kinematics.joints.positions
            output_reference.com_linear_velocity = [0.1, 0.0, 0.0]
            output_list.append(output_reference)

        return output_list

    def initialize_planner(self, paramters:HippoptWalkingParameterTuning):
        logging.basicConfig(level=logging.INFO)

        self.planner_settings = self.get_planner_settings(paramters)
        self.planner = walking_planner.Planner(settings=self.planner_settings)

        self.pf_settings = self.get_pose_finder_settings(
            input_settings=self.planner_settings
        )
        pf = pose_finder.Planner(settings=self.pf_settings)

        horizon = self.planner_settings.horizon_length * self.planner_settings.time_step

        step_length = paramters.step_length

        contact_phases_guess = hp_rp.FeetContactPhasesDescriptor()
        contact_phases_guess.left = [
            hp_rp.FootContactPhaseDescriptor(
                transform=liecasadi.SE3.from_translation_and_rotation(
                    np.array([0.0, 0.1, 0.0]), liecasadi.SO3.Identity()
                ),
                mid_swing_transform=liecasadi.SE3.from_translation_and_rotation(
                    np.array([step_length / 2, 0.1, 0.05]), liecasadi.SO3.Identity()
                ),
                force=np.array([0, 0, 100.0]),
                activation_time=None,
                deactivation_time=horizon / 6.0,
            ),
            hp_rp.FootContactPhaseDescriptor(
                transform=liecasadi.SE3.from_translation_and_rotation(
                    np.array([step_length, 0.1, 0.0]), liecasadi.SO3.Identity()
                ),
                mid_swing_transform=None,
                force=np.array([0, 0, 100.0]),
                activation_time=horizon / 3.0,
                deactivation_time=None,
            ),
        ]

        contact_phases_guess.right = [
            hp_rp.FootContactPhaseDescriptor(
                transform=liecasadi.SE3.from_translation_and_rotation(
                    np.array([step_length / 2, -0.1, 0.0]), liecasadi.SO3.Identity()
                ),
                mid_swing_transform=liecasadi.SE3.from_translation_and_rotation(
                    np.array([step_length, -0.1, 0.05]), liecasadi.SO3.Identity()
                ),
                force=np.array([0, 0, 100.0]),
                activation_time=None,
                deactivation_time=horizon * 2.0 / 3.0,
            ),
            hp_rp.FootContactPhaseDescriptor(
                transform=liecasadi.SE3.from_translation_and_rotation(
                    np.array([1.5 * step_length, -0.1, 0.0]), liecasadi.SO3.Identity()
                ),
                mid_swing_transform=None,
                force=np.array([0, 0, 100.0]),
                activation_time=horizon * 5.0 / 6.0,
                deactivation_time=None,
            ),
        ]

        initial_state = self.compute_initial_state(
            input_settings=self.planner_settings,
            pf_input=pf,
            contact_guess=contact_phases_guess,
        )

        final_state = self.compute_final_state(
            input_settings=self.planner_settings,
            pf_input=pf,
            contact_guess=contact_phases_guess,
        )
        final_state.centroidal_momentum = np.zeros((6, 1))

        middle_state = self.compute_middle_state(
            input_settings=self.planner_settings,
            pf_input=pf,
            contact_guess=contact_phases_guess,
        )

        first_half_guess_length = self.planner_settings.horizon_length // 2
        first_half_guess = hp_rp.humanoid_state_interpolator(
            initial_state=initial_state,
            final_state=middle_state,
            contact_phases=contact_phases_guess,
            contact_descriptor=self.planner_settings.contact_points,
            number_of_points=first_half_guess_length,
            dt=self.planner_settings.time_step,
        )

        second_half_guess_length = (
            self.planner_settings.horizon_length - first_half_guess_length
        )
        second_half_guess = hp_rp.humanoid_state_interpolator(
            initial_state=middle_state,
            final_state=final_state,
            contact_phases=contact_phases_guess,
            contact_descriptor=self.planner_settings.contact_points,
            number_of_points=second_half_guess_length,
            dt=self.planner_settings.time_step,
            t0=first_half_guess_length * self.planner_settings.time_step,
        )

        self.guess = first_half_guess + second_half_guess
        self.references = self.get_references(
            input_settings=self.planner_settings,
            desired_states=self.guess,
        )

        self.planner.set_references(self.references)
        planner_guess = self.planner.get_initial_guess()
        planner_guess.system = [
            walking_variables.ExtendedHumanoid.from_humanoid_state(s) for s in self.guess
        ]
        planner_guess.initial_state = initial_state
        planner_guess.final_state = final_state
        self.planner.set_initial_guess(planner_guess)

    def visualizer_init(self):
        self.visualizer_settings = self.get_visualizer_settings(
            input_settings=self.planner_settings
        )
        self.visualizer = hp_rp.HumanoidStateVisualizer(settings=self.visualizer_settings)

    def visualize_state(self, state):
        self.visualizer.visualize(
            states=state,
            timestep_s=self.planner_settings.time_step,
            time_multiplier=1.0,
            save=False
        )
