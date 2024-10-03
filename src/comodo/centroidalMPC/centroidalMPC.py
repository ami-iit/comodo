from comodo.abstractClasses.planner import Planner
from comodo.centroidalMPC.footPositionPlanner import FootPositionPlanner
import bipedal_locomotion_framework as blf
from datetime import timedelta
import idyntree.bindings as iDynTree
import numpy as np
import matplotlib.pyplot as plt
from comodo.centroidalMPC.mpcParameterTuning import MPCParameterTuning


class CentroidalMPC(Planner):
    def __init__(self, robot_model, step_length, frequency_ms=100):
        self.dT = timedelta(milliseconds=frequency_ms)
        self.dT_in_seconds = frequency_ms / 1000
        self.contact_planner = FootPositionPlanner(
            robot_model=robot_model,
            dT=timedelta(seconds=self.dT_in_seconds),
            step_length=step_length,
        )
        self.centroidal_mpc = blf.reduced_model_controllers.CentroidalMPC()
        scaling = 1.0
        scalingPos = 1.2
        scalingPosY = 0.0
        self.gravity = iDynTree.Vector3()
        self.gravity.zero()
        self.gravity.setVal(2, -blf.math.StandardAccelerationOfGravitation)
        self.contact_planner.set_scaling_parameters(
            scaling=scaling, scalingPos=scalingPos, scalingPosY=scalingPosY
        )
        self.kindyn = robot_model.get_idyntree_kyndyn()
        super().__init__(robot_model)

    def get_frequency_seconds(self):
        return self.dT_in_seconds

    def plan_trajectory(self):
        com = self.kindyn.getCenterOfMassPosition()
        dcom = self.kindyn.getCenterOfMassVelocity()
        Jcm = iDynTree.MatrixDynSize(6, 6 + self.robot_model.NDoF)
        self.kindyn.getCentroidalTotalMomentumJacobian(Jcm)
        nu = np.concatenate((self.w_b, self.s_dot))
        H = Jcm.toNumPy() @ nu
        dcom = self.centroidal_integrator.get_solution()[1]
        angular_mom = self.centroidal_integrator.get_solution()[2]
        self.centroidal_mpc.set_state(com.toNumPy(), dcom, angular_mom)

        self.centroidal_mpc.set_reference_trajectory(
            self.com_traj, self.angular_mom_trak
        )
        self.centroidal_mpc.set_contact_phase_list(self.contact_phase_list)
        success = self.centroidal_mpc.advance()
        if success:
            self.centroidal_dynamics.set_control_input(
                (self.centroidal_mpc.get_output().contacts, np.zeros(6))
            )
            self.centroidal_integrator.integrate(timedelta(0), self.dT)
        return success

    def get_output(self):
        return self.centroidal_mpc.get_output()

    def set_state_with_base(self, s, s_dot, H_b, w_b, t):
        self.s = s
        self.s_dot = s_dot
        self.t = t
        self.w_b = w_b
        self.H_b = H_b
        self.kindyn.setRobotState(self.H_b, self.s, self.w_b, self.s_dot, self.gravity)

    def intialize_mpc(self, mpc_parameters: MPCParameterTuning):
        time_horizon = timedelta(seconds=1.2)

        ## MPC Param Hanlder
        self.mpc_param_handler = blf.parameters_handler.StdParametersHandler()
        self.mpc_param_handler.set_parameter_datetime("sampling_time", self.dT)
        self.mpc_param_handler.set_parameter_datetime("time_horizon", time_horizon)
        self.mpc_param_handler.set_parameter_float(
            "contact_force_symmetry_weight",
            mpc_parameters.contact_force_symmetry_weight,
        )
        self.mpc_param_handler.set_parameter_int("verbosity", 0)
        self.mpc_param_handler.set_parameter_int("number_of_maximum_contacts", 2)
        self.mpc_param_handler.set_parameter_int("number_of_slices", 1)
        self.mpc_param_handler.set_parameter_float("static_friction_coefficient", 0.33)
        self.mpc_param_handler.set_parameter_string("linear_solver", "mumps")

        ## MPC Contact Hanlder

        self.contact_0_handler = blf.parameters_handler.StdParametersHandler()
        self.contact_0_handler.set_parameter_int("number_of_corners", 4)
        self.contact_0_handler.set_parameter_string("contact_name", "left_foot")
        self.contact_0_handler.set_parameter_vector_float("corner_0", [0.1, 0.05, 0.0])
        self.contact_0_handler.set_parameter_vector_float("corner_1", [0.1, -0.05, 0.0])
        self.contact_0_handler.set_parameter_vector_float(
            "corner_2", [-0.1, -0.05, 0.0]
        )
        self.contact_0_handler.set_parameter_vector_float("corner_3", [-0.1, 0.05, 0.0])
        self.contact_0_handler.set_parameter_vector_float(
            "bounding_box_lower_limit", [0.0, 0.0, 0.0]
        )
        self.contact_0_handler.set_parameter_vector_float(
            "bounding_box_upper_limit", [0.0, 0.0, 0.0]
        )

        self.contact_1_handler = blf.parameters_handler.StdParametersHandler()
        self.contact_1_handler.set_parameter_int("number_of_corners", 4)
        self.contact_1_handler.set_parameter_string("contact_name", "right_foot")
        self.contact_1_handler.set_parameter_vector_float("corner_0", [0.1, 0.05, 0.0])
        self.contact_1_handler.set_parameter_vector_float("corner_1", [0.1, -0.05, 0.0])
        self.contact_1_handler.set_parameter_vector_float(
            "corner_2", [-0.1, -0.05, 0.0]
        )
        self.contact_1_handler.set_parameter_vector_float("corner_3", [-0.1, 0.05, 0.0])
        self.contact_1_handler.set_parameter_vector_float(
            "bounding_box_lower_limit", [0.0, 0.0, 0.0]
        )
        self.contact_1_handler.set_parameter_vector_float(
            "bounding_box_upper_limit", [0.0, 0.0, 0.0]
        )

        self.mpc_param_handler.set_group("CONTACT_0", self.contact_0_handler)
        self.mpc_param_handler.set_group("CONTACT_1", self.contact_1_handler)

        self.mpc_param_handler.set_parameter_vector_float(
            "com_weight", mpc_parameters.com_weight
        )
        self.mpc_param_handler.set_parameter_float(
            "contact_position_weight", mpc_parameters.contact_position_weight
        )
        self.mpc_param_handler.set_parameter_vector_float(
            "force_rate_of_change_weight", mpc_parameters.force_rate_change_weight
        )
        self.mpc_param_handler.set_parameter_float(
            "angular_momentum_weight", mpc_parameters.angular_momentum_weight
        )
        self.mpc_param_handler.set_parameter_string("solver_name", "ipopt")

        if not self.centroidal_mpc.initialize(self.mpc_param_handler):
            raise RuntimeError("Error while initializing the MPC")
        else:
            print("MPC Initialized")

    def configure(self, s_init, H_b_init):
        self.contact_planner.update_initial_position(Hb=H_b_init, s=s_init)
        self.contact_planner.compute_feet_contact_position()
        self.contact_phase_list = self.contact_planner.get_contact_phase_list()
        self.centroidal_mpc.set_contact_phase_list(self.contact_phase_list)
        self.contact_planner.initialize_foot_swing_planner()

    def define_test_com_traj(self, com0):
        com_knots = []
        time_knots = []
        # com0 = [-0.053640, 0.0, 0.51767]
        com_knots.append(com0)
        vector_phase_list = self.contact_phase_list
        time_knots.append(vector_phase_list.first_phase().begin_time)

        for item in vector_phase_list:
            if (
                len(item.active_contacts) == 2
                and vector_phase_list.first_phase() is not item
                and vector_phase_list.last_phase() is not item
            ):
                time_knots.append((item.end_time + item.begin_time) / 2)
                p1 = item.active_contacts["left_foot"].pose.translation()
                p2 = item.active_contacts["right_foot"].pose.translation()
                des_com = (p1 + p2) / 2
                des_com[2] = com0[2]
                com_knots.append(des_com)
            elif len(item.active_contacts) == 2 and (
                vector_phase_list.last_phase() is item
            ):
                time_knots.append(item.end_time)
                p1 = item.active_contacts["left_foot"].pose.translation()
                p2 = item.active_contacts["right_foot"].pose.translation()
                des_com = (p1 + p2) / 2
                des_com[2] = com0[2]
                com_knots.append(des_com)

        com_spline = blf.planners.QuinticSpline()
        com_spline.set_initial_conditions(np.zeros(3), np.zeros(3))
        com_spline.set_final_conditions(np.zeros(3), np.zeros(3))
        com_spline.set_knots(com_knots, time_knots)
        tempInt = 1000

        com_traj = []
        angular_mom_traj = []
        velocity = np.zeros(3)
        acceleration = np.zeros(3)
        for i in range(tempInt):
            angular_mom_traj_i = np.zeros(3)
            com_temp = np.zeros(3)
            com_spline.evaluate_point(
                i * self.dT_in_seconds, com_temp, velocity, acceleration
            )
            com_traj.append(com_temp)
            angular_mom_traj.append(angular_mom_traj_i)

        self.centroidal_mpc.set_reference_trajectory(com_traj, angular_mom_traj)
        self.centroidal_mpc.set_contact_phase_list(vector_phase_list)
        self.com_traj = com_traj
        self.angular_mom_trak = angular_mom_traj

    def update_contact_phase_list(self, next_planned_contacts):
        new_contact_list = self.contact_phase_list.lists()

        for key, contact in next_planned_contacts:
            it = new_contact_list[key].get_present_contact(contact.activation_time)
            new_contact_list[key].edit_contact(it, contact)

        self.contact_phase_list.set_lists(new_contact_list)

    def update_references(self):
        self.com_traj = self.com_traj[1:]

    def plot_3d_foot(self):
        left_foot_tag = "left_foot"
        right_foot_tag = "right_foot"
        fig = plt.figure()
        self.left_foot_fig = fig.add_subplot(111, projection="3d")
        fig = plt.figure()
        self.right_foot_fig = fig.add_subplot(111, projection="3d")
        output_mpc = self.get_output()
        contact_left = output_mpc.contacts[left_foot_tag]
        contact_right = output_mpc.contacts[right_foot_tag]
        for item in contact_left.corners:
            self.left_foot_fig.quiver(
                item.position[0],
                item.position[1],
                item.position[2],
                item.force[0],
                item.force[1],
                item.force[2],
            )
        for item in contact_right.corners:
            self.right_foot_fig.quiver(
                item.position[0],
                item.position[1],
                item.position[2],
                item.force[0],
                item.force[1],
                item.force[2],
            )
        self.right_foot_fig.set_xlim([-30, 30])
        self.right_foot_fig.set_ylim([-30, 30])
        self.right_foot_fig.set_zlim([-50, 50])
        self.left_foot_fig.set_xlim([-30, 30])
        self.left_foot_fig.set_ylim([-30, 30])
        self.left_foot_fig.set_zlim([-50, 50])
        plt.show()

    def initialize_centroidal_integrator(self, s, s_dot, H_b, w_b, t):
        self.centroidal_integrator = (
            blf.continuous_dynamical_system.CentroidalDynamicsForwardEulerIntegrator()
        )
        self.centroidal_dynamics = blf.continuous_dynamical_system.CentroidalDynamics()
        self.centroidal_integrator.set_dynamical_system(self.centroidal_dynamics)
        self.set_state_with_base(s, s_dot, H_b, w_b, t)

        com = self.kindyn.getCenterOfMassPosition()
        dcom = self.kindyn.getCenterOfMassVelocity()
        Jcm = iDynTree.MatrixDynSize(6, 6 + self.robot_model.NDoF)
        self.kindyn.getCentroidalTotalMomentumJacobian(Jcm)
        nu = np.concatenate((self.w_b, self.s_dot))
        H = Jcm.toNumPy() @ nu
        self.centroidal_dynamics.set_state((com.toNumPy(), dcom.toNumPy(), H[3:]))
        self.centroidal_integrator.set_integration_step(self.dT)

    def get_references(self):
        output_mpc = self.get_output()
        left_foot_tag = "left_foot"
        right_foot_tag = "right_foot"
        forces_left = np.zeros(3)
        for item in output_mpc.contacts[left_foot_tag].corners:
            forces_left = forces_left + item.force
        forces_right = np.zeros(3)
        for item in output_mpc.contacts[right_foot_tag].corners:
            forces_right = forces_right + item.force
        com = self.centroidal_integrator.get_solution()[0]
        dcom = self.centroidal_integrator.get_solution()[1]
        ang_mom = self.centroidal_integrator.get_solution()[2]
        return com, dcom, forces_left, forces_right, ang_mom
