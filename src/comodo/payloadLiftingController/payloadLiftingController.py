from comodo.abstractClasses.controller import Controller
from comodo.payloadLiftingController.payloadLiftingParameterTuning import PayloadLiftingControllerParameters
import time
import numpy as np
from wholebodycontrollib import statemachine
from wholebodycontrollib import wholebodylib
from wholebodycontrollib import wholebodycontrol

# from wholebodycontrollib import robotInterface


class PayloadLiftingController(Controller):
    def __init__(self, frequency, robot_model) -> None:

        # initialize QP
        self.wrench_qp = wholebodycontrol.WrenchQP()
        self.robot_model = robot_model
        [
            self.Adeq_local,
            self.bdeq_local,
        ] = wholebodycontrol.MomentumController.get_rigid_contact_contraint(
            1 / 3, 1 / 75, np.array([[-0.12, 0.12], [-0.05, 0.05]]), 10
        )
        ## TODO use a unique robot model
        self.model = wholebodylib.robot(self.robot_model.get_idyntree_kyndyn())
        self.B = np.block([[np.zeros([6, self.model.ndof])], [np.eye(self.model.ndof)]])
        self.robot_fall = 1.0
        super().__init__(frequency, robot_model)

    def set_control_gains(
        self, parameters: PayloadLiftingControllerParameters
    ):

        self.postural_task_controller = wholebodycontrol.PosturalTaskController(
            self.model.ndof
        )
        postural_task_gain = wholebodycontrol.PosturalTaskGain(self.model.ndof)
        postural_task_gain.Kp = np.diag(parameters.joints_Kp_parameters)
        postural_task_gain.Kd = 2 * np.power(postural_task_gain.Kp, 1 / 2) / 10
        self.postural_task_controller.set_gain(postural_task_gain)
        self.postural_task_controller.set_desired_posture(self.s, self.s_dot)

        self.momentum_controller = wholebodycontrol.MomentumController(
            self.model.kindyn.model().getTotalMass()
        )
        momentum_controller_gain = wholebodycontrol.MomentumControllerGain()
        momentum_controller_gain.Ki = parameters.momentum_ki  # np.array([50, 50, 100, 0, 0, 0])
        momentum_controller_gain.Kp = parameters.momentum_kp 
        self.momentum_controller.set_gain(momentum_controller_gain)
        self.momentum_controller.set_desired_center_of_mass_trajectory(
            self.p_com, np.zeros(3), np.zeros(3)
        )

    def set_time_interval_state_machine(self, state_0_th, state_1_th, state_2_th):
        self.state_0_th = state_0_th
        self.state_1_th = state_1_th
        self.state_2_th = state_2_th

    def initialize_state_machine(self, joint_pos_1, joint_pos_2):
        # TODO we are using the same robot model and we are setting the state
        # This measn that after this the correct robot state should be set
        # This is a dummy base velocity used only to set the robot state and retrieve the needed info
        w_b = self.model.get_base_velocity_from_contacts(
            self.base_pose,
            self.s,
            self.s_dot,
            [self.robot_model.left_foot_frame, self.robot_model.right_foot_frame],
        )
        self.state_machine = statemachine.StateMachine(False)
        configuration_0 = statemachine.Configuration(
            self.s, self.p_com, self.state_0_th
        )

        base_pose_1 = self.model.get_base_pose_from_contacts(
            joint_pos_1,
            {
                self.robot_model.left_foot_frame: np.eye(4),
                self.robot_model.right_foot_frame: np.eye(4),
            },
        )
        self.model.set_state(base_pose_1, joint_pos_1, w_b, self.s_dot)
        p_com_1 = self.model.get_center_of_mass_position()
        configuration_1 = statemachine.Configuration(
            joint_pos_1, p_com_1, self.state_1_th
        )

        base_pose_2 = self.model.get_base_pose_from_contacts(
            joint_pos_2,
            {
                self.robot_model.left_foot_frame: np.eye(4),
                self.robot_model.right_foot_frame: np.eye(4),
            },
        )
        self.model.set_state(base_pose_2, joint_pos_2, w_b, self.s_dot)
        p_com_2 = self.model.get_center_of_mass_position()
        configuration_2 = statemachine.Configuration(
            joint_pos_2, p_com_2, self.state_2_th
        )

        self.state_machine.add_configuration(configuration_0)
        self.state_machine.add_configuration(configuration_1)
        self.state_machine.add_configuration(configuration_2)
        self.state_machine.add_configuration(configuration_1)
        self.state_machine.add_configuration(configuration_0)
        self.joint_pos_des = np.copy(self.s)
        self.p_com_des = np.copy(self.p_com)
        self.f = np.zeros(12)

    def set_state(self, s, s_dot, t):
        consider_hands_wrenches = False
        self.base_pose = self.model.get_base_pose_from_contacts(
            s,
            {
                self.robot_model.left_foot_frame: np.eye(4),
                self.robot_model.right_foot_frame: np.eye(4),
            },
        )
        self.w_b = self.model.get_base_velocity_from_contacts(
            self.base_pose,
            s,
            s_dot,
            [self.robot_model.left_foot_frame, self.robot_model.right_foot_frame],
        )

        self.base_pose = self.model.get_base_pose_from_contacts(
            s,
            {
                self.robot_model.left_foot_frame: np.eye(4),
                self.robot_model.right_foot_frame: np.eye(4),
            },
        )
        self.w_b = self.model.get_base_velocity_from_contacts(
            self.base_pose,
            s,
            s_dot,
            [self.robot_model.left_foot_frame, self.robot_model.right_foot_frame],
        )

        # get kinematic and dynamic quantities
        self.model.set_state(self.base_pose, s, self.w_b, s_dot)
        self.M = self.model.get_mass_matrix()
        self.h = self.model.get_generalized_bias_force()
        self.Jcm = self.model.get_centroidal_momentum_jacobian()
        self.H = self.model.get_centroidal_momentum()
        self.p_com = self.model.get_center_of_mass_position()
        J_feet = self.model.get_frames_jacobian([self.robot_model.left_foot_frame, self.robot_model.right_foot_frame])
        if consider_hands_wrenches:
            J_l_hand = self.model.get_frames_jacobian([self.robot_model.left_hand_frame])
            J_r_hand = self.model.get_frames_jacobian([self.robot_model.rigth_hand_frame])

        Jdot_nu_feet = self.model.get_frames_bias_acceleration([self.robot_model.left_foot_frame, self.robot_model.right_foot_frame])
        
        if consider_hands_wrenches:
            Jdot_nu_l_hand = self.model.get_frames_bias_acceleration([self.robot_model.left_hand_frame])
            Jdot_nu_r_hand = self.model.get_frames_bias_acceleration([self.robot_model.rigth_hand_frame])
        
        if consider_hands_wrenches:
            self.w_H_frames = self.model.get_frames_transform([self.robot_model.left_foot_frame, self.robot_model.right_foot_frame, self.robot_model.left_hand_frame, self.robot_model.rigth_hand_frame])
        else:
            self.w_H_frames = self.model.get_frames_transform([self.robot_model.left_foot_frame, self.robot_model.right_foot_frame])

        if consider_hands_wrenches:
            # Get adjoint transform from l_hand to r_hand in mixed representation
            w_H_l_hand = self.model.get_frames_transform(["l_hand_palm"])
            w_H_r_hand = self.model.get_frames_transform(["r_hand_palm"])
            r_i   = w_H_r_hand[0:3,3] - w_H_l_hand[0:3,3]
            skew_r_i = np.array([[0,      -r_i[2],  r_i[1]], 
                                    [r_i[2],       0, -r_i[0]], 
                                    [-r_i[1], r_i[0],       0]])
            l_hand_X_r_hand = np.block([[np.eye(3), skew_r_i],
                                    [np.zeros([3, 3]),  np.eye(3)]]) 
                

        if consider_hands_wrenches:
            self.Jf = np.vstack((J_feet,J_l_hand,J_r_hand))
            self.Jc = J_feet
            self.Jdot_nu = Jdot_nu_feet
        else:
            self.Jf = J_feet
            self.Jc = J_feet
            self.Jdot_nu = Jdot_nu_feet
        super().set_state(s, s_dot, t)

    def update_desired_configuration(self):
        updated = self.state_machine.update(self.t)
        (
            self.joint_pos_des,
            self.joint_vel_des,
            self.joint_acc_des,
            _,
            _,
            _,
        ) = self.state_machine.get_state()

        base_pose_des = self.model.get_base_pose_from_contacts(
            self.joint_pos_des,
            {
                self.robot_model.left_foot_frame: np.eye(4),
                self.robot_model.right_foot_frame: np.eye(4),
            },
        )
        w_b_des = self.model.get_base_velocity_from_contacts(
            base_pose_des,
            self.joint_pos_des,
            self.joint_vel_des,
            [self.robot_model.left_foot_frame, self.robot_model.right_foot_frame],
        )
        w_dot_b_des = self.model.get_base_acceleration_from_contacts(
            base_pose_des,
            w_b_des,
            self.joint_pos_des,
            self.joint_vel_des,
            self.joint_acc_des,
            [self.robot_model.left_foot_frame, self.robot_model.right_foot_frame],
        )

        self.model.set_state(
            base_pose_des, self.joint_pos_des, w_b_des, self.joint_vel_des
        )

        self.vel_com_des = self.model.get_center_of_mass_velocity()
        self.p_com_des = self.model.get_center_of_mass_position()
        self.acc_com_des = self.model.get_center_of_mass_acceleration(
            w_dot_b_des, self.joint_acc_des
        )
        return updated

    def run(self):

        # updated = self.update_desired_configuration()
        self.postural_task_controller.set_desired_posture(
            self.joint_pos_des, self.joint_vel_des
        )
        self.momentum_controller.set_desired_center_of_mass_trajectory(
            self.p_com_des, self.vel_com_des, self.acc_com_des
        )

        # compute postural and momentum controller
        [
            tau_0_model,
            tau_0_sigma,
        ] = self.postural_task_controller.get_postural_task_torque(
            self.s, self.s_dot, self.M, self.Jc, self.h
        )
        # Feet wrenches inequality constraints
        [
            Adeq_local_left_foot,
            _,
        ] = self.momentum_controller.tranform_local_wrench_task_into_global(
            self.Adeq_local, self.bdeq_local, self.w_H_frames[:4, :]
        )
        [
            Adeq_local_right_foot,
            _,
        ] = self.momentum_controller.tranform_local_wrench_task_into_global(
            self.Adeq_local, self.bdeq_local, self.w_H_frames[4:8, :]
        )
        Adeq = np.block(
            [
                [Adeq_local_left_foot, np.zeros([Adeq_local_left_foot.shape[0], 6])],
                [np.zeros([Adeq_local_right_foot.shape[0], 6]), Adeq_local_right_foot],
            ]
        )
        bdeq = np.block([self.bdeq_local, self.bdeq_local])
        [Aeq, beq] = self.momentum_controller.get_momentum_control_tasks(
            self.H, self.p_com, self.w_H_frames
        )

        [tau_sigma, tau_model] = wholebodycontrol.get_torques_projected_dynamics(
            tau_0_model, tau_0_sigma, self.Jc, self.Jf, self.Jdot_nu, self.M, self.h, self.B
        )
        # solve QP optimization
        self.f = self.wrench_qp.solve(tau_model, tau_sigma, Aeq, beq, Adeq, bdeq, 0.001, "quadprog")
        # self.f = opti_solution["x"]
        # success = opti_solution["success"]
        if self.f is None:
            self.robot_fall = 0.0
            return False
        # set output torque
        self.torque = tau_sigma @ self.f + tau_model
        return True

    def get_fitness_parameters(self):
        postural_task_error = self.postural_task_controller.get_tracking_error(self.s)
        error_com, error_ang = self.momentum_controller.get_tracking_error(
            self.H, self.p_com
        )
        return postural_task_error, error_com, self.robot_fall
