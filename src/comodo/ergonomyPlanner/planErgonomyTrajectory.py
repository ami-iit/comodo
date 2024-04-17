import numpy as np
from comodo.ergonomyPlanner.optimizer import MultiModelOptimizer
from comodo.robotModel.robotModel import RobotModel
from comodo.ergonomyPlanner.utils import matrix_to_xyzrpy
from comodo.abstractClasses.planner import Planner


class PlanErgonomyTrajectory(Planner):
    def __init__(self, robot_model: RobotModel) -> None:
        self.set_planing_requirements()
        super().__init__(robot_model)

    def set_planing_requirements(
        self,
        z_positions: list = [0.55, 0.85, 0.95],
        feet_area: np.ndarray = np.asarray([-0.07, 0.12, -0.05, 0.05]),
        x_position: float = -0.25,
        x_position_gain: float = 1000,
        f_z_init: float = 9.81 * 16.5,
    ) -> None:

        self.z_positions = z_positions
        self.x_position = x_position
        self.x_position_gain = x_position_gain
        self.feet_area = feet_area
        self.f_z_init = f_z_init

    def instantiate_optimizer(self):
        self.optimizer = MultiModelOptimizer(time_steps=3)
        self.optimizer.add_model(
            self.robot_model.robot_name,
            self.robot_model,
            self.robot_model.joint_name_list,
        )

    def set_initial_configuration_and_joint_limits(self):

        [
            s_init,
            base_orientationQuat_position_init,
        ] = self.robot_model.get_initial_configuration()
        self.optimizer.get_model(self.robot_model.robot_name).set_initial_configuration(
            s_init, np.array([0, 0, 0, 1, 0, 0, 0.65]), "degrees"
        )
        self.optimizer.get_model(self.robot_model.robot_name).set_joints_limits(
            self.robot_model.get_joint_limits(), "radians"
        )

    def set_constraints_feet(self):
        feet_transform_constraint = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        self.optimizer.get_model(self.robot_model.robot_name).add_linear_constraint(
            "l_sole_pos_target", self.robot_model.left_foot_frame
        )
        self.optimizer.get_model(self.robot_model.robot_name).update_linear_constraint(
            "l_sole_pos_target", np.array([0, -0.1, 0]), np.array([1, 1, 1])
        )
        self.optimizer.get_model(self.robot_model.robot_name).add_linear_constraint(
            "r_sole_pos_target", self.robot_model.right_foot_frame
        )
        self.optimizer.get_model(self.robot_model.robot_name).update_linear_constraint(
            "r_sole_pos_target", np.array([0, 0.1, 0]), np.array([1, 1, 1])
        )
        self.optimizer.get_model(self.robot_model.robot_name).add_SO3_constraint(
            "l_sole_rot_target", self.robot_model.left_foot_frame
        )
        self.optimizer.get_model(self.robot_model.robot_name).update_SO3_constraint(
            "l_sole_rot_target", feet_transform_constraint, np.array([1, 1, 1])
        )
        self.optimizer.get_model(self.robot_model.robot_name).add_SO3_constraint(
            "r_sole_rot_target", self.robot_model.right_foot_frame
        )
        self.optimizer.get_model(self.robot_model.robot_name).update_SO3_constraint(
            "r_sole_rot_target", feet_transform_constraint, np.array([1, 1, 1])
        )

    def set_constraints_hands(self):

        for t, z_position in enumerate(self.z_positions):
            self.optimizer.get_model(self.robot_model.robot_name).add_linear_constraint(
                "l_hand_pos_constraint_" + str(t), self.robot_model.left_hand, time=[t]
            )
            self.optimizer.get_model(
                self.robot_model.robot_name
            ).update_linear_constraint(
                "l_hand_pos_constraint_" + str(t),
                np.array([self.x_position, -0.2, z_position]),
                np.array([1, 1, 1]),
            )
            self.optimizer.get_model(self.robot_model.robot_name).add_linear_constraint(
                "r_hand_pos_constraint_" + str(t), self.robot_model.rigth_hand, time=[t]
            )
            self.optimizer.get_model(
                self.robot_model.robot_name
            ).update_linear_constraint(
                "r_hand_pos_constraint_" + str(t),
                np.array([self.x_position, 0.2, z_position]),
                np.array([1, 1, 1]),
            )

    def set_tasks_hands(self):
        self.optimizer.get_model(self.robot_model.robot_name).add_linear_target(
            "l_hand_pos_target", self.robot_model.left_hand
        )
        self.optimizer.get_model(self.robot_model.robot_name).update_linear_target(
            "l_hand_pos_target",
            np.array([self.x_position, 0, 0]),
            np.array([self.x_position_gain, 0, 0]),
        )
        self.optimizer.get_model(self.robot_model.robot_name).add_linear_target(
            "r_hand_pos_target", self.robot_model.rigth_hand
        )
        self.optimizer.get_model(self.robot_model.robot_name).update_linear_target(
            "r_hand_pos_target",
            np.array([self.x_position, 0, 0]),
            np.array([self.x_position_gain, 0, 0]),
        )

    def add_contacts(self):
        ## Contacts
        self.optimizer.get_model(self.robot_model.robot_name).add_contact(
            "l_sole_contact",
            self.robot_model.left_foot_frame,
            np.array([1, 1, 1, 1, 1, 1]),
        )
        self.optimizer.get_model(self.robot_model.robot_name).add_contact(
            "r_sole_contact",
            self.robot_model.right_foot_frame,
            np.array([1, 1, 1, 1, 1, 1]),
        )

        self.optimizer.get_model(
            self.robot_model.robot_name
        ).set_contact_CoP_constraint("l_sole_contact", self.feet_area)
        self.optimizer.get_model(
            self.robot_model.robot_name
        ).set_contact_CoP_constraint("r_sole_contact", self.feet_area)
        self.optimizer.get_model(
            self.robot_model.robot_name
        ).set_torsional_friction_constraint("l_sole_contact", 1 / 75)
        self.optimizer.get_model(
            self.robot_model.robot_name
        ).set_torsional_friction_constraint("r_sole_contact", 1 / 75)
        self.optimizer.get_model(
            self.robot_model.robot_name
        ).set_static_friction_cone_constraint("l_sole_contact", 1 / 3)
        self.optimizer.get_model(
            self.robot_model.robot_name
        ).set_static_friction_cone_constraint("r_sole_contact", 1 / 3)
        ## Set initial wrench vector value (this should be done after adding the constraints)
        self.optimizer.get_model(
            self.robot_model.robot_name
        ).set_initial_contact_wrenches_vector(
            np.array(
                [0, 0, self.f_z_init / 2, 0, 0, 0, 0, 0, self.f_z_init / 2, 0, 0, 0]
            )
        )

    def set_discontinuity_min(self):
        # Set higher gain for joints continuity (it helps symmetry)
        self.optimizer.get_model(
            self.robot_model.robot_name
        ).optimization_costs.discontinuity_minimization = 10 * np.identity(
            self.robot_model.NDoF
        )

    def plan_trajectory(self):

        ## instantiate the optimizer
        ## this is done in a method to ensure that each time a new optimizer is used
        ## since at each optimizer called a new robot model will be loaded

        self.instantiate_optimizer()
        ## populating optimization problem
        self.set_initial_configuration_and_joint_limits()
        self.set_constraints_feet()
        self.set_constraints_hands()
        self.add_contacts()
        self.set_discontinuity_min()

        solver_succeded = self.optimizer.solve()
        H_B_opti = []
        xyz_rpy = []
        s_opti = []
        if solver_succeded:
            H_B_opti = []
            xyz_rpy = []
            s_opti = self.optimizer.get_model(self.robot_model.robot_name).s_opti

            for H_b_i in self.optimizer.get_model(self.robot_model.robot_name).H_B_opti:

                H_B_opti.append(H_b_i)
                xyz_rpy.append(matrix_to_xyzrpy(H_b_i))
            # for s in self.optimizer.get_model(self.robot_model.robot_name).s_opti:
            #     H_b_i = self.robot_model.compute_base_pose_left_foot_in_contact(s)
            #     H_B_opti.append(H_b_i)
            #     xyz_rpy.append(utils.matrix_to_xyzrpy(H_b_i))
        self.s_opti = s_opti
        self.H_b_opti = H_B_opti
        self.xyz_rpy = xyz_rpy

        return solver_succeded
