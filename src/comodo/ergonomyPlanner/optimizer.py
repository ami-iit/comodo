import string
from adam.casadi.computations import KinDynComputations
import numpy as np
import casadi as cs
from comodo.ergonomyPlanner.utils import FromQuaternionToMatrix
from dataclasses import dataclass


@dataclass
class OptimizationModelFlags:
    """Class for keeping track of the flag in the optimization model."""

    # constraints
    use_model_dynamics_constraint: bool = True
    use_joint_limits_constraint: bool = True
    use_task_space_constraints: bool = True
    use_contact_constraints: bool = True
    use_joint_simmetry_constraint: bool = False

    # costs
    use_tau_minimization: bool = True
    use_joints_decentralization_minimization: bool = False
    use_target_joints_configuration_target: bool = False
    use_target_base_position_target: bool = False
    use_target_base_orientation_target: bool = True
    use_task_space_targets: bool = True
    use_discontinuity_minimization: bool = True
    use_joint_simmetry_target: bool = True

    # initialization
    use_joints_initial_configuration: bool = True
    use_tau_initial_configuration: bool = True
    use_base_position_initial_configuration: bool = True
    use_base_orientation_initial_configuration: bool = True
    use_wrench_vector_initial_configuration: bool = True

    # status
    is_solved = False


@dataclass
class ContactConstraintFlags:
    use_local_CoP_constraint: bool = False
    use_torsional_friction_constraint: bool = False
    use_static_friction_cone_constraint: bool = False


class OptimizationModelCosts:
    """Class for keeping track of the costs in the optimization model."""

    def __init__(self, NDoF):
        self.tau_mimimization = np.identity(NDoF)
        self.joint_decentralization_minimization = np.identity(NDoF)
        self.desired_joints_configuration_target = np.identity(NDoF)
        self.desired_base_position_target = np.identity(3)
        self.desired_base_orientation_target = np.identity(3)
        self.discontinuity_minimization = np.identity(NDoF)
        self.symmsteric_joints_target = 1


class ModelTaskSpaceTargets:
    """Class defining the targets in the task space"""

    def __init__(self, target_type, frame_name):
        target_types = ["SO3", "SE3", "Linear", "2DPlane"]
        if not target_type in target_types:
            raise ValueError(
                "[ModelTaskSpaceTargets] : Target type ", target_type, "not valid."
            )
        self.target_type = target_type
        self.frame_name = frame_name


class SO3Target(ModelTaskSpaceTargets):
    """Class defining an SO3 target"""

    def __init__(self, frame_name, target: np.array = np.identity(3)):
        if target.shape != (3, 3):
            raise ValueError(
                "[SO3Target] : Target size not valid (", target.shape, ")."
            )
        self.target = target
        super().__init__("SO3", frame_name)

    def get_error_size(self):
        return 3

    def update_target(self, target: np.array):
        if target.shape != (3, 3):
            raise ValueError(
                "[SO3Target::update_target] : Target size not valid (",
                target.shape,
                ").",
            )
        self.target = target

    def compute_error(self, measured):
        if measured.shape != (3, 3):
            raise ValueError(
                "[SO3Target::compute_error] : Measure size not valid (",
                measured.shape,
                ").",
            )
        return cs.inv_skew(measured @ cs.transpose(self.target))

    def compute_error_from_transform(self, measured_transform):
        if measured_transform.shape != (4, 4):
            raise ValueError(
                "[SO3Target::compute_error_from_transform] : Measure size not valid (",
                measured_transform.shape,
                ").",
            )
        return self.compute_error(self.transform_to_measurement(measured_transform))

    def transform_to_measurement(self, transform):
        return transform[:3, :3]

    def to_string(self, measured_transform: np.array = np.empty):
        print("Target Type : ", self.target_type)
        print("Frame       : ", self.frame_name)
        print("Target      :")
        print(self.target)

        if not measured_transform == np.empty:
            print("Measured    :")
            print(self.transform_to_measurement(measured_transform))
            print("Error       :")
            print(self.compute_error_from_transform(measured_transform))


class LinearTarget(ModelTaskSpaceTargets):
    """Class defining a Linear Position Target"""

    def __init__(self, frame_name, target: np.array = np.zeros(3)):
        if target.size != 3:
            raise ValueError(
                "[LinearTarget] : Target size not valid (", target.size, ")."
            )
        self.target = target
        super().__init__("Linear", frame_name)

    def get_error_size(self):
        return 3

    def update_target(self, target: np.array):
        if target.shape != (3, 1) and target.size != 3:
            raise ValueError(
                "[LinearTarget::update_target] : Target size not valid (",
                target.shape,
                ").",
            )
        self.target = target

    def compute_error(self, measured):
        if measured.shape != (3, 1):
            raise ValueError(
                "[LinearTarget::compute_error] : Measure size not valid (",
                measured.shape,
                ").",
            )
        return measured - self.target

    def compute_error_from_transform(self, measured_transform):
        if measured_transform.shape != (4, 4):
            raise ValueError(
                "[LinearTarget::compute_error_from_transform] : Measure size not valid (",
                measured_transform.shape,
                ").",
            )
        return self.compute_error(self.transform_to_measurement(measured_transform))

    def transform_to_measurement(self, transform):
        return transform[:3, 3]

    def to_string(self, measured_transform: np.array = np.empty):
        print("Target Type : ", self.target_type)
        print("Frame       : ", self.frame_name)
        print("Target      :")
        print(self.target)

        if not measured_transform == np.empty:
            print("Measured    :")
            print(self.transform_to_measurement(measured_transform))
            print("Error       :")
            print(self.compute_error_from_transform(measured_transform))


class SE3Target(ModelTaskSpaceTargets):
    """Class defining an SE3 Target"""

    def __init__(self, frame_name, target: np.array = np.array(3)):
        if target.shape != (4, 4):
            raise ValueError(
                "[SE3Target] : Target size not valid (", target.shape, ")."
            )
        self.target = target
        super().__init__("SE3", frame_name)

    def get_error_size(self):
        return 6

    def update_target(self, target: np.array):
        if target.shape != (4, 4):
            raise ValueError(
                "[SE3Target::update_target] : Target size not valid (",
                target.shape,
                ").",
            )
        self.target = target

    def compute_error(self, measured):
        if measured.shape != (4, 4):
            raise ValueError(
                "[SE3Target::compute_error] : Measure size not valid (",
                measured.shape,
                ").",
            )
        return cs.vcat(
            [
                measured[:3, 3] - self.target[:3, 3],
                cs.inv_skew(measured[:3, :3] @ cs.transpose(self.target[:3, :3])),
            ]
        )

    def compute_error_from_transform(self, measured_transform):
        if measured_transform.shape != (4, 4):
            raise ValueError(
                "[SE3Target::compute_error_from_transform] : Measure size not valid (",
                measured_transform.shape,
                ").",
            )
        return self.compute_error(self.transform_to_measurement(measured_transform))

    def transform_to_measurement(self, transform):
        return transform

    def to_string(self, measured_transform: np.array = np.empty):
        print("Target Type : ", self.target_type)
        print("Frame       : ", self.frame_name)
        print("Target      :")
        print(self.target)

        if not measured_transform == np.empty:
            print("Measured    :")
            print(self.transform_to_measurement(measured_transform))
            print("Error       :")
            print(self.compute_error_from_transform(measured_transform))


class PlaneTarget(ModelTaskSpaceTargets):
    """Class defining an 2D-Plane Target"""

    def __init__(
        self,
        frame_name,
        plane_normal_in_frame: np.array = np.array([1, 0, 0]),
        target: np.array = np.array([1, 0, 0]),
    ):
        if target.size != 3:
            raise ValueError(
                "[PlaneTarget] : Target size not valid (", target.size, ")."
            )
        if plane_normal_in_frame.size != 3:
            raise ValueError(
                "[PlaneTarget] : Plane normal description size not valid (",
                plane_normal_in_frame.size,
                ").",
            )
        # normalize directions
        self.plane_normal_in_frame = plane_normal_in_frame
        self.target = target
        super().__init__("2DPlane", frame_name)

    def get_error_size(self):
        return 1

    def update_target(self, target: np.array):
        if target.size != 3:
            raise ValueError(
                "[PlaneTarget::update_target] : Target size not valid (",
                target.size,
                ").",
            )
        self.target = target

    def compute_error(self, measured):
        if measured.size != 3 and measured.shape != (3, 1):
            raise ValueError(
                "[PlaneTarget::compute_error] : Measure size not valid (",
                measured.shape,
                ").",
            )
        return (
            cs.dot(self.target, measured)
            / (cs.norm_2(self.target) * cs.norm_2(measured))
            - 1
        )
        # return  cs.acos(cs.dot(self.target, measured) / (cs.norm_2(self.target) * cs.norm_2(measured) ))

    def compute_error_from_transform(self, measured_transform):
        if measured_transform.shape != (4, 4):
            raise ValueError(
                "[PlaneTarget::compute_error_from_transform] : Measure size not valid (",
                measured_transform.shape,
                ").",
            )
        return self.compute_error(self.transform_to_measurement(measured_transform))

    def transform_to_measurement(self, transform):
        return transform[:3, :3] @ self.plane_normal_in_frame

    def to_string(self, measured_transform: np.array = np.empty):
        print("Target Type : ", self.target_type)
        print("Frame       : ", self.frame_name)
        print("Target      :")
        print(self.target)

        if not measured_transform == np.empty:
            print("Transform   :")
            print(measured_transform)
            print("Measured    :")
            print(self.transform_to_measurement(measured_transform))
            print("Error       :")
            print(self.compute_error_from_transform(measured_transform))


class ContactConstraint:
    """Constraint describing a rigid contact"""

    def __init__(
        self, frame_name, constrained_velocity_components: np.array = np.ones(6)
    ):
        # TODO: check constrained_velocity_components is a list of ones and zeros
        self.frame_name = frame_name
        self.constrained_velocity_components = constrained_velocity_components
        self.contact_constraint_flags = ContactConstraintFlags()

    def get_contact_dimensions(self):
        return np.sum(self.constrained_velocity_components)

    def set_local_CoP_constraint(
        self, contact_area_size: np.array, normal_wrench_min=0
    ):
        self.contact_area_size = contact_area_size
        self.normal_wrench_min = normal_wrench_min
        self.contact_constraint_flags.use_local_CoP_constraint = True

    def set_torsional_friction_constraint(self, torsional_friction_coefficient):
        self.torsional_friction_coefficient = torsional_friction_coefficient
        self.contact_constraint_flags.use_torsional_friction_constraint = True

    def set_static_friction_cone_constraint(
        self, static_friction_coefficient, normal_wrench_min=0
    ):
        self.static_friction_coefficient = static_friction_coefficient
        self.normal_wrench_min = normal_wrench_min
        self.contact_constraint_flags.use_static_friction_cone_constraint = True

    def get_wrench_from_reduced_wrench_vector(self, f_reduced):
        f = cs.MX.zeros(6)
        index_reduced_vector = 0
        for i in range(0, self.get_contact_dimensions()):
            if not self.constrained_velocity_components[i] == 0:
                f[i] = f_reduced[index_reduced_vector]
                index_reduced_vector = index_reduced_vector + 1
            else:
                f[i] = 0
        return f

    def get_local_wrench(self, f_reduced, frame_orientation):

        # TODO: better way to do this computation and verify if we have to transform the linear part or should be identity
        X_c = cs.MX.zeros(6, 6)
        X_c[:3, :3] = frame_orientation
        X_c[3:, 3:] = frame_orientation

        f = X_c.T @ self.get_wrench_from_reduced_wrench_vector(f_reduced)

        return f

    def get_local_CoP(self, f_reduced, frame_orientation):

        f = self.get_local_wrench(f_reduced, frame_orientation)

        CoP = cs.MX.zeros(3)

        CoP[0] = -f[4] / f[2]
        CoP[1] = f[3] / f[2]

        return CoP


class ModelTarget:
    """Target structure in the optimization model"""

    def __init__(self, target_object, gain: np.array, time_steps):
        # TODO: check gain size
        self.target_object = target_object
        self.gain = gain
        self.time_steps = time_steps


class ModelConstraint:
    """Target structure in the optimization model"""

    def __init__(self, constraint_object, activation_flags, time_steps):
        # TODO: check activation_flags is a list of ones and zeros
        self.constraint_object = constraint_object
        self.activation_flags = activation_flags
        self.time_steps = time_steps


class ModelContactConstraint:
    """Constraint structure in the optimization model"""

    def __init__(self, constraint_object, wrench_vector_indices):
        # TODO: check activation_flags is a list of ones and zeros
        self.constraint_object = constraint_object
        self.wrench_vector_indices = wrench_vector_indices


class OptimizationModel:
    """model object for optimization"""

    def __init__(
        self,
        model: KinDynComputations,
        joints_name_list: list,
        solver: cs.Opti,
        time_steps=1,
    ):
        self.model = model

        self.time_steps = time_steps

        if len(joints_name_list) != self.model.NDoF:
            raise ValueError(
                "[OptimizationModel] : joint name list lenght is different from model DoF."
            )
        self.joints_name_list = joints_name_list

        self.s_min = np.full(self.model.NDoF, -np.pi)
        self.s_max = np.full(self.model.NDoF, np.pi)

        self.s_initial = np.zeros(self.model.NDoF)
        self.tau_initial = np.zeros(self.model.NDoF)
        self.quatPos_B_initial = np.array([1, 0, 0, 0, 0, 0, 0])
        self.fc_initial = np.zeros(0)

        self.s_target = np.zeros(self.model.NDoF)
        self.H_B_target = np.identity(4)

        self.s_opti = None
        self.tau_opti = None
        self.H_B_opti = None
        self.fc_opti = None

        self.symmetric_joints_target_list = []
        self.symmetric_joints_constraint_list = []

        self.task_space_targets = {}
        self.task_space_constraints = {}
        self.contact_constraints = {}

        self.solver = solver
        self.s = []
        self.tau = []
        self.base_orientationQuat_position = []
        self.H_B = []
        self.fc = []

        for t in range(self.time_steps):
            self.s.append(solver.variable(self.model.NDoF))
            self.tau.append(solver.variable(self.model.NDoF))
            self.base_orientationQuat_position.append(solver.variable(7))
            self.H_B.append(
                FromQuaternionToMatrix()(self.base_orientationQuat_position[t])
            )
            self.fc.append(solver.variable(6 * 0))

        # Optimization Flags
        self.optimization_flags = OptimizationModelFlags()

        # Optimization Gains
        self.optimization_costs = OptimizationModelCosts(self.model.NDoF)

    def set_initial_configuration(
        self,
        s_initial: dict,
        quatPos_B_initial: np.array,
        unit_of_measurement: str = "radians",
    ):
        if quatPos_B_initial.size != 7:
            raise ValueError(
                "[OptimizationModel::set_initial_configuration] : initial base is not valid."
            )
        self.quatPos_B_initial = quatPos_B_initial

        if self.model.NDoF != 0:
            for joint_name, joint_initial_configuration in s_initial.items():
                if joint_name in self.joints_name_list:
                    joint_index = self.joints_name_list.index(joint_name)
                    self.s_initial[joint_index] = toRadians(
                        joint_initial_configuration, unit_of_measurement
                    )

    def set_initial_contact_wrenches_vector(self, fc_initial: np.array):
        if fc_initial.size != self.get_contacts_wrench_vector_size():
            raise ValueError(
                "[OptimizationModel::set_initial_contact_wrenches_vector] : initial contact wrenches are not valid."
            )
        self.fc_initial = fc_initial

    def set_initial_tau(self, tau_initial):
        self.tau_initial = tau_initial

    def set_target_configuration(
        self, s_target: dict, H_B_target: np.array, unit_of_measurement: str = "radians"
    ):
        self.H_B_target = H_B_target
        self.optimization_flags.use_target_base_orientation_target = True
        self.optimization_flags.use_target_base_position_target = True

        if self.model.NDoF != 0:
            for joint_name, joint_target_configuration in s_target.items():
                if joint_name in self.joints_name_list:
                    joint_index = self.joints_name_list.index(joint_name)
                    self.s_target[joint_index] = toRadians(
                        joint_target_configuration, unit_of_measurement
                    )

            self.optimization_flags.use_target_joints_configuration_target = True

    def set_symmetric_joints_target_pair(self, joint_name_1: str, joint_name_2: str):
        if all(name in self.joints_name_list for name in [joint_name_1, joint_name_2]):
            self.symmetric_joints_target_list.append([joint_name_1, joint_name_2])
        self.optimization_flags.use_joint_simmetry_target = True
        # TODO: raise warning otherwise

    def set_symmetric_joints_constraint_pair(
        self, joint_name_1: str, joint_name_2: str
    ):
        if all(name in self.joints_name_list for name in [joint_name_1, joint_name_2]):
            self.symmetric_joints_constraint_list.append([joint_name_1, joint_name_2])
        self.optimization_flags.use_joint_simmetry_constraint = True
        # TODO: raise warning otherwise

    def set_symmetric_joints_targets(self, groups_list: list):
        for group in groups_list:
            for index in range(0, len(group) - 1):
                self.set_symmetric_joints_target_pair(group[index], group[index + 1])

    def set_symmetric_joints_constraints(self, groups_list: list):
        for group in groups_list:
            for index in range(0, len(group) - 1):
                self.set_symmetric_joints_constraint_pair(
                    group[index], group[index + 1]
                )

    def set_joint_limits(
        self, joint_name, lower_bound, upper_bound, unit_of_measurement: str = "radians"
    ):
        if lower_bound > upper_bound:
            raise ValueError(
                "[OptimizationModel::set_joint_limits] : joint lower bound should be smaller then upper bound."
            )
        if joint_name in self.joints_name_list:
            joint_index = self.joints_name_list.index(joint_name)
            self.s_min[joint_index] = toRadians(lower_bound, unit_of_measurement)
            self.s_max[joint_index] = toRadians(upper_bound, unit_of_measurement)
            self.optimization_flags.use_joint_limits_constraint = True
        # TODO: add warning if the joint name is not found

    def set_joints_limits(
        self, joints_limits: dict, unit_of_measurement: str = "radians"
    ):
        for joint_name, joint_limits in joints_limits.items():
            self.set_joint_limits(
                joint_name, joint_limits[0], joint_limits[1], unit_of_measurement
            )

    def add_contact(self, contact_name, frame_name, constrained_velocity_components):
        # TODO: check if frame exists in the model
        # TODO: check if contact name already exists
        wrench_vector_indices = np.arange(
            self.get_contacts_wrench_vector_size(),
            self.get_contacts_wrench_vector_size()
            + np.sum(constrained_velocity_components),
        )
        self.contact_constraints[contact_name] = ModelContactConstraint(
            ContactConstraint(frame_name, constrained_velocity_components),
            wrench_vector_indices,
        )
        for t in range(self.time_steps):
            self.fc[t] = self.solver.variable(self.get_contacts_wrench_vector_size())
        if self.fc_initial.size != self.get_contacts_wrench_vector_size():
            self.fc_initial = np.zeros(self.get_contacts_wrench_vector_size())
            # Warning : fc initial has been resized, previous initialization lost

    def set_contact_CoP_constraint(
        self, contact_name, contact_area_size: np.array, normal_wrench_min=0
    ):
        # TODO: check if contact name exists
        self.contact_constraints[
            contact_name
        ].constraint_object.set_local_CoP_constraint(
            contact_area_size, normal_wrench_min
        )

    def set_torsional_friction_constraint(
        self, contact_name, torsional_friction_coefficient
    ):
        self.contact_constraints[
            contact_name
        ].constraint_object.set_torsional_friction_constraint(
            torsional_friction_coefficient
        )

    def set_static_friction_cone_constraint(
        self, contact_name, static_friction_coefficient, normal_wrench_min=0
    ):
        self.contact_constraints[
            contact_name
        ].constraint_object.set_static_friction_cone_constraint(
            static_friction_coefficient, normal_wrench_min
        )

    def add_SO3_target(
        self,
        target_name,
        frame_name,
        target: np.array = np.identity(3),
        gain: np.array = np.ones(3),
        time=[-1],
    ):
        # TODO: check if frame exists in the model
        # TODO: check if target name already exists
        self.task_space_targets[target_name] = ModelTarget(
            SO3Target(frame_name, target), gain, time
        )

    def update_SO3_target(
        self, target_name, target: np.array, gain: np.array = np.ones(3)
    ):
        # TODO: check if target name exists
        self.task_space_targets[target_name].target_object.update_target(target)
        self.task_space_targets[target_name].gain = gain

    def add_linear_target(
        self,
        target_name,
        frame_name,
        target: np.array = np.zeros(3),
        gain: np.array = np.ones(3),
        time=[-1],
    ):
        # TODO: check if frame exists in the model
        # TODO: check if target name already exists
        self.task_space_targets[target_name] = ModelTarget(
            LinearTarget(frame_name, target), gain, time
        )

    def update_linear_target(
        self, target_name, target: np.array, gain: np.array = np.ones(3)
    ):
        # TODO: check if target name exists
        self.task_space_targets[target_name].target_object.update_target(target)
        self.task_space_targets[target_name].gain = gain

    def add_SE3_target(
        self,
        target_name,
        frame_name,
        target: np.array = np.identity(4),
        gain: np.array = np.ones(6),
        time=[-1],
    ):
        # TODO: check if frame exists in the model
        # TODO: check if target name already exists
        self.task_space_targets[target_name] = ModelTarget(
            SE3Target(frame_name, target), gain, time
        )

    def update_SE3_target(
        self, target_name, target: np.array, gain: np.array = np.ones(6)
    ):
        # TODO: check if target name exists
        self.task_space_targets[target_name].target_object.update_target(target)
        self.task_space_targets[target_name].gain = gain

    def add_plane_target(
        self,
        target_name,
        frame_name,
        plane_normal_in_frame: np.array = np.array([1, 0, 0]),
        target: np.array = np.array([1, 0, 0]),
        gain: np.array = np.ones(1),
        time=[-1],
    ):
        # TODO: check if frame exists in the model
        # TODO: check if target name already exists
        self.task_space_targets[target_name] = ModelTarget(
            PlaneTarget(frame_name, plane_normal_in_frame, target), gain, time
        )

    def update_plane_target(
        self, target_name, target: np.array, gain: np.array = np.ones(1)
    ):
        # TODO: check if target name exists
        self.task_space_targets[target_name].target_object.update_target(target)
        self.task_space_targets[target_name].gain = gain

    def remove_target(self, target_name):
        # TODO: check if target name exists
        del self.task_space_targets[target_name]

    def add_SO3_constraint(
        self,
        target_name,
        frame_name,
        target: np.array = np.identity(3),
        activation_flags: np.array = np.ones(3),
        time=[-1],
    ):
        # TODO: check if frame exists in the model
        # TODO: check if target name already exists
        # TODO: check if activation flags are bool (ones and zeros)
        self.task_space_constraints[target_name] = ModelConstraint(
            SO3Target(frame_name, target), activation_flags, time
        )

    def update_SO3_constraint(
        self, target_name, target: np.array, activation_flags: np.array = np.ones(3)
    ):
        # TODO: check if target name exists
        self.task_space_constraints[target_name].constraint_object.update_target(target)
        self.task_space_constraints[target_name].activation_flags = activation_flags

    def add_linear_constraint(
        self,
        target_name,
        frame_name,
        target: np.array = np.zeros(3),
        activation_flags: np.array = np.ones(3),
        time=[-1],
    ):
        # TODO: check if frame exists in the model
        # TODO: check if target name already exists
        # TODO: check if activation flags are bool (ones and zeros)
        self.task_space_constraints[target_name] = ModelConstraint(
            LinearTarget(frame_name, target), activation_flags, time
        )

    def update_linear_constraint(
        self, target_name, target: np.array, activation_flags: np.array = np.ones(3)
    ):
        # TODO: check if target name exists
        self.task_space_constraints[target_name].constraint_object.update_target(target)
        self.task_space_constraints[target_name].activation_flags = activation_flags

    def add_SE3_constraint(
        self,
        target_name,
        frame_name,
        target: np.array = np.identity(4),
        activation_flags: np.array = np.ones(6),
        time=[-1],
    ):
        # TODO: check if frame exists in the model
        # TODO: check if target name already exists
        # TODO: check if activation flags are bool (ones and zeros)
        self.task_space_constraints[target_name] = ModelConstraint(
            SE3Target(frame_name, target), activation_flags, time
        )

    def update_SE3_constraint(
        self, target_name, target: np.array, activation_flags: np.array = np.ones(6)
    ):
        # TODO: check if target name exists
        self.task_space_constraints[target_name].constraint_object.update_target(target)
        self.task_space_constraints[target_name].activation_flags = activation_flags

    def add_plane_constraint(
        self,
        target_name,
        frame_name,
        plane_normal_in_frame: np.array = np.array([1, 0, 0]),
        target: np.array = np.array([1, 0, 0]),
        activation_flags: np.array = np.ones(1),
        time=[-1],
    ):
        # TODO: check if frame exists in the model
        # TODO: check if target name already exists
        # TODO: check if activation flags are bool (ones and zeros)
        self.task_space_constraints[target_name] = ModelConstraint(
            PlaneTarget(frame_name, plane_normal_in_frame, target),
            activation_flags,
            time,
        )

    def update_plane_constraint(
        self, target_name, target: np.array, activation_flags: np.array = np.ones(1)
    ):
        # TODO: check if target name exists
        self.task_space_constraints[target_name].constraint_object.update_target(target)
        self.task_space_constraints[target_name].activation_flags = activation_flags

    def remove_constraint(self, target_name):
        # TODO: check if target name exists
        del self.task_space_constraints[target_name]

    def get_contacts_wrench_vector_size(self):
        contacts_wrench_vector_size = 0
        for contact in self.contact_constraints.values():
            contacts_wrench_vector_size = (
                contacts_wrench_vector_size
                + contact.constraint_object.get_contact_dimensions()
            )
        return contacts_wrench_vector_size

    def contacts_Jacobian(self, H_B, s):
        Jc_list = []
        for contact in self.contact_constraints.values():
            full_jacobian = self.model.jacobian_fun(
                contact.constraint_object.frame_name
            )(H_B, s)
            Jc_list.append(
                full_jacobian[
                    np.nonzero(
                        contact.constraint_object.constrained_velocity_components
                    )[0],
                    :,
                ]
            )
        return cs.vcat(Jc_list)

    def tau_(self, H_B, s, fc):
        g_fun = self.model.gravity_term_fun()
        g = g_fun(H_B, s)
        Jc = self.contacts_Jacobian(H_B, s)
        tau = g[6:] - cs.transpose(Jc[:, 6:]) @ fc

        return tau

    def tau_minimization_cost(self, tau, task_cost=1):
        cost = cs.sumsqr(task_cost @ tau)
        return cost

    def joint_symmetry_cost(self, s, task_cost):
        cost = 0
        for pair in self.symmetric_joints_target_list:
            cost = cost + cs.sumsqr(
                task_cost
                @ (
                    s[self.joints_name_list.index(pair[0])]
                    - s[self.joints_name_list.index(pair[1])]
                )
            )
        return cost

    def joint_decentralization_cost(self, s, task_cost=1):
        # TODO handle case in which min==max
        s_delta = self.s_max - self.s_min
        decentralization = (s - (self.s_min + s_delta / 2)) / (s_delta / 2)
        cost = cs.sumsqr(task_cost @ decentralization)
        return cost

    def joint_error_cost(self, s1, s2, task_cost=1):
        s_error = s1 - s2
        cost = cs.sumsqr(task_cost @ s_error)
        return cost

    def desired_joints_configuration_cost(self, s, task_cost=1):
        s_error = s - self.s_target
        cost = cs.sumsqr(task_cost @ s_error)
        return cost

    def desired_base_position_cost(self, H_B, task_cost=1):
        p_B_error = H_B[:3, 3] - self.H_B_target[:3, 3]
        cost = cs.sumsqr(task_cost @ p_B_error)
        return cost

    def desired_base_orientation_cost(self, H_B, task_cost=1):
        R_B_error = cs.inv_skew(H_B[:3, :3] @ cs.transpose(self.H_B_target[:3, :3]))
        cost = cs.sumsqr(task_cost @ R_B_error)
        return cost

    def task_space_targets_cost(self, H_B, s, t):
        cost = 0
        for model_target in self.task_space_targets.values():
            if model_target.time_steps[0] == -1 or t in model_target.time_steps:
                measured = self.model.forward_kinematics_fun(
                    model_target.target_object.frame_name
                )(H_B, s)
                cost = cost + cs.sumsqr(
                    np.diag(model_target.gain)
                    @ model_target.target_object.compute_error_from_transform(measured)
                )
        return cost

    def add_model_constraints(self):
        if self.optimization_flags.use_model_dynamics_constraint:
            g_fun = self.model.gravity_term_fun()
            M_fun = self.model.mass_matrix_fun()
            B = cs.vertcat(cs.MX.zeros(6, self.model.NDoF), cs.MX.eye(self.model.NDoF))

            for t in range(self.time_steps):
                g = g_fun(self.H_B[t], self.s[t])
                M = M_fun(self.H_B[t], self.s[t])
                Jc = self.contacts_Jacobian(self.H_B[t], self.s[t])
                M_inv = cs.solve(M, cs.MX.eye(M.size1()))
                J_weigthed_mass = Jc @ M_inv @ cs.transpose(Jc)
                J_weigthed_mass_inv = cs.solve(
                    J_weigthed_mass, cs.MX.eye(J_weigthed_mass.size1())
                )
                phi = cs.transpose(Jc) @ J_weigthed_mass_inv @ Jc @ M_inv

                # self.solver.subject_to(M_inv@(cs.MX.eye(phi.size1())-phi)@(B@self.tau[t]-g) == 0.0)

                self.solver.subject_to(
                    M_inv @ (cs.MX.eye(phi.size1()) - phi) @ (B @ self.tau[t] - g)
                    < 0.01
                )
                self.solver.subject_to(
                    M_inv @ (cs.MX.eye(phi.size1()) - phi) @ (B @ self.tau[t] - g)
                    > -0.01
                )

                self.solver.subject_to(
                    self.fc[t]
                    == J_weigthed_mass_inv @ (Jc @ M_inv @ (g - B @ self.tau[t]))
                )

                # self.solver.subject_to( M_inv@(g- B@self.tau[t] - cs.transpose(Jc)@self.fc[t]) == 0)

        if self.optimization_flags.use_joint_limits_constraint:
            for t in range(self.time_steps):
                self.solver.subject_to(self.s_min <= self.s[t])
                self.solver.subject_to(self.s_max >= self.s[t])

        if self.optimization_flags.use_task_space_constraints:
            for t in range(self.time_steps):
                for model_constraint in self.task_space_constraints.values():
                    if (
                        model_constraint.time_steps[0] == -1
                        or t in model_constraint.time_steps
                    ):
                        measured = self.model.forward_kinematics_fun(
                            model_constraint.constraint_object.frame_name
                        )(self.H_B[t], self.s[t])
                        self.solver.subject_to(
                            (
                                np.diag(model_constraint.activation_flags)
                                @ model_constraint.constraint_object.compute_error_from_transform(
                                    measured
                                )
                            )
                            == np.zeros(
                                model_constraint.constraint_object.get_error_size()
                            )
                        )

        if self.optimization_flags.use_contact_constraints:
            for t in range(self.time_steps):
                for model_contact in self.contact_constraints.values():
                    measured = self.model.forward_kinematics_fun(
                        model_contact.constraint_object.frame_name
                    )(self.H_B[t], self.s[t])
                    f_local = model_contact.constraint_object.get_local_wrench(
                        self.fc[t][model_contact.wrench_vector_indices],
                        measured[:3, :3],
                    )
                    if (
                        model_contact.constraint_object.contact_constraint_flags.use_local_CoP_constraint
                    ):
                        CoP = model_contact.constraint_object.get_local_CoP(
                            self.fc[t][model_contact.wrench_vector_indices],
                            measured[:3, :3],
                        )
                        self.solver.subject_to(
                            CoP[0]
                            >= model_contact.constraint_object.contact_area_size[0]
                        )
                        self.solver.subject_to(
                            CoP[0]
                            <= model_contact.constraint_object.contact_area_size[1]
                        )
                        self.solver.subject_to(
                            CoP[1]
                            >= model_contact.constraint_object.contact_area_size[2]
                        )
                        self.solver.subject_to(
                            CoP[1]
                            <= model_contact.constraint_object.contact_area_size[3]
                        )
                        self.solver.subject_to(
                            f_local[2]
                            > model_contact.constraint_object.normal_wrench_min
                        )
                    if (
                        model_contact.constraint_object.contact_constraint_flags.use_torsional_friction_constraint
                    ):
                        self.solver.subject_to(
                            f_local[5]
                            - model_contact.constraint_object.torsional_friction_coefficient
                            * f_local[2]
                            < 0
                        )
                        self.solver.subject_to(
                            -f_local[5]
                            - model_contact.constraint_object.torsional_friction_coefficient
                            * f_local[2]
                            < 0
                        )
                    if (
                        model_contact.constraint_object.contact_constraint_flags.use_static_friction_cone_constraint
                    ):
                        # TODO: handle this repeated constraint in case both friction cone and local cop are active
                        self.solver.subject_to(
                            f_local[2]
                            > model_contact.constraint_object.normal_wrench_min
                        )
                        self.solver.subject_to(
                            f_local[0]
                            < model_contact.constraint_object.static_friction_coefficient
                            * f_local[2]
                        )
                        self.solver.subject_to(
                            -f_local[0]
                            < model_contact.constraint_object.static_friction_coefficient
                            * f_local[2]
                        )
                        self.solver.subject_to(
                            f_local[1]
                            < model_contact.constraint_object.static_friction_coefficient
                            * f_local[2]
                        )
                        self.solver.subject_to(
                            -f_local[1]
                            < model_contact.constraint_object.static_friction_coefficient
                            * f_local[2]
                        )

        if self.optimization_flags.use_joint_simmetry_constraint:
            for t in range(self.time_steps):
                for pair in self.symmetric_joints_constraint_list:
                    self.solver.subject_to(
                        self.s[t][self.joints_name_list.index(pair[0])]
                        == self.s[t][self.joints_name_list.index(pair[1])]
                    )

        # Unit quaternion constraint
        for t in range(self.time_steps):
            self.solver.subject_to(
                cs.sumsqr(self.base_orientationQuat_position[t][:4]) == 1
            )

    def set_initial_variables(self):
        for t in range(self.time_steps):
            if self.optimization_flags.use_joints_initial_configuration:
                self.solver.set_initial(self.s[t], self.s_initial)
            if self.optimization_flags.use_tau_initial_configuration:
                self.solver.set_initial(self.tau[t], self.tau_initial)
            if self.optimization_flags.use_base_position_initial_configuration:
                self.solver.set_initial(
                    self.base_orientationQuat_position[t][4:7],
                    self.quatPos_B_initial[4:7],
                )
            if self.optimization_flags.use_base_orientation_initial_configuration:
                self.solver.set_initial(
                    self.base_orientationQuat_position[t][0:4],
                    self.quatPos_B_initial[0:4],
                )
            if self.optimization_flags.use_wrench_vector_initial_configuration:
                self.solver.set_initial(self.fc[t], self.fc_initial)

    def get_model_optimization_cost(self):
        cost = 0
        if self.optimization_flags.use_tau_minimization:
            for t in range(self.time_steps):
                # cost = cost + self.tau_minimization_cost(self.tau_(self.H_B[t], self.s[t], self.fc[t]), self.optimization_costs.tau_mimimization)
                cost = cost + self.tau_minimization_cost(self.tau[t])
        if self.optimization_flags.use_joints_decentralization_minimization:
            for t in range(self.time_steps):
                cost = cost + self.joint_decentralization_cost(
                    self.s[t],
                    self.optimization_costs.joint_decentralization_minimization,
                )
        if self.optimization_flags.use_target_joints_configuration_target:
            for t in range(self.time_steps):
                cost = cost + self.desired_joints_configuration_cost(
                    self.s[t],
                    self.optimization_costs.desired_joints_configuration_target,
                )
        if self.optimization_flags.use_target_base_position_target:
            for t in range(self.time_steps):
                cost = cost + self.desired_base_position_cost(
                    self.H_B[t], self.optimization_costs.desired_base_position_target
                )
        if self.optimization_flags.use_target_base_orientation_target:
            for t in range(self.time_steps):
                cost = cost + self.desired_base_orientation_cost(
                    self.H_B[t], self.optimization_costs.desired_base_orientation_target
                )
        if self.optimization_flags.use_task_space_targets:
            for t in range(self.time_steps):
                cost = cost + self.task_space_targets_cost(self.H_B[t], self.s[t], t)
        if self.optimization_flags.use_discontinuity_minimization:
            for t in range(self.time_steps - 1):
                cost = cost + self.joint_error_cost(
                    self.s[t],
                    self.s[t + 1],
                    self.optimization_costs.discontinuity_minimization,
                )
        if self.optimization_flags.use_joint_simmetry_target:
            for t in range(self.time_steps):
                cost = cost + self.joint_symmetry_cost(
                    self.s[t], self.optimization_costs.symmsteric_joints_target
                )
        return cost

    def set_solution(self, solution: cs.OptiSol):
        self.s_opti = []
        self.tau_opti = []
        self.H_B_opti = []
        self.fc_opti = []

        for t in range(self.time_steps):
            self.s_opti.append(solution.value(self.s[t]))
            self.tau_opti.append(solution.value(self.tau[t]))
            self.H_B_opti.append(
                FromQuaternionToMatrix()(
                    solution.value(self.base_orientationQuat_position[t])
                )
            )
            self.fc_opti.append(solution.value(self.fc[t]))
        self.optimization_flags.is_solved = True

    def print_targets(self, H_B, s, tau, fc):
        for target_name in self.task_space_targets:
            print("------ TARGET [", target_name, "] ------")
            if self.task_space_targets[target_name].time_steps[0] == -1:
                time_steps = range(self.time_steps)
            else:
                time_steps = self.task_space_targets[target_name].time_steps
            for t in time_steps:
                print("* [Time " + str(t) + "]")
                measured_transform = self.model.forward_kinematics_fun(
                    self.task_space_targets[target_name].target_object.frame_name
                )(H_B[t], s[t])
                self.task_space_targets[target_name].target_object.to_string(
                    measured_transform
                )
            print("* Cost        :")
            print(self.task_space_targets[target_name].gain)
            print("* Time Steps  :")
            print(self.task_space_targets[target_name].time_steps)
        if self.optimization_flags.use_tau_minimization:
            print("------ TARGET [Torque minimization] ------")
            for t in range(self.time_steps):
                print("* [Time " + str(t) + "]")
                print("Torque        ")
                print(tau[t])
                print("Torque Norm   ")
                print(cs.sumsqr(tau[t]))

    def print_constraints(self, H_B, s):
        for constraint_name in self.task_space_constraints:
            print("------ CONSTRAINT [", constraint_name, "] ------")
            if self.task_space_constraints[constraint_name].time_steps[0] == -1:
                time_steps = range(self.time_steps)
            else:
                time_steps = self.task_space_constraints[constraint_name].time_steps

            for t in time_steps:
                print("* [Time " + str(t) + "]")
                measured_transform = self.model.forward_kinematics_fun(
                    self.task_space_constraints[
                        constraint_name
                    ].constraint_object.frame_name
                )(H_B[t], s[t])
                self.task_space_constraints[
                    constraint_name
                ].constraint_object.to_string(measured_transform)
            print("* Activation  :")
            print(self.task_space_constraints[constraint_name].activation_flags)
            print("* Time Steps  :")
            print(self.task_space_constraints[constraint_name].time_steps)


class ContactDescription:
    """Constraint describing a contact between models"""

    def __init__(
        self,
        model_name_1,
        contact_name_1,
        frame_name_1,
        model_name_2,
        contact_name_2,
        frame_name_2,
        constrained_velocity_components: np.array = np.ones(6),
    ):
        # TODO: check constrained_velocity_components is a list of ones and zeros
        self.model_name_1 = model_name_1
        self.contact_name_1 = contact_name_1
        self.frame_name_1 = frame_name_1
        self.model_name_2 = model_name_2
        self.contact_name_2 = contact_name_2
        self.frame_name_2 = frame_name_2

        self.constrained_velocity_components = constrained_velocity_components


class MultiModelOptimizer:
    """optimizer class"""

    def __init__(self, time_steps):
        self.models = {}
        self.solver = cs.Opti()
        self.time_steps = time_steps

        # p_opts = {"expand": True}
        p_opts = {}
        s_opts = {"max_iter": 100000, "linear_solver": "ma27"}
        self.solver.solver("ipopt", p_opts, s_opts)

        self.contacts = {}

    def add_model(self, name, model: KinDynComputations, joints_name_list: list):
        self.models[name] = OptimizationModel(
            model, joints_name_list, self.solver, self.time_steps
        )

    def get_model(self, name) -> OptimizationModel:
        return self.models[name]

    def add_contact(
        self,
        contact_name: string,
        model_name_1: string,
        frame_name_1: string,
        model_name_2: string,
        frame_name_2: string,
        constrained_velocity_components,
    ):
        contact_name_1 = contact_name + "_" + model_name_1
        contact_name_2 = contact_name + "_" + model_name_2
        self.models[model_name_1].add_contact(
            contact_name_1, frame_name_1, constrained_velocity_components
        )
        self.models[model_name_2].add_contact(
            contact_name_2, frame_name_2, constrained_velocity_components
        )

        # Add contact description
        self.contacts[contact_name] = ContactDescription(
            model_name_1,
            contact_name_1,
            frame_name_1,
            model_name_2,
            contact_name_2,
            frame_name_2,
            constrained_velocity_components,
        )

    def add_contacts_constraint(self):
        for contact_name, contact_description in self.contacts.items():

            # Add position and orientation constraints
            for t in range(self.time_steps):
                fk_model1 = self.models[
                    contact_description.model_name_1
                ].model.forward_kinematics_fun(contact_description.frame_name_1)(
                    self.models[contact_description.model_name_1].H_B[t],
                    self.models[contact_description.model_name_1].s[t],
                )
                if np.any(contact_description.constrained_velocity_components[0:3]):
                    constraint_name = contact_name + "_linear_constraint_" + str(t)
                    self.models[contact_description.model_name_2].add_linear_constraint(
                        constraint_name, contact_description.frame_name_2, time=[t]
                    )
                    self.models[
                        contact_description.model_name_2
                    ].update_linear_constraint(
                        constraint_name,
                        fk_model1[0:3, 3],
                        contact_description.constrained_velocity_components[0:3],
                    )
                if np.any(contact_description.constrained_velocity_components[3:6]):
                    constraint_name = contact_name + "_SO3_constraint_" + str(t)
                    self.models[contact_description.model_name_2].add_SO3_constraint(
                        constraint_name, contact_description.frame_name_2, time=[t]
                    )
                    self.models[contact_description.model_name_2].update_SO3_constraint(
                        constraint_name,
                        fk_model1[0:3, 0:3],
                        contact_description.constrained_velocity_components[3:6],
                    )

            # Add wrench constraint
            for t in range(self.time_steps):
                wrench_vector_index_1 = (
                    self.models[contact_description.model_name_1]
                    .contact_constraints[contact_description.contact_name_1]
                    .wrench_vector_indices
                )
                wrench_vector_index_2 = (
                    self.models[contact_description.model_name_2]
                    .contact_constraints[contact_description.contact_name_2]
                    .wrench_vector_indices
                )
                self.solver.subject_to(
                    self.models[contact_description.model_name_1].fc[t][
                        wrench_vector_index_1
                    ]
                    == -self.models[contact_description.model_name_2].fc[t][
                        wrench_vector_index_2
                    ]
                )

    def solve(self):
        cost_function = 0

        self.add_contacts_constraint()

        for model in self.models.values():
            model.add_model_constraints()
            model.set_initial_variables()
            cost_function = cost_function + model.get_model_optimization_cost()

        self.solver.minimize(cost_function)
        try:
            sol = self.solver.solve()
        except:
            return False

        for model in self.models.values():
            model.set_solution(sol)
        return True


def toRadians(angle, unit_of_measurement: str):
    if unit_of_measurement == "radians":
        return angle
    elif unit_of_measurement == "degrees":
        return angle * np.pi / 180.0
    else:
        raise ValueError("[toRadians] : Measurement unit do not exist")
