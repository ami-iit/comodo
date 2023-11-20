import yarp
import numpy as np
import time


class robotInterface:
    def __init__(
        self, robot_name, local_port_prefix, axis_list, remote_control_board_list
    ) -> None:

        self.robotName = robot_name
        self.axis_list = axis_list
        self.remote_control_board_list = remote_control_board_list
        self.local_port_prefix = local_port_prefix
        self.ndof = len(self.axis_list)

        self.measurement_buffer = yarp.DVector()

    def open(self) -> bool:

        yarp.Network.init()
        if not yarp.Network.isNetworkInitialized() or not yarp.Network.checkNetwork(
            5.0
        ):
            raise ValueError("Failed to initialize the YARP network.")

        props = yarp.Property()
        props.put("device", "remotecontrolboardremapper")

        # Store the joint names
        axes_names = yarp.Bottle()
        axes_list = axes_names.addList()
        for axis in self.axis_list:
            axes_list.addString(axis)
        props.put("axesNames", axes_names.get(0))

        remote_control_boards = yarp.Bottle()
        remote_control_boards_list = remote_control_boards.addList()
        for control_board in self.remote_control_board_list:
            remote_control_boards_list.addString(control_board)
        props.put("remoteControlBoards", remote_control_boards.get(0))

        props.put("localPortPrefix", self.local_port_prefix)

        remote_control_board_props = props.addGroup("REMOTE_CONTROLBOARD_OPTIONS")
        remote_control_board_props.put("writeStrict", "on")

        self.robotDriver = yarp.PolyDriver(props)

        if not self.robotDriver.isValid():
            raise ValueError("Failed to open the RemoteControlBoardRemapper.")

        self.__init_interfaces()

    def __init_interfaces(self):

        self.iCtrlMode = self.robotDriver.viewIControlMode()
        self.iEnc = self.robotDriver.viewIEncoders()
        self.iPosCtrl = self.robotDriver.viewIPositionControl()
        self.iTrqCtrl = self.robotDriver.viewITorqueControl()

        time.sleep(1.0)

    def __set_all_control_modes(self, control_mode) -> bool:
        buffer = yarp.IVector()
        buffer.reserve(self.ndof)

        for _ in range(self.ndof):
            buffer.append(control_mode)

        return self.iCtrlMode.setControlModes(buffer)

    @staticmethod
    def to_array(yarp_vector) -> np.ndarray:
        return np.array([yarp_vector[i] for i in range(yarp_vector.size())])

    @staticmethod
    def to_yarp(numpy_array) -> yarp.DVector:
        buffer = yarp.DVector()
        buffer.reserve(numpy_array.size)
        _ = [buffer.append(float(val)) for val in numpy_array]
        return buffer

    def get_joints_position(self):

        self.measurement_buffer.resize(self.ndof)

        if not self.iEnc.getEncoders(self.measurement_buffer):
            raise ValueError("Failed to get joints position.")

        return np.deg2rad(self.to_array(self.measurement_buffer))

    def get_joints_velocity(self):

        self.measurement_buffer.resize(self.ndof)

        if not self.iEnc.getEncoderSpeeds(self.measurement_buffer):
            raise ValueError("Failed to get joints velocity.")

        return np.deg2rad(self.to_array(self.measurement_buffer))

    def get_joints_torque(self):

        self.measurement_buffer.resize(self.ndof)

        if not self.iTrqCtrl.getTorques(self.measurement_buffer):
            raise ValueError("Failed to get joints torque.")

        return self.to_array(self.measurement_buffer)

    def set_position_control_mode(self):
        return self.__set_all_control_modes(yarp.VOCAB_CM_POSITION)

    def set_torque_control_mode(self):
        return self.__set_all_control_modes(yarp.VOCAB_CM_TORQUE)

    def move(self, positions: np.ndarray) -> bool:
        if positions.size != self.ndof:
            raise ValueError("Failed to set move position.")

        return self.iPosCtrl.positionMove(self.to_yarp(np.rad2deg(positions)))

    def is_motion_done(self) -> bool:
        return self.iPosCtrl.checkMotionDone()

    def wait_motion(self):
        while not self.is_motion_done():
            time.sleep(0.1)

    def set_joints_torque(self, torques: np.ndarray) -> bool:
        if torques.size != self.ndof:
            raise ValueError("Failed to set joint torques.")

        return self.iTrqCtrl.setRefTorques(self.to_yarp(torques))
