import time

import idyntree.bindings as iDynTree


class iDynTreeVisualizer:
    def __init__(self) -> None:
        pass

    def prepare_visualization(self):
        self.viz = iDynTree.Visualizer()
        vizOpt = iDynTree.VisualizerOptions()
        vizOpt.winWidth = 1500
        vizOpt.winHeight = 1500
        self.viz.init(vizOpt)
        self.env = self.viz.enviroment()
        self.env.setElementVisibility("floor_grid", True)
        self.env.setElementVisibility("world_frame", False)
        self.viz.setColorPalette("meshcat")
        self.env.setElementVisibility("world_frame", False)
        self.frames = self.viz.frames()

    def modify_camera(self, length_board):
        center_point = iDynTree.Position(
            length_board / 2, length_board / 2, 0.0
        )  # Center of the chessboard (z = 0)
        camera_position = iDynTree.Position(
            length_board / 2, -length_board / 2 - 3, 4.0
        )  # Position in front of the chessboard (x, y, z)
        # Access and configure the camera
        camera = self.viz.camera()
        # Set the camera position and orientation
        camera.setPosition(camera_position)  # Camera in front of the chessboard
        camera.setTarget(center_point)  # Focus on the chessboard center
        self.viz.camera().animator().enableMouseControl(True)

    def add_model(
        self, joint_name_list, base_link, model_name, urdf_path=None, urdf_string=None
    ):
        if urdf_path is None and urdf_string is None:
            raise ValueError("Either urdf_path or urdf_string must be provided.")
        if urdf_path is not None and urdf_string is not None:
            raise ValueError("Only one of urdf_path or urdf_string must be provided.")

        mdlLoader = iDynTree.ModelLoader()
        if urdf_path is not None:
            mdlLoader.loadReducedModelFromFile(urdf_path, joint_name_list, base_link)
        elif urdf_string is not None:
            mdlLoader.loadReducedModelFromString(
                urdf_string, joint_name_list, base_link
            )
        self.viz.addModel(mdlLoader.model(), model_name)

    def add_multiple_instances(
        self,
        joint_name_list,
        base_link,
        urdf_path,
        model_name,
        number_of_robots,
        lengths_table_in_meters,
    ):
        self.coordinates = self.generate_equally_spaced_coordinates(
            n=number_of_robots, l=lengths_table_in_meters
        )
        self.name_of_robots = []
        for i in range(number_of_robots):
            name_i = "ergoCub" + str(i)
            self.name_of_robots.append(nam_i)
            idyntree_viz.add_model(
                joint_name_list=list(joints),
                base_link="root_link",
                urdf_path=model_path,
                model_name=name_i,
            )

    def update_multiple_instances(data_i):
        for idx_robot, name_i in enumerate(self.name_of_robots):
            coordinates_i = self.coordinates[idx_robot]
            data_robot_i = data_i[idx_robot]
            # robot_i = replicated_matrix[idx_robot]
            # data_i = robot_i[time_istant]
            idyntree_viz.update_model(
                np.array(data_robot_i.joint_positions),
                np.array(data_robot_i.base_transform),
                model_name=name_i,
                delta_x=coordinates_i[0],
                delta_y=coordinates_i[1],
            )

    def update_model(self, s, H_b, model_name, delta_x=0, delta_y=0):
        s_idyn = self.get_idyntree_joint_pos(s)
        H_b_idyn = self.get_idyntree_transf(H_b, delta_x, delta_y)
        self.viz.modelViz(model_name).setPositions(H_b_idyn, s_idyn)

    def get_idyntree_transf(self, H, delta_x, delta_y):
        pos_iDynTree = iDynTree.Position()
        R_iDynTree = iDynTree.Rotation()
        R_iDynTree.FromPython(H[:3, :3])
        pos_x = H[0, 3] + delta_x
        pos_y = H[1, 3] + delta_y
        pos_z = H[2, 3]
        pos_iDynTree.setVal(0, float(pos_x))
        pos_iDynTree.setVal(1, float(pos_y))
        pos_iDynTree.setVal(2, float(pos_z))
        for i in range(3):
            for j in range(3):
                R_iDynTree.setVal(j, i, float(H[j, i]))
        T = iDynTree.Transform()
        T.setPosition(pos_iDynTree)
        T.setRotation(R_iDynTree)
        T.setPosition(pos_iDynTree)
        return T

    def visualize(self, time_step):
        time_now = time.time()
        while time.time() - time_now < time_step and self.viz.run():
            self.viz.draw()

    def generate_equally_spaced_coordinates(self, n, l):
        """
        Generates n equally spaced (x, y) coordinates within a chessboard.

        Args:
        - n: Number of coordinates to generate.
        - l: Length of the chessboard (meters).

        Returns:
        - A list of n tuples, where each tuple is (x, y) representing a coordinate.
        """
        # Calculate the number of rows and columns in the grid
        grid_size = math.ceil(math.sqrt(n))  # Square root gives closest grid dimensions
        step = l / grid_size  # Distance between points in the grid

        coordinates = []
        for i in range(grid_size):
            for j in range(grid_size):
                # Compute the coordinate
                x = round(i * step + step / 2, 5)  # Center of the grid cell in x
                y = round(j * step + step / 2, 5)  # Center of the grid cell in y
                coordinates.append((x, y))
                if len(coordinates) == n:  # Stop if we've generated enough points
                    return coordinates
        return coordinates

    def get_idyntree_joint_pos(self, s):
        N_DoF = len(s)
        s_idyntree = iDynTree.VectorDynSize(N_DoF)
        for i in range(N_DoF):
            s_idyntree.setVal(i, s[i])
        return s_idyntree
