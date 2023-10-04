from urdfModifiers.core.fixedOffsetModifier import FixedOffsetModifier
from urdfModifiers.core.jointModifier import JointModifier
from urdfModifiers.core.modification import Modification
from urdfModifiers.utils import *
from urdfModifiers.geometry import *
# import numpy as np
from urdfpy import URDF 
import tempfile

class createUrdf:
    def __init__(self, original_urdf_path, save_gazebo_plugin = True) -> None:
        self.original_urdf_path = original_urdf_path
        self.save_gazebo_plugin = save_gazebo_plugin
        if(save_gazebo_plugin):
            path_temp_dummy = tempfile.NamedTemporaryFile(mode = 'w+')
            self.dummy_file = path_temp_dummy.name
        
            self.robot, self.gazebo_plugin_text = utils.load_robot_and_gazebo_plugins(
                self.original_urdf_path, self.dummy_file
            )
        else: 
            self.robot = URDF.load(self.original_urdf_path)

    def modify_lengths(self, length_multipliers: dict):
        for name, modification in length_multipliers.items():
            fixed_offset_modifier = FixedOffsetModifier.from_name(name, self.robot)
            fixed_offset_modifications = Modification()
            fixed_offset_modifications.add_dimension(modification, absolute=False)
            # Apply the modifications
            fixed_offset_modifier.modify(fixed_offset_modifications)

    def modify_densities(self, densities: dict):
        for name, modification in densities.items():
            fixed_offset_modifier = FixedOffsetModifier.from_name(name, self.robot)
            fixed_offset_modifications = Modification()
            fixed_offset_modifications.add_density(modification, absolute=True)
            # Apply the modifications
            fixed_offset_modifier.modify(fixed_offset_modifications)

    def write_urdf_to_file(self):
        """Saves the URDF to a valid .urdf file, also adding the gazebo_plugins"""
        tempFileOut = tempfile.NamedTemporaryFile(mode = 'w+')
        self.robot.save(tempFileOut.name)

        if(self.save_gazebo_plugin):
            lines = []
            with open(tempFileOut.name, "r") as f:
                lines = f.readlines()
                last_line = lines.pop()
                lines = lines + self.gazebo_plugin_text
                lines.append(last_line)
        robot_urdf_string = tempFileOut.read()
        robot_urdf_string=robot_urdf_string.replace("<?xml", '')
        robot_urdf_string=robot_urdf_string.replace("version='1.0'", '')
        robot_urdf_string=robot_urdf_string.replace("encoding='UTF-8'?>",'')
        return robot_urdf_string

    def reset_modifications(self):
        if(self.save_gazebo_plugin):
            path_temp_dummy = tempfile.NamedTemporaryFile(mode = 'w+')
            self.dummy_file = path_temp_dummy.name
            self.robot, self.gazebo_plugin_text = utils.load_robot_and_gazebo_plugins(
                self.original_urdf_path, self.dummy_file
            )
        else: 
            self.robot = URDF.load(self.original_urdf_path)
