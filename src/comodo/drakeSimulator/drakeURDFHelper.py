# some useful functions to load robot models
# into the drake simulator

import logging
import os
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import meshio
import numpy as np
from amo_urdf import *
from pydrake.geometry import (
    AddCompliantHydroelasticProperties,
    AddContactMaterial,
    AddRigidHydroelasticProperties,
)
from pydrake.geometry import Box as BoxDrake
from pydrake.geometry import HalfSpace, ProximityProperties
from pydrake.math import RigidTransform
from pydrake.multibody.plant import CoulombFriction
from tqdm import tqdm


class DrakeURDFHelper:
    """A simple class to make a given URDF drake compatible"""

    def __init__(self):
        pass

    def load_urdf(self, urdf_path=None, urdf_string=None, mesh_path=None):
        # read the urdf
        self.urdf_path = str(urdf_path)
        # save the string temporarily to parse the xml
        # TODO: This function blows up if not careful during an optimisation
        if urdf_string is not None:
            tmpfile = tempfile.NamedTemporaryFile(mode="w+")
            tmpfile.write(urdf_string)
            self.urdf_path = tmpfile.name

        self.mesh_path = mesh_path
        self.urdf_out_path = os.getcwd() + "/urdfs"

        # load the xml from the urdf_string
        tree = ET.parse(self.urdf_path)
        self.root = tree.getroot()

    def convert_meshes_to_obj(self):
        """Currently drake only loads .obj files and not .stl files.
        This is a simple routine to allow this conversion"""
        # read the meshes and convert them from .stl to .obj so that they can be loaded to Drake
        logging.info("Converting all the meshes")
        for child in tqdm(
            self.root.findall("./link/visual/geometry/mesh")
            or self.root.findall("./link/collision/geometry/mesh")
        ):
            # extract mesh location and name associated with the urdf
            path = child.attrib["filename"]
            child_mesh_path = self.mesh_path + path.split("package://")[1]
            # convert the mesh and save it in the same folder
            child_mesh_name = child_mesh_path.replace("stl", "obj")
            if not Path(child_mesh_name).is_file():
                temp_mesh = meshio.read(child_mesh_path)
                temp_mesh.write(child_mesh_name)
            # replace the urdf to load .obj files
            child.set("filename", path.replace("stl", "obj"))

        logging.info("Converted all the meshes from .stl to .obj")

    def remove_all_collisions(self, list=[]):
        """Removes all the collision tags from the URDF"""
        for link in self.root.findall("./link"):
            if link.attrib["name"] not in list:
                for col in link.findall("./collision"):
                    link.remove(col)

    def fix_not_in_joint_list(self, red_joint_name_list):
        """Converts any joint not in the
        reduced joint name list (red_joint_name_list) into fixed joints"""
        joints = self.root.findall("./joint")
        for j in joints:
            if j.attrib["name"] not in red_joint_name_list:
                j.attrib["type"] = "fixed"

    def convert_xml_to_amo(self):
        """Converts the loaded xml to amo urdf format"""
        # convert the xml to amo urdf format
        self.amo_robot_dsl = xml_to_amo(self.root)
        # blockPrint()
        self.amo_robot = eval(self.amo_robot_dsl)
        # enablePrint()

    def add_acutation_tags(self):
        """Adds the actuation tags to the amo urdf"""
        # extract all the non-fixed joints
        joints = self.root.findall("./joint")
        joint_names = []
        for j in joints:
            if j.attrib["type"] != "fixed":
                joint_names.append([j.attrib["name"], j.attrib["type"]])

        # add actuation tags to the amo urdf
        for j in joint_names:
            actuator_name = "actuator_" + j[0]
            transmission_name = "transmission" + j[0]
            temp_trans = Transmission(
                Type("SimpleTransmission"),
                Actuator(Mechanicalreduction("1"), name=actuator_name),
                Transjoint(Hardwareinterface("EffortJointInterface"), j[0]),
                name=transmission_name,
            )
            # Add the transmission to the robot URDF
            self.amo_robot(temp_trans)

    def write_to_file(self, urdf_name):
        """Writes the amo_urdf to a file"""
        # create a folder in the current directory called urdfs
        if not os.path.exists(self.urdf_out_path):
            os.makedirs(self.urdf_out_path)
        filename = self.urdf_out_path + "/" + urdf_name + ".urdf"
        with open(filename, "w") as f:
            print(self.amo_robot, file=f)

        logging.info("Saved the urdf to {}".format(filename))

    def get_urdf_string(self):
        """Returns the urdf string"""
        return str(self.amo_robot)

    def get_sim_joint_order(self, plant, robot_model_sim):
        sim_joint_order = []
        for ii in plant.GetJointIndices(robot_model_sim):
            jj = plant.get_joint(ii)
            if jj.type_name() == "revolute":
                sim_joint_order.append(jj.name())
        return sim_joint_order

    # add ground with friction
    def add_ground_with_friction(self, plant):
        surface_friction_ground = CoulombFriction(
            static_friction=1.0, dynamic_friction=1.0
        )
        proximity_properties_ground = ProximityProperties()
        # https://drake.mit.edu/pydrake/pydrake.geometry.html?highlight=addcontactmaterial#pydrake.geometry.AddContactMaterial
        AddContactMaterial(
            dissipation=1e4,
            point_stiffness=1e7,
            friction=surface_friction_ground,
            properties=proximity_properties_ground,
        )
        AddRigidHydroelasticProperties(0.01, proximity_properties_ground)

        plant.RegisterCollisionGeometry(
            plant.world_body(),
            RigidTransform(),
            HalfSpace(),
            "ground_collision",
            proximity_properties_ground,
        )
        plant.RegisterVisualGeometry(
            plant.world_body(),
            RigidTransform(),
            HalfSpace(),
            "ground_visual",
            np.array(
                [0.5, 0.5, 0.5, 0.0]
            ),  # modify to set the RGBA color of the ground plane
        )

    def add_soft_feet_collisions(self, plant, xMinMax, yMinMax, foot_frames):
        surface_friction_feet = CoulombFriction(
            static_friction=1.0, dynamic_friction=1.0
        )
        proximity_properties_feet = ProximityProperties()
        AddContactMaterial(
            dissipation=1e4,
            point_stiffness=1e7,
            friction=surface_friction_feet,
            properties=proximity_properties_feet,
        )
        # https://drake.mit.edu/pydrake/pydrake.geometry.html?highlight=addcontactmaterial#pydrake.geometry.AddCompliantHydroelasticProperties
        AddCompliantHydroelasticProperties(
            resolution_hint=0.01,
            hydroelastic_modulus=1e8,
            properties=proximity_properties_feet,
        )

        for ii in foot_frames:
            # for collision
            plant.RegisterCollisionGeometry(
                plant.GetBodyByName(ii),
                RigidTransform(np.array([0, 0, 0])),
                BoxDrake((xMinMax[1] - xMinMax[0]) / 2, yMinMax[1] - yMinMax[0], 2e-3),
                ii + "_collision",
                proximity_properties_feet,
            )

            # for visual
            plant.RegisterVisualGeometry(
                plant.GetBodyByName(ii),
                RigidTransform(np.array([0, 0, 0])),
                BoxDrake((xMinMax[1] - xMinMax[0]) / 2, yMinMax[1] - yMinMax[0], 2e-3),
                ii + "_collision",
                np.array(
                    [1.0, 1.0, 1.0, 1]
                ),  # modify to set the RGBA color of the feet collisions
            )
