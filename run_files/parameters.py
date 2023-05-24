import os
from os.path import join
import numpy as np


class Parameters:
    def __init__(self, parameters):
        self.parameters = parameters
        self.package_dir = os.path.realpath(join(os.path.dirname(__file__), '..'))
        self.dir_src = join(self.package_dir, 'setup_files')

        self.general = self.parameters['general']

        self.w_dir = join(self.package_dir, self.general['case_dir'])
        self.case_file = self.general['case_file']
        self.mesh_dir = join(self.package_dir, self.general['mesh_dir'])
        self.f_dir = join(self.package_dir, self.general['geometry_dir'])

        self.ml_cmd = self.general['module_load_commands']
        self.version = self.general['fluent_version']
        self.hostfile = self.general.get('hostfile')

        self.cores = self.general['cores']
        self.dimensions = self.general['dimensions']
        self.dimensions = 3  # overwrite value

        self.r_yarn = self.general['yarn_radius']
        self.l_yarn = self.general['yarn_length']
        self.origin = self.general.get('origin', 0)
        self.yarn_axis = np.array(self.general['yarn_axis'])
        self.yarn_axis = self.yarn_axis / np.linalg.norm(self.yarn_axis)

        self.mesh_background = self.general['generate_background_mesh']
        self.mesh_component = self.general['generate_component_mesh']
