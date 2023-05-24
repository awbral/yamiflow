from parameters import Parameters
import os


class Mesh(Parameters):
    def __init__(self, parameters):
        super().__init__(parameters)

        self.meshes = self.parameters['meshes']
        if not os.path.exists(self.mesh_dir):
            os.makedirs(self.mesh_dir)
