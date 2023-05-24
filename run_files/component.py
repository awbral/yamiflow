from mesh import Mesh
import os
from os.path import join
import numpy as np
import math as m
import subprocess
from scipy.interpolate import splprep, splev


class Component(Mesh):
    def __init__(self, parameters):
        super().__init__(parameters)

        self.component = self.meshes['component']
        self.r = self.component['growth_rate']  # cell-to-cell growth ratio for geometric law (first guess)
        self.a = self.component['first_cell']  # first cell height (in m)
        self.b = self.component['last_cell']  # last cell height (in m)
        self.dz = self.component['axial_cell']  # axial cell thickness (in m)

    def generate_component(self, number):
        write_dir = join(self.mesh_dir, f'fiber_{number:02d}')
        if not os.path.exists(write_dir):
            os.mkdir(write_dir)

        fiber = np.loadtxt(join(self.f_dir, f'fiber_{number:02d}.txt'))
        fiber_length = fiber[0, 2]
        fiber_diameter = fiber[0, 0]
        fiber_radius = fiber_diameter / 2
        # b = a * r^(n-1)
        n = round(m.log10(self.b / self.a) / m.log10(self.r) + 1)  # number of cells (make integer)
        r = (self.b / self.a) ** (1 / (n - 1))  # actual cell-to-cell growth ratio
        L = self.a * ((1 - r ** n) / (
                1 - r))  # length of side needed to grow cell from size a to b with ratio r
        overset_radius = fiber_radius + L
        n_circle_arc = round(m.pi * overset_radius / (2 * self.b)) + 1
        n_axial = int(round(fiber_length / self.dz) + 1)
        # print(f'Axial divisions: ', n_axial)

        with open(join(self.dir_src, 'component_mesh_script_template.rpl'), 'r') as infile:
            with open(join(write_dir, f'fiber_{number:02d}.rpl'), 'w+') as outfile:
                for line in infile:
                    line = line.replace('|NUMBER|', f'{number:02d}')
                    line = line.replace('|FIBER_LENGTH|', f'{fiber_length:.5e}')
                    line = line.replace('|FIBER_RADIUS|', f'{fiber_radius:.5e}')
                    line = line.replace('|OVERSET_RADIUS|', f'{overset_radius:.5e}')
                    line = line.replace('|SPHERE_TAIL|', f'{fiber_length + overset_radius:.5e}')
                    line = line.replace('|BLOCK_TAIL|', f'{fiber_length + fiber_radius:.5e}')
                    line = line.replace('|WRITE_DIR|', write_dir)
                    line = line.replace('|OVERSET_RADIAL|', str((n + 1)))
                    line = line.replace('|START_CELL|', f'{self.a:.2e}')
                    line = line.replace('|END_CELL|', f'{self.b:.2e}')
                    line = line.replace('|CIRCLE_ARC|', str(int(n_circle_arc)))
                    line = line.replace('|SPHERE_ARC|', str(int(round(n_circle_arc / m.pi * 2))))
                    line = line.replace('|INSIDE_FIBER|', str(17))
                    line = line.replace('|MANTLE_AXIAL|', str(n_axial))
                    if '|' in line:
                        raise ValueError('Line still contains "|" after replacement: ' + line)
                    outfile.write(line)
                outfile.close()
            infile.close()

        cmd = f'{self.ml_cmd}; icemcfd -batch -script fiber_{number:02d}.rpl &>> fiber_{number:02d}.log'
        subprocess.run(cmd, shell=True, cwd=write_dir, executable='/bin/bash')

        # deform straight fiber
        mesh_file = join(write_dir, f'fiber_{number:02d}.msh')
        new_mesh_file = join(write_dir, f'fiber_{number:02d}_deformed.msh')

        end_of_nodes_nr = int(
            subprocess.getoutput('grep -n -x "))" ' + mesh_file).split(':')[0])  # find end of nodelist

        # get nodes from mesh file
        straight_nodes = np.loadtxt(mesh_file, skiprows=7, max_rows=end_of_nodes_nr - 8)  # load nodelist
        indices = np.argsort(straight_nodes[:, 2])
        nodes_sorted = straight_nodes[indices]  # sort nodes on increasing z-coordinate

        # get centerline information
        step = int(round(fiber[1:].shape[0] / (n_axial - 1)))
        smooth = 1e-12
        start = 1
        tck_u = splprep([fiber[start::step, 0], fiber[start::step, 1], fiber[start::step, 2]],
                        s=smooth * fiber[start::step].shape[0], k=3)
        u = np.linspace(0, 1, n_axial)
        centerline = np.array(splev(u, tck_u[0])).T  # (slightly) smoothed centerline coordinates
        der1 = np.array(splev(u, tck_u[0], der=1)).T  # first derivative: tangential vector
        deformed_nodes = np.zeros(nodes_sorted.shape)

        # initialize transformation for upstream points
        T_0 = np.eye(4)
        R_old = np.eye(4)
        t_old = np.array([0, 0, 1])
        j = 0
        t_new = der1[j] / np.linalg.norm(der1[j])
        n = np.cross(t_old, t_new)
        c = np.dot(t_old, t_new)
        N = np.array([
            [0, -n[2], n[1]],
            [n[2], 0, -n[0]],
            [-n[1], n[0], 0]
        ])
        K = np.eye(3) + N + np.dot(N, N) * 1 / (1 + c)
        R_new = np.eye(4)
        R_new[0:3, 0:3] = K
        T_1 = np.eye(4)
        T_1[:3, 3] = centerline[j]  # translate cross-section to centerline start point

        for i, point in enumerate(nodes_sorted):
            if point[2] <= 1.e-7:
                # head part
                M = T_1 @ R_new @ R_old @ T_0
                deformed_nodes[i] = (M @ np.hstack((point, np.array([1.]))).T)[:3]
            elif 1.e-7 < point[2] < fiber_length + 1.e-7:
                # mantle of yarn
                if int(round(point[2] / fiber_length * (n_axial - 1))) == j:
                    # point in same plane as previous point, so no extra rotation
                    T_0[2, 3] = -point[2]
                    M = T_1 @ R_new @ R_old @ T_0
                    deformed_nodes[i] = (M @ np.hstack((point, np.array([1.]))).T)[:3]
                elif int(round(point[2] / fiber_length * (n_axial - 1))) == j + 1:
                    # point in next plane, so new rotation matrix
                    t_old = t_new
                    R_old = R_new @ R_old
                    j += 1
                    T_0[2, 3] = -point[2]
                    t_new = der1[j] / np.linalg.norm(der1[j])
                    n = np.cross(t_old, t_new)
                    c = np.dot(t_old, t_new)
                    N = np.array([
                        [0, -n[2], n[1]],
                        [n[2], 0, -n[0]],
                        [-n[1], n[0], 0]
                    ])
                    K = np.eye(3) + N + np.dot(N, N) * 1 / (1 + c)
                    R_new = np.eye(4)
                    R_new[0:3, 0:3] = K
                    T_1[:3, 3] = centerline[j]
                    M = T_1 @ R_new @ R_old @ T_0
                    deformed_nodes[i] = (M @ np.hstack((point, np.array([1.]))).T)[:3]
                else:
                    # unwanted situation
                    raise ValueError(
                        f'j = {j}, while z/length*n_elements = {int(round(point[2] / fiber_length * (n_axial - 1)))}')
            elif fiber_length + 1.e-7 <= point[2]:
                # tail part
                if j < centerline.shape[0] - 1:
                    raise ValueError(f'j has not reached the end of the centerline: '
                                     f'j = {j} while centerline contains {centerline.shape[0]} elements.')
                else:
                    T_0[2, 3] = -fiber_length
                    M = T_1 @ R_new @ R_old @ T_0
                    deformed_nodes[i] = (M @ np.hstack((point, np.array([1.]))).T)[:3]

        deformed_nodes = deformed_nodes[np.argsort(indices)]  # put nodes back in initial order

        rf = open(mesh_file, 'r')
        lines = rf.readlines()
        rf.close()
        lines[-4] = lines[-4].replace('interface', 'overset')

        with open(new_mesh_file, 'w') as wf:
            wf.writelines(lines[:7])
            np.savetxt(wf, deformed_nodes, fmt='%.16e')
            wf.writelines(lines[end_of_nodes_nr - 1:])
            wf.close()
