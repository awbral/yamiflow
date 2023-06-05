from parameters import Parameters
from background import Background
from component import Component
import os
from os.path import join
import shutil
import multiprocessing
import subprocess
import time
import numpy as np
from scipy.spatial import KDTree
from scipy.interpolate import RBFInterpolator
import json
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.size'] = 20
matplotlib.rcParams['axes.linewidth'] = 1
matplotlib.rcParams['lines.linewidth'] = 2
matplotlib.rcParams['axes.labelweight'] = 'bold'
matplotlib.rcParams['axes.labelpad'] = 15
matplotlib.rcParams['xtick.major.size'] = 6
matplotlib.rcParams['xtick.minor.size'] = 2.4
matplotlib.rcParams['ytick.major.size'] = 6
matplotlib.rcParams['ytick.minor.size'] = 2.4
matplotlib.rcParams['savefig.dpi'] = 600
matplotlib.rcParams['figure.autolayout'] = True
matplotlib.rcParams['figure.figsize'] = 15, 8


class Analysis(Parameters):
    def __init__(self, parameters):
        super().__init__(parameters)

        self.journal = self.parameters['journal']
        self.first_fiber = self.journal['first_fiber']
        self.last_fiber = self.journal['last_fiber']
        self.load_separately = self.journal['load_separately']
        self.v = int(self.journal['velocity_inlet'])
        self.p = self.journal['pressure_ambient']
        self.T = self.journal['temperature_ambient']
        self.rho = self.p / (287 * self.T)
        self.turb_intensity = self.journal['turbulent_intensity']
        self.turb_length_scale = self.journal['turbulent_length_scale']
        self.threshold = self.journal['gap_threshold']
        self.n_it_1 = self.journal['n_1sto_iter']
        self.n_it_2 = self.journal['n_2ndo_iter']

        self.post = self.parameters['post_processing']
        self.write_vtk = self.post['write_vtk']
        self.diameter_fraction = self.post['rbf_interpolator']['diameter_fraction']
        self.smoothing = self.post['rbf_interpolator']['smoothing']
        self.mnpf = self.post['max_nodes_per_face']

        self.fluent_process = None
        self.max_wait_time = 24 * 3600 * 3
        self.vtk_dir = None
        self.p_dir = join(self.w_dir, 'post_processing')

        self.p_force = np.zeros(self.dimensions)
        self.t_force = np.zeros(self.dimensions)
        self.moment = 0

        self.v_sweep = False
        if self.parameters.get('velocity_sweep') is not None:
            self.v_sweep = True
            self.velocity_array = np.array(self.parameters['velocity_sweep']['velocity_array'], dtype=int)
            self.v = self.velocity_array[0]

    def run(self):
        self.make_meshes()
        self.initialize()
        self.run_simulation()
        self.wait_message('data_stored')
        self.post_process()
        self.finalize()

    def make_meshes(self):
        # make and refine background
        print('Preparing background mesh...')
        background_mesh = Background(self.parameters)
        if self.mesh_background:
            background_mesh.generate_background()
        else:
            print('\t Skipping mesh generation...')

        # make component meshes
        if self.mesh_component:
            print('Preparing component meshes...')
            component_mesh = Component(self.parameters)
            for number in range(self.first_fiber, self.last_fiber + 1):
                component_mesh.generate_component(number)

    def initialize(self):
        print('Initializing case... \n')
        # create working directory
        if not os.path.exists(self.w_dir):
            os.makedirs(self.w_dir)
        self.remove_all_messages()

        # create post-processing directory
        if not os.path.exists(self.p_dir):
            os.makedirs(self.p_dir)

        # calculate angle and axis of rotation for fibers
        t_old = np.array([0, 0, 1])  # fibers are by default aligned with z-axis
        n = np.cross(t_old, self.yarn_axis)  # rotation axis: cross product
        norm_n = np.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2)
        theta = np.degrees(np.arcsin(norm_n))  # angle of rotation
        c = np.dot(t_old, self.yarn_axis)  # compute inner product to detect 0° or 180° angles
        if abs(c - 1) < 1e-5:  # yarn axis almost aligned with z-axis (about 0.25° difference): do nothing
            n = np.array([1, 0, 0])
            theta = 0.
        elif abs(c + 1) < 1e-5:  # yarn axis almost aligned with negative z-axis: do 180° rotation around x-axis
            n = np.array([1, 0, 0])
            theta = 180.
        else:  # normal rotation
            n = n / norm_n

        # prepare journal script
        if not (os.path.exists(join(self.mesh_dir, 'meshes_combined.cas.h5')) or self.load_separately):
                self.load_separately = True
        load_separately = '#f'
        if self.load_separately:
            load_separately = '#t'
        mu = 1.7894e-05 # viscosity of air in Fluent
        mu_t_over_v = self.rho * np.sqrt(3./2.) * self.turb_intensity / 100. * self.turb_length_scale * 0.09
        mu_t = mu_t_over_v * self.v
        turb_viscosity_ratio = mu_t/mu
        with open(join(self.dir_src, 'run_template.jou')) as infile:
            with open(join(self.w_dir, 'run.jou'), 'w') as outfile:
                for line in infile:
                    line = line.replace('|FIRST_FIBER|', str(self.first_fiber))
                    line = line.replace('|LAST_FIBER|', str(self.last_fiber))
                    line = line.replace('|YARN_LENGTH|', f'{self.l_yarn:.5e}')
                    line = line.replace('|LOAD_SEPARATELY|', load_separately)
                    line = line.replace('|MESH_DIR|', self.mesh_dir)
                    line = line.replace('|V|', f'{self.v}')
                    line = line.replace('|P|', f'{self.p:.2f}')
                    line = line.replace('|T|', f'{self.T:.2f}')
                    line = line.replace('|I|', f'{self.turb_intensity:.2f}')
                    line = line.replace('|MU_T|', f'{turb_viscosity_ratio:.5e}')
                    line = line.replace('|DZ|', f'{-self.origin - self.l_yarn / 2:.5e}')
                    line = line.replace('|ROT_ANGLE|', f'{theta:.5e}')
                    line = line.replace('|ROT_AXIS|', f'{n[0]:.5e} {n[1]:.5e} {n[2]:.5e}')
                    line = line.replace('|YARN_AXIS|', f'{self.yarn_axis[0]:.5e} {self.yarn_axis[1]:.5e} '
                                                       f'{self.yarn_axis[2]:.5e}')
                    line = line.replace('|GAP|', f'{self.threshold:.2e}')
                    line = line.replace('|CASE|', self.case_file)
                    line = line.replace('|1STO|', str(self.n_it_1))
                    line = line.replace('|2NDO|', str(self.n_it_2))
                    outfile.write(line)

        # velocity sweep: append journal file with loop that performs velocity sweep
        v_array_str = ''
        if self.v_sweep:
            for i in range(1, self.velocity_array.shape[0]):
                v_array_str += f' {self.velocity_array[i]}'
            subprocess.run('head -n-6 run.jou > run_adapted.jou; rm run.jou; mv run_adapted.jou run.jou',
                           executable='/bin/bash', shell=True, cwd=self.w_dir, env=None)
            with open(join(self.w_dir, 'run.jou'), 'a') as outfile:
                outfile.write('\n\n')
                outfile.write('; start velocity sweep loop\n')
                outfile.write('(define v_array (list' + v_array_str + '))\n')
                outfile.write(f'(define mu_t_over_v {mu_t_over_v:.5e})\n')
                outfile.write('(do ((k 0 (+ k 1 ))) ((>= k (length v_array)))\n')
                outfile.write("\t(rpsetvar 'udf/v (list-ref v_array k))\n")
                outfile.write("\t(display (%rpgetvar 'udf/v))\n")
                outfile.write('\t(set! case-name-update (string-append case-name '
                              '(format  #f "_v~a" (list-ref v_array k))))\n')
                outfile.write(f'\t(set! turb-visc-ratio (/ (* mu_t_over_v (list-ref v_array k)) {mu:.5e}))\n')
                outfile.write('\t(display turb-visc-ratio)\n')
                outfile.write('\t(ti-menu-load-string (format #f "/define/boundary-conditions/velocity-inlet inlet no '
                              'no yes yes no ~a no ~a no ~a no no yes ~a ~a\\n" (list-ref v_array k) '
                              'pressure-ambient temperature-ambient turb-intensity turb-visc-ratio))\n')
                outfile.write('\t(ti-menu-load-string (format #f "/solve/iterate ~a\\n" n-2ndo-iter))\n')
                outfile.write('\t(system "date")\n')
                outfile.write('\t(ti-menu-load-string (format #f "/file/write-data ~a\\n" case-name-update))\n')
                outfile.write('\t(system "date")\n')
                outfile.write('\t(ti-menu-load-string (format #f "/define/user-defined/execute-on-demand '
                              '\\"store_pressure_traction::post_process\\"\\n"))\n')
                outfile.write(')\n')
                outfile.write('(send_message "data_stored")\n')
                outfile.write('\n\n')
                outfile.write('; print time and exit\n')
                outfile.write('!date\n')
                outfile.write('/exit yes\n')
        else:
            v_array_str = f' {self.v}'

        # copy UDF
        shutil.copy(join(self.dir_src, 'post_process.c'), self.w_dir)

        # check number of cores
        if self.hostfile is not None:
            with open(self.hostfile) as fp:
                max_cores = len(fp.readlines())
        else:
            max_cores = multiprocessing.cpu_count()
        if self.cores < 1 or self.cores > max_cores:
            print(f'Number of cores incorrect, changed from {self.cores} to {max_cores}')
            self.cores = max_cores

        # refine background
        background_mesh = Background(self.parameters)
        background_mesh.refine_background()

        # prepare journal for visualization in post-processing
        with open(join(self.dir_src, 'save_pictures_template.jou')) as infile:
            with open(join(self.w_dir, 'save_pictures.jou'), 'w') as outfile:
                for line in infile:
                    line = line.replace('|YARN_LENGTH|', f'{self.l_yarn:.5e}')
                    line = line.replace('|YARN_RADIUS|', f'{self.r_yarn:.5e}')
                    line = line.replace('|ROT_ANGLE|', f'{theta:.5e}')
                    line = line.replace('|ROT_AXIS|', f'{n[0]:.5e} {n[1]:.5e} {n[2]:.5e}')
                    line = line.replace('|V_LIST|', v_array_str)
                    outfile.write(line)

    def run_simulation(self):

        # start actual simulation
        print('Starting simulation... \n')
        log = join(self.w_dir, 'run.log')
        cmd0 = f'{self.ml_cmd};'
        cmd1 = f'fluent -r{self.version} 3ddp -gu '
        cmd2 = f'-t{self.cores} -i run.jou &>> {log}'
        if self.hostfile is not None:
            cmd1 += f'-pib.ofed -cnf={self.hostfile} -ssh -mpi=intel '
        cmd = cmd0 + cmd1 + cmd2
        self.fluent_process = subprocess.Popen(cmd, executable='/bin/bash', shell=True, cwd=self.w_dir, env=None)
        # wait for case info to be exported after calculation
        self.wait_message('case_info_exported')

        # write bcs.txt file
        report = join(self.w_dir, 'report.sum')
        check = 0
        info = []
        with open(report, 'r') as file:
            for line in file:
                if check == 3 and line.islower():
                    name, thread_id, _ = line.strip().split()
                    if 'fiber' in name:
                        _, zone, number = name.split('_')
                        if zone == 'mantle':
                            info.append(' '.join((name, thread_id)))  # always consider fiber mantle
                        else:
                            fiber = np.loadtxt(join(self.f_dir, f'fiber_{number}.txt'))
                            if zone == 'head':
                                if np.abs(fiber[1, 2]) > 0.025 * self.l_yarn:
                                    # only consider fiber head if it is not at yarn start
                                    info.append(' '.join((name, thread_id)))
                            if zone == 'tail':
                                if np.abs(fiber[-1, 2] - self.l_yarn) > 0.025 * self.l_yarn:
                                    # only consider fiber tail if it is not at yarn end
                                    info.append(' '.join((name, thread_id)))
                if check == 3 and not line.islower():
                    break
                if check == 2:  # skip 1 line
                    check = 3
                if 'name' in line and check == 1:
                    check = 2
                if 'Boundary Conditions' in line:
                    check = 1
        with open(join(self.w_dir, 'bcs.txt'), 'w') as file:
            file.write(str(len(info)) + '\n')
            for line in info:
                file.write(line + '\n')
        self.send_message('thread_ids_written_to_file')

        self.wait_message('nodes_stored')
        # write list of gap regions to file
        mesh_list = []
        with open(join(self.w_dir, 'bcs.txt'), 'r') as infile:
            n_threads = int(infile.readline())

            for l_nr in range(n_threads):
                line = infile.readline()
                _, zone, f_id = line.split()[0].split('_')
                t_id = int(line.split()[1])
                if zone == 'mantle':
                    mesh = join(self.p_dir, f'nodes_thread{t_id}.dat')
                    mesh_list.append((f_id, t_id, np.loadtxt(mesh, skiprows=1)[:, :3]))
            infile.close()

        gap_pairs_str = ''
        n_gaps = 0
        for i in range(len(mesh_list) - 1):
            f_id_i = mesh_list[i][0]
            kdtree_i = KDTree(mesh_list[i][2])
            for j in range(i + 1, len(mesh_list)):
                f_id_j = mesh_list[j][0]
                if int(f_id_i.split(':')[0]) == int(f_id_j.split(':')[0]):
                    # these two regions belong to the same fiber, but got separated in different threads
                    continue
                kdtree_j = KDTree(mesh_list[j][2])
                indices = kdtree_i.query_ball_tree(kdtree_j, r=self.threshold)
                if any(indices):
                    n_gaps += 1
                    gap_pairs_str += '"' + f_id_i + '" "' + f_id_j + '"\n'

        with open(join(self.w_dir, 'gap_list.txt'), 'w+') as outfile:
            outfile.write(f'{n_gaps} \n')
            outfile.write(gap_pairs_str)
            outfile.close()

        self.send_message('gap_ids_written_to_file')

    def post_process(self):
        # get pressure and traction data
        print('Starting post_processing... \n')

        # get thread ids from bcs.txt file
        bcs_file = open(join(self.w_dir, 'bcs.txt'), 'r')
        lines = bcs_file.readlines()
        thread_ids = [int(line.strip().split()[1]) for line in lines[1:]]
        numbers = [int(line.strip().split()[0].split('_')[-1].split(':')[0]) for line in lines[1:]]

        # compute forces
        if self.write_vtk:
            self.vtk_dir = join(self.w_dir, 'VTK_files')
            if not os.path.exists(self.vtk_dir):
                os.mkdir(self.vtk_dir)

        if self.v_sweep:
            velocity_array = self.velocity_array
        else:
            velocity_array = np.array(self.v, dtype=int)

        force_matrix = np.zeros((velocity_array.shape[0], 3 * self.dimensions + 2))
        coefficient_matrix = np.zeros((velocity_array.shape[0], self.dimensions + 2))
        cutoff = 0  # ignore first 20 %
        yarn_start = (-self.yarn_axis / 2 + cutoff) * self.l_yarn

        for iteration in range(velocity_array.shape[0]):
            self.p_force = np.zeros(self.dimensions)
            self.t_force = np.zeros(self.dimensions)
            self.moment = 0
            v = velocity_array[iteration]
            for (thread, number) in zip(thread_ids, numbers):
                fiber = np.loadtxt(os.path.join(self.f_dir, f'fiber_{number:02d}.txt'))
                x_c = fiber[0, 0] * self.diameter_fraction  # characteristic length unit: fraction of fiber diameter
                epsilon = 3 / (x_c * np.sqrt(2))  # shape parameter proportional to 3σ of standard normal distribution

                print(f'\treading file "pressure_traction_v{v}_thread{thread}.dat"')
                data = np.loadtxt(join(self.p_dir, f'pressure_traction_v{v}_thread{thread}.dat'), skiprows=1)
                data[:, 3 * self.dimensions] -= self.p  # substract reference pressure to obtain relative pressures
                if np.sum(data[:, 3 * self.dimensions + 1]) > 0:
                    # only perform data interpolation if there is a gap present
                    data = self.interpolate_data(data, epsilon)
                yarn_start_array = np.repeat(yarn_start.reshape(1, -1), data.shape[0], axis=0)
                is_past_start = np.dot(data[:, :3] - yarn_start_array, self.yarn_axis) >= 0
                data = data[is_past_start]
                fp = np.zeros((data.shape[0], self.dimensions))
                ft = np.zeros((data.shape[0], self.dimensions))
                for j in range(self.dimensions):
                    # forces after interpolation (do compute forces even if no interpolation was performed)
                    fp[:, j] = data[:, self.dimensions + j] * data[:, 3 * self.dimensions]
                    self.p_force[j] += np.sum(fp[:, j])
                    ft[:, j] = data[:, 2 * self.dimensions + j]
                    self.t_force[j] += np.sum(ft[:, j])
                centroids = data[:, 0:self.dimensions]
                yarn_axis_array = np.repeat(self.yarn_axis.reshape(1, -1), data.shape[0], axis=0)
                yarn_start_array = np.repeat(yarn_start.reshape(1, -1), data.shape[0], axis=0)
                # substract axial force component from forces
                f_lat = (ft + fp) - yarn_axis_array * np.dot((ft + fp), self.yarn_axis).reshape(-1, 1)
                # calculate radial vector of every point: r = OP - (OP . yarn_axis).yarn_axis
                r = (centroids - yarn_start_array) - yarn_axis_array \
                    * np.dot((centroids - yarn_start_array), self.yarn_axis).reshape(-1, 1)
                self.moment += np.sum(np.cross(r, f_lat), axis=0)
                if self.write_vtk:
                    nodefile = os.path.join(self.p_dir, f'nodes_thread{thread}.dat')
                    vtkfile = os.path.join(self.vtk_dir, f'v{v}_thread{thread}.vtk')
                    self.write_vtk_files(data, nodefile, vtkfile, thread)

            # convert to linear force
            self.p_force /= (self.l_yarn * (1 - cutoff))
            self.t_force /= (self.l_yarn * (1 - cutoff))
            tot_force = self.p_force + self.t_force
            # residual to check whether moment is parallel to yarn axis
            norm_moment = np.linalg.norm(self.moment)
            residual = np.dot(self.moment / norm_moment, self.yarn_axis)
            # convert to linear moment, multiply with residual to get directionality right (residual should be +/- 1)
            self.moment = norm_moment / (self.l_yarn * (1 - cutoff)) * residual

            force_matrix[iteration, 1:] = np.hstack((self.p_force, self.t_force, tot_force, self.moment))
            force_matrix[iteration, 0] = v

            # calculate coefficients
            e_z = np.array([0, 0, 1])
            c = np.dot(e_z, self.yarn_axis)
            if abs(1 - abs(c)) <= 1e-5:  # almost axial flow (difference less than 0.25°)
                # drag direction is zero
                e_trans_drag = e_z * 0
                # lift direction is aligned with non-axial force coefficient
                e_trans_lift = tot_force - np.dot(tot_force, self.yarn_axis) * self.yarn_axis
                e_trans_lift /= np.linalg.norm(e_trans_lift)
            else:
                e_trans_lift = np.cross(self.yarn_axis, e_z)
                e_trans_lift /= np.linalg.norm(e_trans_lift)
                e_trans_drag = e_z - np.dot(e_z, self.yarn_axis) * self.yarn_axis
                e_trans_drag /= np.linalg.norm(e_trans_drag)
            print('Unit vector in drag direction: ', e_trans_drag)
            print('Unit vector in lift direction: ', e_trans_lift)

            c_long = np.dot(tot_force, self.yarn_axis) * 2 / (self.rho * v ** 2 * 2 * self.r_yarn * np.pi)
            if c < 0:  # yarn_axis and velocity direction partly oppose each other --> revert sign of coefficient
                c_long *= -1
            c_trans_d = np.dot(tot_force, e_trans_drag) * 2 / (self.rho * v ** 2 * 2 * self.r_yarn)
            c_trans_l = np.dot(tot_force, e_trans_lift) * 2 / (self.rho * v ** 2 * 2 * self.r_yarn)
            c_m = self.moment * 2 / (self.rho * v ** 2 * np.pi * (2 * self.r_yarn) ** 2)
            mu = 1.7894e-05
            Re = v * (2 * self.r_yarn) * self.rho / mu

            coefficient_matrix[iteration, :] = np.array([Re, c_long, c_trans_d, c_trans_l, c_m])
        # ------------------------
        # write forces and moment to file
        np.savetxt(join(self.w_dir, 'linear_force.dat'), force_matrix, fmt='%27.17e',
                   header='v [m/s]\tf_pres,x\tf_pres,y\tf_pres,z\tf_trac,x\tf_trac,y\tf_trac,z\tf_tot,x\tf_tot,y\t'
                          'f_tot,z [N/m]\tM,yarn_axis [N]'.expandtabs(28), comments='\t'.expandtabs(4))

        np.savetxt(join(self.w_dir, 'coefficients.dat'), coefficient_matrix, fmt='%27.17e',
                   header='Re_d\tc_longitudinal\tc_transversal,drag\tc_transversal,lift\tc_moment'.expandtabs(28),
                   comments='\t'.expandtabs(4))

        plt.figure()
        plt.plot(coefficient_matrix[:, 0], coefficient_matrix[:, 1], label=r'$c_{a}$')
        plt.plot(coefficient_matrix[:, 0], coefficient_matrix[:, 2], label=r'$c_{t,d}$')
        plt.plot(coefficient_matrix[:, 0], coefficient_matrix[:, 3], label=r'$c_{t,l}$')
        plt.plot(coefficient_matrix[:, 0], coefficient_matrix[:, 4], label=r'$c_{m}$')
        plt.tight_layout()
        plt.legend()
        plt.xlabel('Flow Reynolds number [-]')
        plt.ylabel('Force/moment coefficient [-]')
        plt.savefig(join(self.w_dir, f'{self.case_file}_coefficients.png'))

    def finalize(self):
        print('Writing parameters to "parameters_simulation.json"... \n')
        # write parameter file to case file (for future reference)
        with open(join(self.w_dir, 'parameters_simulation.json'), 'w') as outfile:
            json.dump(self.parameters, outfile)
        print('Done. Exiting... \n')

    def send_message(self, message):
        file = join(self.w_dir, message + '.fiber')
        open(file, 'w').close()
        return

    def wait_message(self, message):
        cumul_time = 0
        file = join(self.w_dir, message + '.fiber')
        while not os.path.isfile(file):
            time.sleep(0.01)
            cumul_time += 0.01
            if cumul_time > self.max_wait_time:
                raise RuntimeError(f'Simulation timed out, waiting for message: {message}.fiber')
            if self.fluent_process.poll() is not None:
                raise RuntimeError(f'Fluent process interrupted without sending message: {message}.fiber')
        os.remove(file)
        return

    def remove_all_messages(self):
        for file_name in os.listdir(self.w_dir):
            if file_name.endswith('.fiber'):
                file = join(self.w_dir, file_name)
                os.remove(file)

    # noinspection PyMethodMayBeStatic
    def neighbour_search(self, index, queue, data):
        """
        This function looks for the row numbers in array of the neighbours of face at row index (in same array)
        and adds them to queue. Neighbours share nodes
        :param index: row number of face for which neighbours are sought
        :param queue: queue list to add row numbers of neighbouring faces to
        :param data: numpy array containing all the data in file 'pressure_gap_timestepX_threadY.dat'
        :return: nothing
        """
        node_ids = data[index, -self.mnpf:]
        mask = np.isin(data[:, -self.mnpf:], node_ids).astype(int)
        for k in range(self.mnpf - 1):  # neighbouring faces have between 0 and n-1 nodes in common
            queue.extend(np.asarray(np.sum(mask, axis=1) == (k + 1)).nonzero()[0].tolist())
        return

    def interpolate_data(self, data, epsilon):
        """
        This function performs radial basis interpolation on pressure inside the gap region given the pressure on the
        edges of the gap
        :param data: numpy array containing all the data in file 'pressure_gap_timestepX_threadY.dat'
        :param epsilon: shape function of gaussian RBF kernel
        :return: updated data array with interpolated pressures
        """
        visited = np.zeros(data.shape[0])
        gap_ids = data[:, 3 * self.dimensions + 1].nonzero()[0]

        gap_list = []
        edge_list = []
        queue = []
        for i in gap_ids:
            if not visited[i]:  # new gap region reached
                visited[i] = 1
                gap_int_list = [i]
                edge_int_list = []

                self.neighbour_search(i, queue, data)

                while len(queue) > 0:
                    k = queue.pop(0)
                    if not visited[k]:
                        visited[k] = 1
                        if not data[k, 3 * self.dimensions + 1]:
                            # flow is calculated in this face, and lies next to gap face -> edge
                            edge_int_list.append(k)
                        else:  # gap face -> continue neighbour walk
                            gap_int_list.append(k)
                            self.neighbour_search(k, queue, data)
                # no more neighbouring gap faces -> start looking for next gap region
                gap_list.append(gap_int_list)
                edge_list.append(edge_int_list)

        for g, e in zip(gap_list, edge_list):
            # interpolate pressure in gap
            p_e = data[e, 3 * self.dimensions]
            p_e_dev = p_e - np.mean(p_e)
            interpolator = RBFInterpolator(y=data[e, 0:self.dimensions], d=p_e_dev,
                                           neighbors=(self.dimensions - 1) ** 4, smoothing=self.smoothing,
                                           kernel='gaussian', epsilon=epsilon, degree=-1)
            data[g, 3 * self.dimensions] = np.mean(p_e) + interpolator.__call__(data[g, 0:self.dimensions])
            # interpolate traction in gap
            for i in range(self.dimensions):
                t_e = data[e, 2 * self.dimensions + i]
                t_e_dev = t_e - np.mean(t_e)
                interpolator = RBFInterpolator(y=data[e, 0:self.dimensions], d=t_e_dev,
                                               neighbors=(self.dimensions - 1) ** 4, smoothing=self.smoothing,
                                               kernel='gaussian', epsilon=epsilon, degree=-1)
                data[g, 2 * self.dimensions + i] = np.mean(t_e) + interpolator.__call__(data[g, 0:self.dimensions])
        return data

    # noinspection PyMethodMayBeStatic
    def write_vtk_files(self, data, nodefile, vtkfile, thread_id):
        nodes = np.loadtxt(nodefile, skiprows=1)
        args = np.unique(nodes[:, -1].astype(int), return_index=True)[1].tolist()
        nodes = nodes[args]

        with open(vtkfile, 'w+') as outfile:
            outfile.write('# vtk DataFile Version 4.2\n')
            outfile.write(f'Thread {thread_id}\n')
            outfile.write('ASCII\nDATASET POLYDATA\n')
            outfile.write(f'POINTS {nodes.shape[0]} float\n')
            np.savetxt(outfile, nodes[:, 0:-1], fmt='%.10e')
            outfile.write(f'POLYGONS {data.shape[0]} {data.shape[0] * (self.mnpf + 1)} \n')
            for i in range(data.shape[0]):
                line = f'{self.mnpf}'
                for k in range(self.mnpf):
                    line += f' {np.where(nodes[:, -1] == data[i, -(self.mnpf - k)])[0][0]}'
                line += '\n'
                outfile.write(line)
            outfile.write(f'\nCELL_DATA {data.shape[0]}\n')
            outfile.write('SCALARS pressure float 1\n')
            outfile.write('LOOKUP_TABLE default\n')
            np.savetxt(outfile, data[:, 3 * self.dimensions], fmt='%.10e')
            outfile.write('SCALARS is_gap_face int 1\n')
            outfile.write('LOOKUP_TABLE default\n')
            np.savetxt(outfile, data[:, 3 * self.dimensions + 1], fmt='%d')
            outfile.write('VECTORS shear_stress float\n')
            np.savetxt(outfile,
                       data[:, 2 * self.dimensions:3 * self.dimensions] / data[:, self.dimensions:2 * self.dimensions],
                       fmt='%.10e')
            outfile.write('SCALARS shear_stress_mag float 1\n')
            outfile.write('LOOKUP_TABLE default\n')
            np.savetxt(outfile, np.linalg.norm(
                data[:, 2 * self.dimensions:3 * self.dimensions] / data[:, self.dimensions:2 * self.dimensions],
                axis=1), fmt='%.10e')
            outfile.close()
