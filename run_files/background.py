from mesh import Mesh
import os
from os.path import join
import math as m
import subprocess
import numpy as np


class Background(Mesh):
    def __init__(self, parameters):
        super().__init__(parameters)

        self.background = self.meshes['background']
        self.n_coarse = self.background['n_coarse']  # approximate number of cells on coarsest mesh
        self.l_fine = self.background['l_fine']  # edge of the smallest cell on background
        self.ri_frac = self.background['ri_frac']  # radius of inner refinement zone as fraction of yarn radius

        if not os.path.exists(join(self.mesh_dir, 'background')):
            os.mkdir(join(self.mesh_dir, 'background'))

        self.domain = (11 * self.l_yarn) ** 3
        self.l_coarse = (self.domain / self.n_coarse) ** (1 / 3)
        self.n_refine = round(m.log10(self.l_coarse / self.l_fine) / m.log10(2.))
        self.r_arr = np.zeros(self.n_refine + 1)
        self.v_arr = np.zeros(self.n_refine + 1)
        self.r_arr[-1] = self.domain ** (1 / 3)
        self.v_arr[-1] = self.domain

        k = self.r_yarn / self.l_fine
        r_inner = self.ri_frac * k * self.l_fine

        for i in range(self.n_refine):
            if i == 0:
                self.r_arr[i] = r_inner
                self.v_arr[i] = (self.l_yarn + 2 * self.r_yarn) * r_inner ** 2 * np.pi
            else:
                self.r_arr[i] = self.r_arr[i - 1] + 15 * self.l_fine * 2 ** i
                self.v_arr[i] = (self.l_yarn + 2 * self.r_arr[i]) * self.r_arr[i] ** 2 * np.pi

    def generate_background(self):
        print('\t Generating mesh...')

        cmd = f'cd background; cp {self.dir_src}/background_template.rpl ./; ' \
              f'sed -e "s/|START|/{-self.r_arr[-1] / 2:.5e}/g" background_template.rpl > background.rpl; ' \
              f'sed -i "s/|LENGTH|/{self.r_arr[-1]:.5e}/g" background.rpl; ' \
              f'sed -i "s/|N_DIVS|/{int(round(self.r_arr[-1] / (self.l_fine * 2 ** self.n_refine) + 1))}/g" ' \
              f'background.rpl; cd .. '
        subprocess.run(cmd, shell=True, cwd=self.mesh_dir, executable='/bin/bash')

        cmd2 = f'cd background; ml purge; {self.ml_cmd}; icemcfd -batch -script background.rpl &> icem.log;'
        subprocess.run(cmd2, shell=True, cwd=self.mesh_dir, executable='/bin/bash')

    def refine_background(self):
        print('\t Writing refinement parameters...')
        with open(os.path.join(self.w_dir, 'refine_background.jou'), 'w+') as outfile:
            outfile.write('/file/set-tui-version "22.2" \n')
            outfile.write('/file/set-batch-options n n n \n')
            outfile.write('/file/read-case "background.msh" \n')
            outfile.write(f'/mesh/adapt/set/maximum-refinement-level {self.n_refine} \n')
            for n in range(1, self.n_refine + 1):
                r = self.r_arr[-n - 1]
                a_start = (-self.l_yarn / 2 - r) * self.yarn_axis
                a_end = (self.l_yarn / 2 + r) * self.yarn_axis
                outfile.write(f'/mesh/adapt/cell-registers/add "layer_{n}" type cylinder inside? yes axis-begin '
                              f'{a_start[0]:.10e} {a_start[1]:.10e} {a_start[2]:.10e} axis-end '
                              f'{a_end[0]:.10e} {a_end[1]:.10e} {a_end[2]:.10e} radius {r:.10e} quit quit \n')
                outfile.write(f'/mesh/adapt/cell-registers/refine "layer_{n}" \n')
            outfile.write(f'/file/write-case {self.case_file} \n')
            outfile.write('/exit \n')
            outfile.close()

        print('\t Refining background mesh...')
        log = join(self.w_dir, 'refine_background.log')
        cmd0 = f'{self.ml_cmd}; cp {self.mesh_dir}/background/background.msh .; '
        cmd1 = f'fluent -r{self.version} 3ddp -g '
        cmd2 = f'-t{int(np.ceil(self.cores * 0.83))} -i refine_background.jou &>> {log}; '
        if self.hostfile is not None:
            cmd1 += f'-pib.ofed -cnf={self.hostfile} -ssh -mpi=intel '
        cmd = cmd0 + cmd1 + cmd2
        subprocess.run(cmd, executable='/bin/bash', shell=True, cwd=self.w_dir, env=None)
