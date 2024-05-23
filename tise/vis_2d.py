import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from copy import copy
from main import read_parameters_from_file
from tise.grid import create_grid
from tise.hamilton import calculate_potential

filenames_state = glob('tise/2D/*.npy')
filenames_param = glob('tise/2D/*.dipf')

parameter_template = {
                        'eps': float,
                        'mu': float,
                        'N': int,
                        'D': int,
                        'res_recal_iterations': int,
                        'energy': float,
                        'mu_x': str,
                        'mu_p': str,
                        'sig_x': str,
                        'sig_p': str,
                        'Propability_in_R': float,
                        'tolerance_cg': float,
                        'tolerance_pm': float
                    }

params = copy(parameter_template)
read_parameters_from_file(filenames_param[0], params)
N = params['N']
D = params['D']
eps = params['eps']
mu = params['mu']
energy = (params['energy'])
mu_x = np.fromstring(params['mu_x'].strip('[').strip(']'), sep=',')
mu_p = np.fromstring(params['mu_p'].strip('[').strip(']'), sep=',')
sig_x = np.fromstring(params['sig_x'].strip('[').strip(']'), sep=',')
sig_p = np.fromstring(params['sig_p'].strip('[').strip(']'), sep=',')
propr = (params['Propability_in_R'])
tol_cg = (params['tolerance_cg'])
tol_pm = (params['tolerance_pm'])


state = np.load(filenames_state[0])
grid = create_grid(N, D)
potential = calculate_potential(grid, mu, eps)
r_space = np.linspace(-N * eps, N * eps, 2 * N + 1)
print(mu_x, mu_p, sig_x, sig_p, propr)
y, x = np.meshgrid(r_space, r_space)
fig, [ax1, ax2] = plt.subplots(1, 2)
c_potential = ax1.pcolormesh(x, y, potential)
fig.colorbar(c_potential, ax=ax1, location='bottom',
             label=r"$\dfrac{V(x,y)}{\hbar \omega}$", orientation='horizontal')
ax1.set_aspect('equal')
ax1.set_xticks([-3, -2, -1, 0, 1, 2, 3])
ax1.set_yticks([-3, -2, -1, 0, 1, 2, 3])
c_state = ax2.pcolormesh(x, y, (eps)**(-D / 2) * state)
fig.colorbar(c_state, ax=ax2, location='bottom',
             label=r"$\dfrac{\psi (x,y)}{r^{-1}}$", orientation='horizontal')
ax2.set_aspect('equal')
ax2.set_xticks([-3, -2, -1, 0, 1, 2, 3])
ax2.set_yticks([-3, -2, -1, 0, 1, 2, 3])
fig.supxlabel(r"$\dfrac{x}{\mathrm{r}}$")
ax1.set_ylabel(r"$\dfrac{y}{\mathrm{r}}$", fontsize="large")
fig.suptitle(rf'Calculated lowest Eigenenergy: $E_0 = {energy:.4f}\hbar \omega$')
fig.tight_layout()
plt.show()
