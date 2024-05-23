import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from copy import copy
from tise.main import read_parameters_from_file

filenames_state = glob('tise/pm_tol/*.npy')
filenames_param = glob('tise/pm_tol/*.dipf')
ind_key = 'tolerance_pm'
print(filenames_param)
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
fig, ax = plt.subplots(1, 1)
ax2 = ax.twinx()
fig2, obs = plt.subplots(3, 2, sharex=True)
eps = []
energy = []
mu_x = []
mu_p = []
sig_x = []
sig_p = []
propr = []
tol_cg = []
tol_pm = []
ind_v = []
for state in filenames_state:
    for file in filenames_param:
        if state.split('.')[0] == file.split('.')[0]:
            params = copy(parameter_template)
            read_parameters_from_file(file, params)
            x = params['eps']*np.linspace(-params['N'], params['N'], 2*params['N']+1)
            if params['tolerance_pm'] == 1e-11:
                y = params['mu']/8*(x**2-1)**2
                ax2.plot(x, y, color='tab:red')
            ax.plot(x, np.load(state)/(np.sqrt(params['eps'])), label=rf'{ind_key} = {params[ind_key]}')
            eps.append(params['eps'])
            energy.append(params['energy'])
            mu_x.append(float(params['mu_x'].strip('[').strip(']')))
            mu_p.append(float(params['mu_p'].strip('[').strip(']')))
            sig_x.append(float(params['sig_x'].strip('[').strip(']')))
            sig_p.append(float(params['sig_p'].strip('[').strip(']')))
            propr.append(params['Propability_in_R'])
            tol_cg.append(params['tolerance_cg'])
            tol_pm.append(params['tolerance_pm'])
            ind_v.append(params[ind_key])
ax.set_xlim(-3, 3)
eps = np.array(eps)
energy = np.array(energy)
mu_x = np.array(mu_x)
mu_p = np.array(mu_p)
sig_x = np.array(sig_x)
sig_p = np.array(sig_p)
propr = np.array(propr)
tol_cg = np.array(tol_cg)
tol_pm = np.array(tol_pm)
ind_v = np.array(ind_v)
ax.grid()
fig.legend(bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)
ax.set_xlabel(r'x/r')
ax.set_ylabel(r'$\psi(x)/r^{1/2}$')
ax2.set_ylabel(r'$V(x)/\hbar \omega$', color='tab:red')
sorting = np.argsort(ind_v)
obs[0, 0].plot(ind_v[sorting], energy[sorting], marker='+')
obs[0, 0].set_xscale('log')
obs[0, 0].set_title(r'Energy/$\hbar \omega$')
obs[0, 0].set_xscale('log')
obs[0, 1].plot(ind_v[sorting], propr[sorting], marker='+')
obs[0, 1].set_xscale('log')
obs[0, 1].set_title(r'P(in R)')
obs[0, 1].set_xscale('log')
obs[1, 0].plot(ind_v[sorting], mu_x[sorting], marker='+')
obs[1, 0].set_xscale('log')
obs[1, 0].set_title(r'$\mu_x/r$')
obs[1, 0].set_xscale('log')
obs[1, 1].plot(ind_v[sorting], mu_p[sorting], marker='+')
obs[1, 1].set_xscale('log')
obs[1, 1].set_title(r'$\mu_p/(\hbar /r)$')
obs[1, 1].set_xscale('log')
obs[2, 0].plot(ind_v[sorting], sig_x[sorting], marker='+')
obs[2, 0].set_xscale('log')
obs[2, 0].set_title(r'$\sigma_x/r$')
obs[2, 0].set_xscale('log')
obs[2, 1].plot(ind_v[sorting], sig_p[sorting], marker='+')
obs[2, 1].set_xscale('log')
obs[2, 1].set_title(r'$\sigma_p/(\hbar /r)$')
obs[2, 1].set_xscale('log')

fig2.supxlabel(ind_key)
fig2.tight_layout()
plt.show()
