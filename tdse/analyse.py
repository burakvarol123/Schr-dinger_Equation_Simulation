import sys
import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from copy import copy
from typing import List, Tuple, Callable
from tise.grid import create_grid
from tise.main import read_parameters_from_file
from tise.observables import (
        prop_transmitt,
        mu_O,
        var_O,
        calculate_mean_position,
        calculate_var_position)
from tdse.main import gen_potential, read_data
from tdse.time_propagators import gen_hamilton


def compare_versions(v1: str, v2: str) -> int:
    '''
    Compares two version strings,
    returns:
        -1 iff v1 < v2
        0  iff v1 = v2
        +1 iff v1 > v2
    '''
    tokens1 = v1.split('.')
    tokens2 = v2.split('.')
    for i in range(min(len(tokens1), len(tokens2))):
        if tokens1[i] < tokens2[i]:
            return -1
        if tokens1[i] > tokens2[i]:
            return 1
    return 0


def gen_calculate_observables(parameters: dict) -> Callable[[np.ndarray], Tuple[float, float, float, np.ndarray]] :
    '''
    Uses the parameters to generate a function to calculate the mean
    values of the hamiltonian, potential energy, kinetic energy
    as well as position operator
    '''
    potential = gen_potential(
        parameters['N'],
        parameters['D'],
        parameters['B'],
        parameters['mu'])

    def pot_op(state):
        return state * potential

    ham_op = gen_hamilton(parameters['B']**2 / 2, potential)
    kin_op = gen_hamilton(parameters['B']**2 / 2, 0 * potential)

    def inner(state: np.ndarray) -> Tuple[float, float, float, np.ndarray]:
        '''
        Calculates following observables:
            total energy
            kinetic energy
            potential energy
            mean position
            uncertainty of position
            norm
        '''
        total_energy = mu_O(state, ham_op)
        kinetic_energy = mu_O(state, kin_op)
        potential_energy = mu_O(state,  pot_op)
        position = calculate_mean_position(state, parameters['B'])
        norm = np.sqrt(np.real(np.vdot(state, state)))
        var_pos = np.sqrt(calculate_var_position(state, parameters['B']))
        return total_energy, kinetic_energy, potential_energy, \
            position, var_pos, norm
    return inner


def animate_1d(parameters: dict, data):
    '''
    Generate an animated gif from the simulation parameters
    and its corresponding simulation data.
    The animated gif will be store in $out.gif
    '''
    fig, [ax, ap] = plt.subplots(2)
    fig.tight_layout(pad=2.5)
    ax2 = ax.twinx()
    B = parameters['B']
    N = parameters['N']
    D = parameters['D']
    MU = parameters['mu']
    n_t = parameters['T']//parameters['dt']
    pot = gen_potential(N, D, B, MU)
    linev, = ax2.plot(np.linspace(-N/B, N/B, 2*N+1), pot, color='k')
    linex, = ax.plot(np.linspace(-N/B, N/B, 2*N+1), np.abs(data[0])**2, color='tab:blue')
    fpsi_eu = np.fft.fft(data[0], norm='ortho')
    linep, = ap.plot(np.linspace(-B/2, B/2, 2*N+1),
                     np.abs(np.roll(fpsi_eu, N))**2, color='tab:blue')
    ap.set_xticks(np.linspace(-B/2, B/2, 11))
    ax.set_xticks(np.linspace(-N/B, N/B, 11))
    n_frames = min(400, n_t)
    fig.legend()
    ax.set_xlabel(r'$\dfrac{x}{\Delta x}$')
    ax.set_ylabel(r'$|\psi(x)|^2 \Delta x$')
    ap.set_xlabel(r'$\dfrac{p_x}{h/\Delta x}$')
    ap.set_ylabel(r'$|\psi(p)|^2 \dfrac{2\pi\hbar}{\Delta x}$')
    ani_data = data[::int(n_t // n_frames)]

    def update_ani(i):
        print(f'generating frame {i}')
        linex.set_ydata(np.abs(ani_data[i])**2)
        fpsi = np.fft.fft(ani_data[i], norm='ortho')
        linep.set_ydata(np.abs(np.roll(fpsi, N))**2)
        return linex, linep
    ani = animation.FuncAnimation(fig, update_ani, frames=n_frames, interval=40)
    print('saving animation...')
    ani.save(filename=parameters['out']+".gif", writer="pillow")
    print('done')


def animate_2d(parameters: dict, data):
    fig, [ax, ap] = plt.subplots(2)
    fig.tight_layout(pad=2.5)
    B = parameters['B']
    N = parameters['N']
    D = parameters['D']
    n_t = int(parameters['T']//parameters['dt'])

    grid = create_grid(N, D, int)
    x = grid[:, :, 0]
    y = grid[:, :, 1]
    linex = ax.pcolormesh(x/B, y/B, np.abs(data[0])**2)
    fpsi = np.fft.fftn(data[0], norm='ortho')
    linep = ap.pcolormesh(x*B/(2*N), y*B/(2*N), np.abs(np.fft.fftshift(fpsi))**2)

    ax.set_xticks(np.linspace(-N/B, N/B, 11))
    ax.set_yticks(np.linspace(-N/B, N/B, 11))
    ap.set_xticks(np.linspace(-B/2, B/2, 11))
    ap.set_yticks(np.linspace(-B/2, B/2, 11))

    ax.set_xlabel(r'$\dfrac{x}{\Delta x}$')
    ax.set_ylabel(r'$\dfrac{y}{\Delta x}$')
    ap.set_xlabel(r'$\dfrac{p_x}{h/\Delta x}$')
    ap.set_ylabel(r'$\dfrac{p_y}{h/\Delta x}$')

    n_frames = min(400, n_t)
    fig.legend()
    fig.tight_layout()
    fig.colorbar(linex, ax=ax, label=r'$\dfrac{\psi(x)}{\Delta x^{-1}}$')
    fig.colorbar(linep, ax=ap, label=r'$\dfrac{\psi(p)}{\Delta x / h}$')
    ani_data = data[::int(n_t // n_frames)]

    def update_ani1(i):
        '''
        Update animation (matplotlib version above ???)
        '''
        print(f'generating frame {i}')
        fpsi = np.fft.fftn(ani_data[i], norm='ortho')
        linex.set_array((np.abs(ani_data[i])**2))
        linep.set_array((np.abs(np.fft.fftshift(fpsi))**2))
        return linex, linep

    def update_ani2(i):
        '''
        Update animation (matplotlib version below ???)
        '''
        print(f'generating frame {i}')
        fpsi = np.fft.fftn(ani_data[i], norm='ortho')
        linex.set_array((np.abs(ani_data[i])**2)[:-1, :-1].flatten())
        linep.set_array((np.abs(np.fft.fftshift(fpsi))**2)[:-1, :-1].flatten())
        return linex, linep
    if compare_versions(matplotlib.__version__, '3.7') < 0:
        ani = animation.FuncAnimation(fig, update_ani2, frames=range(n_frames), interval=40)
    else:
        ani = animation.FuncAnimation(fig, update_ani1, frames=range(n_frames), interval=40)
    print('saving animation...')
    ani.save(filename=parameters['out']+".gif", writer="pillow")
    print('done')


def analyse(parameters: dict, data: np.ndarray) -> dict:
    '''
    Analyse the simulation data for 
        total energy, kinetic energy, potential energy, mean position,=
        uncertainty of position, norm
    '''
    calculate_observables = gen_calculate_observables(parameters)
    observables = list(map(calculate_observables, data))
    observables = {
        'total_energy':         list(map(lambda x: x[0], observables)),
        'kinetic_energy':       list(map(lambda x: x[1], observables)),
        'potential_energy':     list(map(lambda x: x[2], observables)),
        'mean_position':        list(map(lambda x: x[3], observables)),
        'uncertainty_position': list(map(lambda x: x[4], observables)),
        'norm':                 list(map(lambda x: x[5], observables))}
    return observables



if __name__ == "__main__":
    filenames = sys.argv[1:]
    parameter_template = {
        'B': int,
        'mu': float,
        'N': int,
        'D': int,
        'tolerance_cg': float,
        'res_recal_iterations': int,
        's': float,
        'k': lambda s: np.fromstring(s.strip('[').strip(']'), \
                dtype=float, sep=','),
        'n0': lambda s: np.fromstring(s.strip('[').strip(']'), \
                dtype=float, sep=','),
        'dt': float,
        'T': float,
        'out': str 
    }
    for filename in filenames:
        parameters = copy(parameter_template)
        try:
            parameters = read_parameters_from_file(filename, parameters)
        except:
            print(f"The file {parameters['out']} could not be opened!")
            exit(-1)
        print(f"Read data from {parameters['out']}")
        data = read_data(parameters['out'] + '.npy')
        print("Calculate observables")
        observables = analyse(parameters, data)
        print("Generate energy plot")
        ts = np.arange(len(data)) * parameters['dt']
        fig, axis = plt.subplots()
        axis.set_xlabel(r"$\hat{t} = \frac{\hbar}{m\Delta x^2}$")
        axis.set_ylabel(r"$E_0 = \frac{\hbar^2}{m\Delta x^2}$")
        axis.grid()
        axis.plot(ts, observables['total_energy'], label='Total Energy')
        axis.plot(ts, observables['kinetic_energy'], label='Kinetic Energy')
        axis.plot(ts, observables['potential_energy'], label='Potential Energy')
        fig.legend()
        plt.savefig(f'{parameters["out"]}_energies.png')
        print("Generate location and variance of location plot")
        fig, axis = plt.subplots()
        axis.set_xlabel(r"$\hat{t} = \frac{\hbar}{m\Delta x^2}$")
        axis.set_ylabel(r"$\Delta x$")
        axis.grid()
        for d in range(parameters['D']):
            axis.plot(ts, list(map(lambda x: x[d], observables['mean_position'])), label=f'Mean Position {d + 1}')
            axis.plot(ts, list(map(lambda x: x[d], observables['uncertainty_position'])), label=f'Uncertainty of Position {d + 1}')
        fig.legend()
        plt.savefig(f'{parameters["out"]}_position.png')
        print("Generate norm plot")
        fig, axis = plt.subplots()
        axis.set_xlabel(r"$\hat{t} = \frac{\hbar}{m\Delta x^2}$")
        axis.grid()
        axis.plot(ts, observables['norm'], label='Norm')
        fig.legend()
        plt.savefig(f'{parameters["out"]}_norm.png')


        if parameters['D'] not in [1, 2]:
            print(f'Animation is only supported for 1d and 2d; D={parameters["D"]}')
        else:
            print("animate")
            {1: animate_1d, 2: animate_2d}[parameters['D']](parameters, data)
            break
