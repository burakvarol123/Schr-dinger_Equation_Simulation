import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from glob import glob
from copy import copy
from functools import reduce
from typing import List, Tuple, Callable
from tise.grid import create_grid, mult_along_axis, single_slice
from tise.main import read_parameters_from_file
from tdse.time_propagators import (
    gen_cranic_propagator,
    gen_euler_propoagator,
    gen_strangsplitting_propagator)
from tdse.initialize_psi import wavepacket


def read_data(fn: str) -> List[np.ndarray]:
    '''
    Reads the simulation data (stored as npy) from fn.
    A list of state vectors just feels a little bit more
    natural than a ndarray
    '''
    raw_data = np.load(fn)
    number_of_timesteps = raw_data.shape[0]
    result = []
    for i in range(number_of_timesteps):
        result.append(raw_data[i, :])
    return result


def gen_potential(n: int, d: int, b: int, mu: float,
                  axis: int = 0, origin: int or None = None) -> np.ndarray:
    """generate potential barrier of width b starting at origin

    Args:
        n (int): 2n+1 is the total number of points
        d (int): dimenstions
        b (int): width of the barrier
        mu (float): hight of the barrier
        axis (int, optional): axis normal to the barrier. Defaults to 0.
        origin (intorNone, optional): starting point of the barrier. Defaults to None.

    Returns:
        np.ndarray: vector containin the potential
    """
    temp = np.ones((2*n+1, )*d)
    vv = np.array([1]*b + [0]*(2*n+1-b))
    if origin is None:
        origin = n
    v = mult_along_axis(temp, np.roll(vv, origin), axis)
    return mu*v


def prop_transmitt(psi: np.ndarray, b: int, axis: int = 0,
                   origin: int or None = None) -> float:
    """calculates propabilty of particle having traveld through the barrier

    Args:
        psi (np.ndarray): wavefunctio
        b (int): width of barier
        axis (int, optional): axis normal to the barrier. Defaults to 0.
        origin (intorNone, optional): starting point of barrier. Defaults to None.

    Returns:
        float: propability that psi is in [origin+b, N]
    """
    if origin is None:
        origin = psi.shape[0]//2
    psi_cap = single_slice(psi, axis, origin + b, None)
    return np.real(np.vdot(psi_cap, psi_cap))


def run_calcs(propagator: Callable[[np.ndarray], np.ndarray], params: dict) ->  np.ndarray:
    """main function of the program. Runs simulation for all 3 propagators for
       given set of parameters.

    Args:
        propagator (Callable[[np.ndarray], np.ndarray]): The time propagator to use
        params (dict): n (int): 2n+1 is the number of points
                       d (int): number of dimensions
                       b (int): barrierwidth in number of points
                       s (float): width of pulse in units of b
                       k (np.ndarray): k-vector in units of 2pi/b
                       n0 (np.ndarray): mid-point of the pulse
                       mu (float): hight of the barrier
                       dt (float): timestep
                       n_t (int): number of timesteps

    Returns:
        An array of state vectors
    """
    n = params['N']
    d = params['D']
    b = params['B']
    s = params['s']
    k = params['k']
    n0 = params['n0']
    mu = params['mu']
    dt = params['dt']
    n_t = int(params['T'] / params['dt'])
    assert k.shape == (d, ), "dim k must equal d"
    assert n0.shape == (d, ), "dim n0 must equal d"
    grid = create_grid(n, d, int)
    psi = wavepacket(grid, b, s, k, n0)
    pot = gen_potential(n, d, b, mu)
    kap = b ** 2 / 2
    psi_t = [psi]
    for i in range(n_t):
        psi = propagator(psi)
        psi_t.append(psi)
        print(i)
    return np.stack(psi_t)


if __name__ == "__main__":
    filenames = [glob(s) for s in sys.argv[1:]]
    filenames = reduce(lambda x, y: x + y, filenames, [])
    parameter_template = {
        'B': int,
        'mu': float,
        'N': int,
        'D': int,
        'tolerance_cg': float,
        'res_recal_iterations': int,
        's': float,
        'k': lambda s: np.fromstring(s.strip('[').strip(']'),\
                dtype=float, sep=','),
        'n0': lambda s: np.fromstring(s.strip('[').strip(']'),\
                dtype=float, sep=','),
        'dt': float,
        'T': float,
        'gen_propagator': lambda s: {
            'euler': gen_euler_propoagator,
            'strang_splitting': gen_strangsplitting_propagator,
            'cranic': gen_cranic_propagator}[s],
        'out': str
    }
    for filename in filenames:
        parameters = copy(parameter_template)
        parameters = read_parameters_from_file(filename, parameters)

        pot = gen_potential(
            parameters['N'],
            parameters['D'],
            parameters['B'],
            parameters['mu'])
        propagator = parameters['gen_propagator'](
                pot, parameters['B']**2 / 2, parameters['dt'],
                tolerance=parameters['tolerance_cg'])
        result = run_calcs(propagator, parameters)
        print('saving to '+parameters['out']+'.npy')
        np.save(parameters['out'], result)
