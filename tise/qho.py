'''
Calculates the energy of the discretizes quantum harmonic oszillator
in a nondimensionalised way to solidify the understanding
of nondimensionalisation.

Problems: Until now the exponential dampening of the hermite polynomial
has to be withouth the factor of 0.5 in order to get the right results.
This is different from the derivation in the notes and the wikipedia page
about the quantum harmonic oszillator, but with the factor of 0.5
the calculated result is exactly half of the expected energy.
'''

import numpy as np
from tise.grid import create_grid


def gen_psi0(grid, l, beta):
    # return (1 / ( np.pi * l**2))**(1 / 4) * np.exp(-beta**(-2) * grid**2)
    return (1 / ( np.pi * l**2))**(1 / 4) * np.exp(-0.5 * beta**(-2) * grid**2)

def gen_psi1(grid, l, beta):
    # return 2**(-0.5) * (1 / (np.pi * l**2))**(1/4) * np.exp(-beta**(-2) * grid**2) * (1 / beta) * grid
    return 2**(-0.5) * (1 / (np.pi * l**2))**(1 / 4) * np.exp(-0.5 * beta**(-2) * grid**2) * (1 / beta) * grid


def hamiltonian(psi, beta, grid):
    return -0.5 * beta**2 * (np.roll(psi, -1, 0) + np.roll(psi, 1, 0) - 2 * psi) + 0.5 * beta**(-2) * grid**2 * psi


if __name__ == "__main__":
    L = 100
    N = 10000000
    grid = create_grid(N, 1)
    # N = 2 * N + 1
    l = 1
    beta = l * N / L
    psi0 = gen_psi0(grid, l, beta)
    psi1 = gen_psi1(grid, l, beta)
    Hpsi0 = hamiltonian(psi0, beta, grid)
    Hpsi1 = hamiltonian(psi1, beta, grid)
    print(np.vdot(psi0, Hpsi0) / np.vdot(psi0, psi0))
    print(np.vdot(psi1, Hpsi1) / np.vdot(psi1, psi1))
