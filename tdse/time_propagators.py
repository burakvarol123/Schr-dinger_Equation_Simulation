import numpy as np
from typing import Callable
from tise.grid import create_grid
from tise.eigen import apply_inverted
from tise.observables import mu_O


def gen_hamilton(kap: float, pot: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Generates hamiltonian for givn parameters; needed for euler and cranck nicolson

    Args:
        k (float): _description_
        pot (np.ndarray): _description_

    Returns:
        Callable[[np.ndarray], np.ndarray]: _description_
    """
    def inner(psi: np.ndarray) -> np.ndarray:
        _sum = np.zeros(psi.shape, dtype=psi.dtype)
        d = len(psi.shape)
        for i in range(d):
            _sum += np.roll(psi, 1, axis=i) + np.roll(psi, -1, axis=i) - 2 * psi
        _sum *= -kap
        # _sum *= -b ** 2 / (2 * mu)
        return _sum + pot * psi
    return inner


def gen_euler_propoagator(pot: np.ndarray, kap: float,
                          dt: float,
                          **kwargs) -> Callable[[np.ndarray], np.ndarray]:
    """generate Euler propagator

    Args:
        pot (np.ndarray): potential as array
        k (float): prefactor of kinetic operator
        dt (float): timestep

    Returns:
        Callable[[np.ndarray], np.ndarray]: propagator for chosen parameters
    """
    hamiltonian = gen_hamilton(kap, pot)

    def inner(psi: np.ndarray):
        return psi - 1j * dt * hamiltonian(psi)
    return inner


def gen_cranic_propagator(pot: np.ndarray, kap: float,
                          dt: float,
                          **kwargs) -> Callable[[np.ndarray], np.ndarray]:
    """Generate crank nicolson propagator

    Args:
        pot (np.ndarray): potential as array
        k (float): prefactor of kinetic operator
        dt (float): timestep

    Returns:
        Callable[[np.ndarray], np.ndarray]: propagator for chosen parameters
    """
    hamiltonian = gen_hamilton(kap, pot)

    # Just to get the same optional parameter behaviour back ackain
    # while using **kwargs
    if 'tolerance' not in kwargs:
        kwargs['tolerance'] = 10e-10

    def not_inverted(_psi: np.ndarray) -> np.ndarray:
        return _psi + 0.25 * dt ** 2 * hamiltonian(hamiltonian(_psi))

    def inverted(_psi: np.ndarray) -> np.ndarray:
        return apply_inverted(not_inverted, _psi, kwargs['tolerance'], 100)[0]

    def inner(psi: np.ndarray) -> np.ndarray:
        n_q = inverted(psi)
        return n_q - 1j * dt * hamiltonian(n_q) - 0.25 * dt ** 2 * hamiltonian(hamiltonian(n_q))

    return inner


def gen_strangsplitting_propagator(pot: np.ndarray, kap: float,
                                   dt: float,
                                   **kwargs) -> np.ndarray:
    """generate strang splitting propagator

    Args:
        pot (np.ndarray): potential as array
        k (float): prefactor of kinetic operator
        dt (float): timestep

    Returns:
        Callable[[np.ndarray], np.ndarray]: propagator for chosen parameters
    """
    num = pot.shape[0]  # num = 2*N+1
    D = len(pot.shape)
    grid = create_grid(num // 2, D)
    # transform the grid so that -N -> 0, N -> 2N+1
    # to be consistent with the lecture
    # this transformation only matters for the kinetic energy propagator
    grid = grid + (num // 2) * np.ones_like(grid)
    pot_propagator = np.exp(-.5j * dt * pot)
    kin_propagator = np.exp(-1j * dt * 4 * kap * np.sum(np.sin(np.pi * grid / num)**2, axis=-1))

    def inner(psi: np.ndarray):
        eta = pot_propagator * psi
        Feta = np.fft.fftn(eta, norm='ortho')
        # for i in range(D):
        #     Feta = np.fft.fft(Feta, norm='ortho', axis=i)
        Fxeta = kin_propagator * Feta
        xeta = np.fft.ifftn(Fxeta, norm='ortho')
        # for i in range(D):
        #     xeta = np.fft.ifft(xeta, norm='ortho', axis=i)
        return pot_propagator * xeta
    return inner


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    def test_unitarity(grid, kappa, dt, number_of_tests):
        """
        tests the unitarity of the crank-nicoslon propagator by testing the norm conservation
        test is done for single timestep with random potentials and random (normalized) wavefunctions
        """
        max_error_unitarity = 0
        for i in range(number_of_tests):
            pot = np.random.random(grid.shape[:-1])
            psi = np.random.random(grid.shape[:-1])
            psi /= np.sqrt(np.vdot(psi, psi))
            propagate_cranic = gen_cranic_propagator(pot, kappa, dt)
            psi2 = propagate_cranic(psi)
            error_unitarity = np.abs(1 - np.vdot(psi2, psi2))
            if max_error_unitarity < error_unitarity:
                max_error_unitarity = error_unitarity
        return max_error_unitarity


    def test_ss_unitarity(grid, kappa, dt, number_of_tests):
        '''
        Tests the unitarity of the strang splitting algorithm
        by applying it multiple times to random wave functions
        '''
        max_error = 0
        for i in range(number_of_tests):
            potential = np.random.random(size=grid.shape[:-1])
            propagator = gen_strangsplitting_propagator(potential, kappa, dt)
            psi = np.random.random(size=grid.shape[:-1])
            psi /= np.sqrt(np.vdot(psi, psi))
            psi = propagator(psi)
            error = np.abs(np.vdot(psi, psi) - 1)
            if error > max_error:
                max_error = error
        return max_error


    def test_energy_conservation(grid, kappa, dt, number_of_tests):
        """
        Test the energy conservation of the Crank-Nicolson propagator.
        The test is done for single timestep with random potentials and random (normalized) wavefunctions
        """
        max_error = 0
        for i in range(number_of_tests):
            potential = np.random.random(size=grid.shape[:-1])
            psi = np.random.random(size=grid.shape[:-1])
            psi /= np.sqrt(np.vdot(psi, psi))
            hamilton = gen_hamilton(kap, potential)
            energy0 = mu_O(psi, hamilton)
            propagate_cranic = gen_cranic_propagator(potential, kappa, dt)
            psi = propagate_cranic(psi)
            energy1 = mu_O(psi, hamilton)
            error = np.abs(energy1 - energy0)
            if error > max_error:
                max_error = error
        return max_error

    D = 1
    N = 100
    potential = np.zeros((2 * N + 1,) * D)
    k = 100
    dt = 0.1
    grid = create_grid(N, D)
    psi = np.exp(2 * np.pi * 1j * 5 * grid[:, 0] / (2 * N + 1))
    propagator_eu = gen_euler_propoagator(potential, k, dt)
    propagator_cn = gen_cranic_propagator(potential, k, dt)
    propagator_ss = gen_strangsplitting_propagator(potential, k, dt)
    fig, axs = plt.subplots(3)
    psi_eu = propagator_eu(psi)
    psi_cn = propagator_cn(psi)
    psi_ss = propagator_ss(psi)
    axs[0].set_title('Euler')
    axs[1].set_title('Crank Nicolson')
    axs[2].set_title('Strang Splitting')
    for i in range(5):
        axs[0].plot(np.real(psi_eu), label=f'{i}')
        axs[1].plot(np.real(psi_cn))
        axs[2].plot(np.real(psi_ss))
        psi_eu = propagator_eu(psi_eu)
        psi_cn = propagator_cn(psi_cn)
        psi_ss = propagator_ss(psi_ss)
    fig.tight_layout()
    fig.legend(loc='upper right', title=r'$\dfrac{t}{dt}$')
    plt.show()

    n = 800
    d = 1
    b = 40
    s = 0.1
    k = np.array([0.5*n/b])
    n0 = np.array([-7*b*s])
    mu = 500
    dt = 0.00001
    n_t = 800
    # pot = gen_potential(n, d, b, mu)
    grid = create_grid(n, d, int)
    kap = b ** 2 / 2
    print("Error in cn unitarity is: ", test_unitarity(grid, kap, dt, 100))
    print("Error in ss unitarity is: ", test_ss_unitarity(grid, kap, dt, 100))
    print("Error in cn energy conservation is: ", test_energy_conservation(grid, kap ,dt, 100))
