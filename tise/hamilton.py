import numpy as np
from typing import Tuple, Callable
from tise.grid import create_grid, calculate_grid_distance_squared


def calculate_potential(grid: np.ndarray, mu: float, eps: float) -> np.ndarray:
    '''
    Implements the potential mu/8 * (eps**2 * |x|^2 - 1)^2
    '''
    nsqr = calculate_grid_distance_squared(grid)
    potential = mu / 8 * (eps**2 * nsqr - 1)**2
    return potential


def kinetic_energy(mu: float, eps: float, psi: np.ndarray) -> np.ndarray:
    """
    The kinetic energy part, -1 / (2 * mu * eps ** 2) sum_k psi(n+e_k)+ psi(n-e_k)+2*psi(n)
    Args:
        mu: mwr**"2/h_bar
        eps: a / r
        psi: the wave function


    Returns: the kinetic energy matrix

    """
    _sum = np.zeros(psi.shape, dtype=psi.dtype)
    D = len(psi.shape)
    for k in range(D):
        _sum += np.roll(psi, 1, axis=k) + np.roll(psi, -1, axis=k) - 2 * psi
    _sum *= -1 / (2 * mu * eps ** 2)
    return _sum


def create_hamiltonian(grid: np.ndarray, mu: float, eps: float) -> Callable[[np.ndarray], np.ndarray]:
    '''
    Create the hamiltonian

    Args:
        grid: grid that spans from -N to N in 2*N+1 points in every dimension.
        mu: mwr**"2/h_bar
        eps: a / r

    Returns: the hamilton Callable[[np.ndarray], np.ndarray]
    '''
    pot = calculate_potential(grid, mu, eps)

    def inner(psi: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        return pot * psi + kinetic_energy(mu, eps, psi)
    return inner


if __name__ == "__main__":
    def test_linearity(hamiltonian: Callable[[np.ndarray], np.ndarray], shape: Tuple[int, ...]) -> float:
        '''
        Calculates the highest deviation between H(alpha*x + beta*y) and
        alpha*H(x) + beta*H(y) in 1000 runs.

        hamiltonian: The hamiltonian to test
        shape: The shape of the wavefunction
        returns: The max error of linearity relation H(ax+by)-aH(x)-bH(y)
        '''
        max_error = 0
        for i in range(100):
            alpha = np.random.random()
            beta = np.random.random()
            x = np.random.rand(*(shape[:-1]))
            y = np.random.rand(*(shape[:-1]))
            non_linearity = hamiltonian(alpha * x + beta * y) - alpha * hamiltonian(x) - beta * hamiltonian(y)
            error = np.vdot(non_linearity, non_linearity)
            if error > max_error:
                max_error = error
        return max_error

    def test_positivity(hamiltonian: Callable[[np.ndarray], np.ndarray], shape: Tuple[int, ...]):
        '''
        As in a discrete case every Callable[[np.ndarray], np.ndarray] can be expressed as a matrix
        one can test the positivity of a discrete Callable[[np.ndarray], np.ndarray] by testing
        its matrix representation for positivity.

        hamiltonian: The hamiltonian to test
        shape: The shape of the grid
        returns: The minimal braket <phi|H|phi> in location basis
        '''
        min_value = np.infty
        for i in range(100):
            psi = np.random.rand(*(shape[:-1]))
            value = np.vdot(psi, hamiltonian(psi))
            if value < min_value:
                min_value = value
        return min_value

    def get_kin_eigenvalue(mu: float, epsilon: float, N: int, k: np.ndarray):
        '''
        Get the eigenvalue of a planar wave with wavevector k
        '''
        return 2 / (mu * epsilon**2) * np.sum(np.sin(np.pi * k / N)**2)

    def const_vector_field(v: np.ndarray, shape: Tuple) -> np.ndarray:
        """This is a way to implement a constant vectorfield in python.

        Args:
            v (np.ndarray): vector to be broadcast on the grid
            shape (Tuple): shape of grid

        Returns:
            np.ndarray: array corrisponding to constant vector field
        """
        temp = v.reshape((len(v), ) + (1, ) * (len(shape) - 1))
        field = np.zeros(shape) + np.moveaxis(temp, 0, -1)
        return field

    def test_planar_waves(grid: np.ndarray, mu: float, eps: float):
        '''
        Test if the discretised planar wave functions are eigenfunctions
        of the discretised kinetic energy Callable[[np.ndarray], np.ndarray]

        grid: The grid on which planar waves are defined (only 1d accepted yet)
        mu, eps: The problem specific parameters
        returns: The max difference between normalised psi and Tpsi
        '''
        max_error = 0
        min_k = 0
        max_k = 1e4
        N = grid.shape[0]
        D = grid.shape[-1]
        for i in range(100):
            k = np.random.randint(min_k, max_k, size=D)
            nk = np.sum(grid * const_vector_field(k, grid.shape), axis=-1)
            psi = N ** (-D / 2) * np.exp(2 * np.pi * 1j * nk / grid.shape[0])
            Tpsi = kinetic_energy(mu, eps, psi)
            diff = get_kin_eigenvalue(mu, eps, grid.shape[0], k) * psi - Tpsi
            error = np.vdot(diff, diff)
            if error > max_error:
                max_error = error
        return max_error

    def test_hermiticity(hamiltonian: Callable[[np.ndarray], np.ndarray], shape: Tuple[int, ...]):
        '''
        Test if <Hpsi|psi> == <psi|Hpsi>

        hamiltonian: The hamiltonian to check
        shape: The shape of the grid
        '''
        max_error = 0
        for i in range(100):
            psi = np.random.rand(*(shape[:-1]))
            phi = np.random.rand(*(shape[:-1]))
            error = np.vdot(hamiltonian(psi), phi) - np.vdot(psi, hamiltonian(phi))
            if error > max_error:
                max_error = error
        return error

    # Specify here all parameters you want to run the tests for
    # Here Ds, min_Ns, max_Ns have to be of the same length
    Ds = [1, 2, 3]
    min_Ns = [2, 10, 8]
    max_Ns = [3, 12, 10]
    mus = [1.0, 2.0, 3.0]
    epss = [0.03, 0.04, 1.0]
    max_lin_error = 0
    min_braket = np.infty
    max_herm_error = 0
    max_pwe_error = 0
    for i, D in enumerate(Ds):
        for N in range(min_Ns[i], max_Ns[i]):
            for mu in mus:
                for eps in epss:
                    grid = create_grid(N, D)
                    hamiltonian = create_hamiltonian(grid, mu, eps)
                    lin_error = test_linearity(hamiltonian, grid.shape)
                    braket = test_positivity(hamiltonian, grid.shape)
                    herm_error = test_hermiticity(hamiltonian, grid.shape)
                    pwe_error = test_planar_waves(grid, mu, eps)
                    if lin_error > max_lin_error:
                        max_lin_error = lin_error
                    if braket < min_braket:
                        min_braket = braket
                    if herm_error > max_herm_error:
                        max_herm_error = herm_error
                    if pwe_error > max_pwe_error:
                        max_pwe_error = pwe_error

    print("Error in linearity: ", max_lin_error)
    print("Smallest <phi|H|phi>: ", min_braket)
    print("Error in hermiticity: ", max_herm_error)
    print("Error in planar wave eigenvectorness: ", max_pwe_error)
