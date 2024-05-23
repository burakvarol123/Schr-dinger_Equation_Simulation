import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import Tuple
from tise.grid import create_grid

matplotlib.rcParams['mathtext.fontset'] = "cm"


def const_vector_field(v: np.ndarray, shape: Tuple) -> np.ndarray:
    """This is a way to implement a constant vectorfield in python.

    Args:
        v (np.ndarray): vector to be broadcast on the grid
        shape (Tuple): shape of grid

    Returns:
        np.ndarray: array corrisponding to constant vector field
    """
    temp = v.reshape((len(v), )+(1, )*(len(shape) - 1))  # extending the dimensions of v
    field = np.zeros(shape) + np.moveaxis(temp, 0, -1)  # moving v to the right axis and extend it over
    return field


def wavepacket(grid: np.ndarray, b: int, s: float, k: np.ndarray, n0: np.ndarray) -> np.ndarray:
    """Implementing a Wavepacket according to parameters in given dimension

    Args:
        grid (np.ndarray): coordinate grid
        b (int): length unit
        s (float): sigma of the packet in b
        k (np.ndarray): k-vector in units of 1/b
        n0 (np.ndarray): midpoint of the packet

    Returns:
        np.ndarray: array form of the packet
    """
    k = k/b
    n0_field = const_vector_field(n0, grid.shape)
    k_field = const_vector_field(k, grid.shape)
    gaussexp = - 1/(4*(s*b)**2)*np.sum((grid-n0_field)**2, axis=-1)
    phase = 2*np.pi*np.sum(grid*k_field, axis=-1)
    psi = np.exp(gaussexp + 1j*phase)
    return psi


if __name__ == "__main__":
    D = 1
    N = 250
    B = 50
    S = 1
    N0 = np.array([0])
    K = np.array([1])
    grid = create_grid(N, D)
    psi = wavepacket(grid, B, S, K, N0)
    plt.plot(1/B*grid.flatten(), psi.real, label=r'$\dfrac{\Re (\psi(x,y))}{a^{1/2}}$')
    plt.plot(1/B*grid.flatten(), psi.imag, label=r'$\dfrac{\Im (\psi(x,y))}{a^{1/2}}$')
    plt.legend(fontsize=12)
    plt.grid()
    plt.show()
    D = 2
    fig, [ax1, ax2] = plt.subplots(1, 2)
    N0 = np.array([3, 5])
    K = np.array([1, 1])
    grid = create_grid(N, D)
    x = grid[:, :, 0] / B
    y = grid[:, :, 1] / B
    psi = wavepacket(grid, B, S, K, N0)
    c_real = ax1.pcolormesh(x, y, psi.real)
    fig.colorbar(c_real, ax=ax1, location='bottom',
                 label=r"$\dfrac{\Re (\psi(x,y))}{a}$", orientation='horizontal')
    c_imag = ax2.pcolormesh(x, y, psi.imag)
    fig.colorbar(c_imag, ax=ax2, location='bottom',
                 label=r"$\dfrac{\Im (\psi(x,y))}{a}$", orientation='horizontal')
    plt.show()
