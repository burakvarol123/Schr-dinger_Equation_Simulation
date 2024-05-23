import numpy as np
from typing import Callable
from tise.grid import create_grid, calculate_grid_distance_squared


def mu_O(psi: np.ndarray, apply_O: Callable[[np.ndarray], np.ndarray]
         ) -> float:
    """calculate the expectation value of an observable O

    Args:
        psi (np.ndarray): array representing the wavefunction
        apply_O (function): function which applies O to a wavefunction,
        which is given as argument

    Returns:
        float: expectation value <psi|O|psi>
    """
    Opsi = apply_O(psi)
    psiOpsi = np.vdot(psi, Opsi)
    return np.real(psiOpsi / np.vdot(psi, psi))


def var_O(psi: np.ndarray, apply_O: Callable[[np.ndarray], np.ndarray]
          ) -> float:
    """calculate the variance of an observable O

    Args:
        psi (np.ndarray): array representing the wavefunction
        apply_O (function): function which applies O to a wavefunction, which
        is given as argument

    Returns:
        float: variance <psi|O**2|psi>-<psi|O|psi>**2
    """
    norm = np.real(np.vdot(psi, psi))
    Opsi = apply_O(psi)
    psiOOpsi = np.vdot(Opsi, Opsi) / norm
    psiOpsi = np.vdot(psi, Opsi) / norm
    return np.real(psiOOpsi - psiOpsi**2)


def stat_psi(psi: np.ndarray,
             apply_O: object
             ) -> (float, float):
    """Basically both fuctions above combined

    Args:
        psi (np.ndarray): array representing the wavefunction
        apply_O (function): function which applies O to a wavefunction,
        which is given as argument

    Returns:
        float: expectation value <psi|O|psi>
        float: variance <psi|O**2|psi>-||<psi|O|psi>||**2
    """
    norm = np.real(np.vdot(psi, psi))
    Opsi = apply_O(psi)
    psiOOpsi = np.real(np.vdot(Opsi, Opsi)) / norm
    psiOpsi = np.vdot(psi, Opsi) / norm
    # np.tensordot(np.conj(Opsi), psi, len(psi[0])) vector observables?
    return np.real(psiOpsi), psiOOpsi - np.real(np.vdot(psiOpsi, psiOpsi))


def apply_x_1d(psi: np.ndarray, axis: int, r: np.float32) -> np.ndarray:
    """Applies the location operator x along an axis to psi

    Args:
        psi (np.ndarray): Wavefuction as a (N,)*D shape array
        axis (int): axis along whic to multiply
        r (np.float32): Gridscaling in units of lattice spacing
    Returns:
        np.ndarray: x_axis|psi>
    """
    dim = psi.shape[axis]
    assert dim % 2 == 1, "dim must be an odd number in the form 2N+1"
    x = np.linspace(-int((dim - 1) / 2), int((dim - 1) / 2), dim, dtype=np.int32)
    x_psi = 1 / r * mult_along_axis(psi, x, axis)
    return x_psi


def apply_p_1d(psi: np.ndarray, axis: int, r: np.float32) -> np.ndarray:
    """Applies the location operator p along an axis to psi

    Args:
        psi (np.ndarray): Wavefuction as a (N,)*D shape array
        axis (int): axis along which to multiply
        r (np.float32): Gridscaling in units of lattice spacing
    Returns:
        np.ndarray: p_axis|psi>
    """
    sym_der = .5j * (r * np.roll(psi, -1, axis) - r * np.roll(psi, 1, axis))
    return sym_der


def sphere_r_prob(psi: np.ndarray, r: float) -> float:
    """Propability to find a the particle within a radius of r for a given wf

    Args:
        psi (np.ndarray): D-Dimensional array representing the wf
        r (float): radius of the sphere in units of lattice discretisation

    Returns:
        float: Propability(Particle measured within r)
    """
    n_grid = create_grid(psi.shape[0] // 2, len(psi.shape))
    nsqr = calculate_grid_distance_squared(n_grid)
    psi_in_r = np.where(nsqr < r**2, psi, 0)
    return np.real(np.vdot(psi_in_r, psi_in_r)) / np.real(np.vdot(psi, psi))


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


def calculate_mean_position(state, R):
    '''
    Calculates the mean position of the state for an arbitrary number
    of dimensions
    '''
    d = len(state.shape)
    result = np.zeros(d)
    for axis in range(d):
        result[axis] = mu_O(state, lambda x: apply_x_1d(x, axis, R))
    return result


def calculate_var_position(state, R):
    '''
    Calculates the var position of the state for an arbitrary number
    of dimensions
    '''
    d = len(state.shape)
    result = np.zeros(d)
    for axis in range(d):
        result[axis] = var_O(state, lambda x: apply_x_1d(x, axis, R))
    return result


# single_slice is not used in the code here, but might be usful at some point
# i.e. it might be useful to calculate the probability of of a particle beeing inside a box of uneven sidelengths
def single_slice(
    a: np.ndarray,
    axis: int,
    start: int or None,
    stop: int or None,
    step: int or None = None,
) -> np.ndarray:
    """slices an array at a specific axis.
    -> a[(:,:,...,start:end:step,)]
                  ^axis position
    Args:
        a (np.ndarray): array to be sliced
        axis (int): axis along which the slice is to be slicing
        start (intorNone): start value for the slicing
        stop (intorNone): stop value of slice
        step (intorNone, optional): step size of the slice. Defaults to None.

    Returns:
        np.ndarray: sliced array
    """
    assert (axis >= 0) and (axis < a.ndim), f"axis: {axis}, should be 0 <= axis <= a.dim - 1"
    slicer = (slice(None),) * (axis) + (slice(start, stop, step),)
    return a[slicer]


# this function comes from stackoverflow and slightly modified
def mult_along_axis(A: np.ndarray, B: np.ndarray, axis: int
                    ) -> np.ndarray:
    """ multiplies a 1d array (B) along an axis of an Nd-array
    (A). Good for x.

    Args:
        A (np.ndarray): Array of of shape (N, )*D
        B (np.ndarray): Array of shape (N, )
        axis (int): axis along which to multiply; axis < D

    Returns:
        np.ndarray: _description_
    """
    assert axis <= A.ndim, \
        f"axis: {axis} is out of bound for np.ndarray with A.ndim: {A.ndim}"
    assert A.shape[axis] == B.size, \
        f"Shape of axis: {A.shape[axis]} do not match size of B: {B.size}"

    # np.broadcast_to puts the new axis as the last axis, so
    # we swap the given axis with the last one, to determine the
    # corresponding array shape. np.swapaxes only returns a view
    # of the supplied array, so no data is copied unnecessarily.
    shape = np.swapaxes(A, A.ndim - 1, axis).shape

    # Broadcast to an array with the shape as above. Again,
    # no data is copied, we only get a new look at the existing data.
    B_brc = np.broadcast_to(B, shape)

    # Swap back the axes. As before, this only changes our "point of view".
    B_brc = np.swapaxes(B_brc, A.ndim - 1, axis)

    return A * B_brc


if __name__ == "__main__":
    import numpy as np
    import observables
    import matplotlib.pyplot as plt

    DIM = 1
    N = 100
    X = np.linspace(-N, N, 2 * N + 1, dtype=np.int32)
    SIGMA = N / 10
    R = SIGMA
    MU = 0
    print(f'psi will have {(2*N+1)**DIM} entries')

    psi_1d = np.exp(-(X - MU)**2 / (4 * SIGMA**2))
    psi = np.ones((2 * N + 1, ) * DIM)

    for axis in range(DIM):
        psi = observables.mult_along_axis(psi, psi_1d, axis)

    prop_in_r = observables.sphere_r_prob(psi, SIGMA)
    print(prop_in_r)

    coordinate_label = ['x', 'y', 'z']

    fig, axs = plt.subplots(DIM, 2, figsize=(10, 5))
    for axis in range(DIM):
        slicer = list((slice(N, N + 1), ) * DIM)
        slicer[axis] = slice(None)
        axs[1].plot(X / SIGMA, np.imag(observables.apply_p_1d(psi, axis, R)
                                       [tuple(slicer)].flatten()), label=r'$\hat{p}\psi (x)$')
        axs[1].plot(X / SIGMA, np.real(observables.apply_x_1d(psi, axis, R)
                                       [tuple(slicer)].flatten()), label=r'$\hat{x}\psi (x)$')
        axs[0].plot(X / SIGMA, np.real(psi[tuple(slicer)].flatten()), label=r'$\psi(x)$')
        axs[0].plot(X / SIGMA, np.absolute(psi[tuple(slicer)].flatten())**2, label=r'$\|\psi (x)\|^2$')
        axs[0].set_ylabel('A.U.')
        axs[0].legend()
        axs[1].legend()
        mu_x = observables.mu_O(psi, lambda x: observables.apply_x_1d(x, axis, R))
        sig_x = np.sqrt(observables.var_O(psi, lambda x: observables.apply_x_1d(x, axis, R)))

        mu_p = observables.mu_O(psi, lambda p: observables.apply_p_1d(p, axis, R))
        sig_p = np.sqrt(observables.var_O(psi, lambda p: observables.apply_p_1d(p, axis, R)))

        print(mu_x, mu_p, sig_p, sig_x, sig_x * sig_p)

    fig.supxlabel(r'$\dfrac{x}{R}$')
    fig.suptitle(rf'$\sigma_x={{{sig_x:.3f}}}R, \sigma_p={{{sig_p:.3f}}}\frac{{\hbar}}{{R}}, \mu_x={{{mu_x:.3f}}}R$, ' +
                 rf'$\mu_p={{{mu_p:.3f}}}\frac{{\hbar}}{{R}}, P(\mathrm{{in \ sphere}})={{{prop_in_r:.3f}}}$')
    fig.tight_layout()
    plt.show()
