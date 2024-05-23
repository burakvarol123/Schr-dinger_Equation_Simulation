import numpy as np


def create_grid(N: int, D: int, dtype: np.dtype = np.int32) -> np.ndarray:
    """Create an array corrisponding to the coordinate grid

    Args:
        N (int): 2*N+1 is Number of gridpoints per dimension
        D (int): Dimension
        dtype (_type_, optional): datatype of array entries. Defaults to np.int32.

    Returns:
        np.ndarray: coordinate grid of shape (N, )*D + (D, )
    """
    dim = []
    num_points = 2 * N + 1
    x = np.linspace(-N, N, num_points, dtype)
    for i in range(D):
        dim.append(x)
    grid = np.meshgrid(*dim, indexing="ij")
    grid_2 = np.stack([vec for vec in grid], axis=-1)
    return grid_2


def calculate_grid_distance_squared(grid2: np.ndarray) -> np.ndarray:
    """Calculates the distance of each grid point to the origin squaqred

    Args:
        grid2 (np.ndarray): array corrisponding to the grid, created by create grid

    Returns:
        np.ndarray: n squared array
    """
    return np.sum(grid2**2, axis=-1)


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
    print(create_grid(5, 1).shape)
    print(create_grid(5, 2).shape)
