'''
'''
import numpy as np
from typing import Tuple, Callable


def get_hightest_eigenvalue(operator: Callable[[np.ndarray], np.ndarray],
                            test_vector: np.ndarray,
                            tolerance: float) -> Tuple[float, np.ndarray, int]:
    '''
    Calculates the highest eigenvalue of $operator using the power methode.

    operator: The operator to get the highest eigenvalue of
    test_vector: starting vector for for the power method
    tolerance: The tolerated error
    '''
    # The power methode uses the fact that every hermitian operator
    # has a complete set of eigenvectors, i.e. every vector
    # can be represented as a linear combination of eigenvectors
    # The highest eigenvalue is calculated by iteratively applying
    # the operator to the test vector and normalise the result.
    # Doing so all all components which are scaled less than the
    # highest eigenvalue are 'dampened to zero'
    normalised_test_vector = test_vector / np.linalg.norm(test_vector)
    Av = operator(normalised_test_vector)
    tmp_eigenvalue = np.linalg.norm(Av)
    res = Av - normalised_test_vector * tmp_eigenvalue
    niters = 0
    esqr = tolerance**2
    while esqr < np.real(np.vdot(res, res)):
        normalised_test_vector = Av / tmp_eigenvalue
        tmp_eigenvalue = np.linalg.norm(normalised_test_vector)
        print(f'niter_pm {niters} eigenv {tmp_eigenvalue} res {np.real(np.vdot(res, res))} ', end="")
        Av = operator(normalised_test_vector)
        res = Av - normalised_test_vector * tmp_eigenvalue
        niters += 1
    normalised_test_vector = normalised_test_vector / tmp_eigenvalue
    return tmp_eigenvalue, normalised_test_vector, niters


def apply_inverted(operator: Callable[[np.ndarray], np.ndarray],
                   b: np.ndarray, tolerance: float, res_recal_iterations=100,
                   max_iter: int = int(1e6)) -> Tuple[np.ndarray, int]:
    '''
    Applies $operator^-1 to $b using the conjugate gradient methode.

    operator: The operator to apply inverted
    b: The vector $operator^-1 should be applied on
    tolerance: The tolerated error
    res_recal_iterations: Every $res_recal_iterations iteration the residuum
                          will be calculated exactly. Fine tuning this
                          parameter might speed up convergence. Defaults to 100
    max_iter: maximum iterations for cg algorithm. Defaults to 1e6
    '''
    # The conjugate gradient methode solves the linear system Ax=b
    # by iteratively adjusting an initially guessed start vector
    # in the direction which reduces the residuum the most.
    x = np.zeros(b.shape)
    r = b - operator(x)
    d = r
    niters = 0
    rsqr = np.real(np.vdot(r, r))
    esqr = tolerance**2
    bsqr = np.real(np.vdot(b, b))
    while (rsqr > esqr*bsqr) and (niters < max_iter):  # could also use maximum norm; independent of N is nice
        z = operator(d)
        a = rsqr / np.real(np.vdot(d, z))
        x = x + a * d
        if niters % res_recal_iterations == 0:
            r = b - operator(x)
            rsqr = np.real(np.vdot(r, r))
            d = r
        else:
            r = r - a * z
        rsqr_old = rsqr
        rsqr = np.real(np.vdot(r, r))
        beta = rsqr / rsqr_old
        # by iteratively adjusting an initially guessed start vector
        # in the direction which reduces the residuum the most.
        d = r + beta * d
        niters += 1
    if niters >= max_iter:
        raise ArithmeticError('CG did not converge, check tolerances')
    return x, niters


def gen_invert_operator(operator: Callable[[np.ndarray], np.ndarray],
                        tolerance: float,
                        res_recal_iterations: int) -> Callable[[np.ndarray], np.ndarray]:
    '''
    Generates an inverted version of operator using
    the apply_inverted methode. The parameters $tolerance and
    $res_recal_iterations are dircetly passed to apply_inverted

    operator: The operator to be inverted
    tolerance: The tolerated error (needed as apply_inveted is used)
    res_recal_iterations: The res_recal_iterations parameter of apply_inveted
    '''
    def inner(x):
        result, niter_cg = apply_inverted(operator, x, tolerance, res_recal_iterations)
        print(f'cg iter {niter_cg} ')
        return result
    return inner


def get_smallest_eigenvalue(operator: Callable[[np.ndarray], np.ndarray],
                            test_vector: np.ndarray,
                            tolerance_cg: float, tolerance_pm: float,
                            res_recal_iterations: int) -> Tuple[float, np.ndarray, int]:
    '''
    Get the smalles eigenvalue calculating the highest eigenvalue of
    $operator^-1

    operator: The operator of which the smalles eigenvalue is calculated
    shape: The shape of the operator as a matrix
    tolerance_cg: The tolerated error in conjugate gradient algorithm
    tolerance_pm: The tolerated error in power method
    res_recal_iterations: The res_recal_iterations parameter of apply_inveted
    '''
    eigenvalue, eigenvector, iterations = get_hightest_eigenvalue(
        gen_invert_operator(operator, tolerance_cg, res_recal_iterations),
        test_vector,
        tolerance_pm)
    return 1 / eigenvalue, eigenvector, iterations


if __name__ == "__main__":
    tolerance = 1e-6

    def __test_get_highest_eigenvalue__(N):
        '''
        Tests the methode get highes eigenvalue by calculating the highest
        eigenvalue of random matrices and testing if these are indeed
        eigenvalues. Strictly speaking this is not testing if the found
        eigenvalue is the highest eigenvalue but due to the nature of the
        algorithm in question this is somewhat built-in
        '''
        max_error = 0
        for i in range(1000):
            A = np.random.random(size=(N, N)) + 1j * np.random.random(size=(N, N))
            A = np.dot(A, A.conj().T) + np.random.random() * np.identity(N)
            test_vector = np.random.random(size=(N, 1))
            eigenvalue, eigenvector, _ = get_hightest_eigenvalue(lambda x: A @ x, test_vector, tolerance)
            error = abs(1 - eigenvalue / np.vdot(eigenvector, A @ eigenvector))
            if error > max_error:
                max_error = error
        return max_error

    def __test_get_smallest_eigenvalue__(N):
        '''
        Tests the methode get_smallest_eigenvalue by calculating the smallest
        eigenvalue of random matrices and testing if these are indeed
        eigenvalues. Strictly speaking this is not testing if the found
        eigenvalue is the smallest eigenvalue but due to the nature of the
        algorithm in question this is somewhat built-in
        '''
        max_error = 0
        for i in range(100):
            A = np.random.random(size=(N, N)) + 1j * np.random.random(size=(N, N))
            A = np.dot(A, A.conj().T) + np.random.random() * np.identity(N)
            test_vector = np.random.random(size=(N, 1))
            eigenvalue, eigenvector, _ = get_hightest_eigenvalue(lambda x: A @ x, test_vector, tolerance)
            eigenvalue, eigenvector, _ = get_smallest_eigenvalue(lambda x: A @ x, test_vector, tolerance,
                                                                 tolerance, 100)
            ket = A @ eigenvector
            error = abs(1 - eigenvalue / np.vdot(eigenvector, ket))
            if error > max_error:
                max_error = error
            # max_error = error
        return max_error

    def __test_apply_inverted__(N):
        '''
        Test the apply_inverted methode by comparing the result with
        the multiplication of the proper inverse matrix multiplied to a test
        vector
        '''
        test_vector = np.random.random(size=(N, 1))
        test_vector = test_vector/np.linalg.norm(test_vector)
        max_error = 0
        for i in range(1000):
            A = np.random.random(size=(N, N)) + 1j * np.random.random(size=(N, N))
            A = np.dot(A, A.conj().T) + np.random.random() * np.identity(N)
            approximation, _ = apply_inverted(lambda x: A @ x, test_vector, tolerance, 100)
            exact = np.asarray(np.matrix(A).getI() @ test_vector)
            d = approximation - exact
            error = np.dot(d.T, d)
            if error > max_error:
                max_error = error
        return max_error

    # A = np.diag([1, 2, 3])
    # oA = lambda x: A @ x
    # oA_inv = gen_invert_operator(oA, 1e-8, 100)
    # print(oA(np.array([1, 1, 1])))
    # print(oA_inv(np.array([1, 1, 1])))
    N = 3
    test_get_highest_eigenvalue = __test_get_highest_eigenvalue__(N)
    test_apply_inverted = __test_apply_inverted__(N)
    test_get_smallest_eigenvalue = __test_get_smallest_eigenvalue__(N)
    print('test get highes eigenvalue: ', test_get_highest_eigenvalue)
    print('test apply_inverted: ', test_apply_inverted)
    print('test get lowest eigenvalue: ', test_get_smallest_eigenvalue)
