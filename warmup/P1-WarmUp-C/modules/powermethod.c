/******************************************************************************
The function 'power_method' finds the largest eigenvalue and the corresponding
eigenvector of a hermitian positive-definite matrix.

Parameters:

v              Pointer to a complex array with N components (which is
               assumed to be already allocated). 'v' is thought as an
               N-component complex vector and is used as output.
               'v' returns an approximation of the eigenvector associated
               to the largest eigenvalue of 'A'.

mu             Pointer to a double, used as output. '*mu' returns an
               approximation of the largest eigenvalue of 'A'.

A(out,in)      Pointer to a function that calculates
                  out = A.in
               where 'in' and 'out' are pointers to complex arrays with N
               components, thought as N-component complex vectors, and 'A'
               is some positive-definite NxN matrix.

N              Number of components of the vector 'v'

eps,maxit      The PM is stopped when the maximum number of iterations 'maxit'
               is reached, or when |A.v-mu*v| <= res, where the simbol |z|
               represents the Euclidean norm of the vector z.

The function returns the number of iterations used by the PM.
******************************************************************************/
int power_method(double complex *v, double *mu,
                 void (*Af)(double complex *, double complex *), int N,
                 double eps, int maxit) {

  // add your code here
}
