#include "linalg.h"
#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/******************************************************************************
The function 'conjugate_gradient' solves the equation A.x=b for x, using the
conjugate gradient (CG) algorithm, where 'x' and 'b' are vectors and 'A' is a
hermitian positive-definite matrix.

Parameters:

x,b            Pointers to complex arrays with N components (they are both
               assumed to be already allocated). 'x' and 'b' are thought as
               N-component complex vectors. 'b' is used as input, while
               'x' is used as output. 'x' returns an approximation of
               'A^{-1}.b'.

A(out,in)      Pointer to a function that calculates
                  out = A.in
               where 'in' and 'out' are pointers to complex arrays with N
               components, thought as N-component complex vectors, and 'A'
               is some positive-definite NxN matrix.

N              Number of components of the vectors 'x' and 'b'

eps,maxit      The CG is stopped when the maximum number of iterations 'maxit'
               is reached, or when |A.x-b| <= res*|b|, where the simbol |z|
               represents the Euclidean norm of the vector z.

The function returns the number of iterations used by the CG.
******************************************************************************/
int conjugate_gradient(double complex *x, double complex *b,
                       void (*A)(double complex *, double complex *), int N,
                       double eps, int maxit) {

  // add your code here
}
