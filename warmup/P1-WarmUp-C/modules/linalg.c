#include <complex.h>
#include <stdlib.h>

/******************************************************************************
The function 'norm2' returns the squared norm of the N-vector 'v'.

Parameters:
   double complex *v : pointer to a vector with N components
   int N             : number of elements of the vector 'v'

Output: squared norm of the vector 'v'
******************************************************************************/
double norm2(double complex *v, int N) {
  double ret = 0.0;
  for (int i = 0; i < N; i++) {
    ret += creal(v[i]) * creal(v[i]) + cimag(v[i]) * cimag(v[i]);
  }
  return ret;
}

/******************************************************************************
The function 'real_sprod' returns the real part of the scalar product of the
N-vectors 'v' and 'w'.

Parameters:
   double complex *v, *w : pointers to two vector with N components
   int N                 : number of elements of the vectors 'v' and 'w'

Output: real part of scalar product of vectors 'v' and 'w'
******************************************************************************/
double real_sprod(double complex *v, double complex *w, int N) {
  double ret = 0.0;
  for (int i = 0; i < N; i++) {
    ret += creal(v[i]) * creal(w[i]) + cimag(v[i]) * cimag(w[i]);
  }
  return ret;
}
