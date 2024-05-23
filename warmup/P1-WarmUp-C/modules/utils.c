#include <complex.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <stdlib.h>

/******************************************************************************
The function 'generate_positive_definite_matrix' contructs an NxN
positive-definite matrix 'A'.

Parameters:
   double complex *A : pointer to an allocated array with size N*N
                       on return, it contains the generated flattened matrix
   int N             : number of rows and cols of matrix 'A'
******************************************************************************/
static void generate_positive_definite_matrix(gsl_rng *rng, double complex *A,
                                              int N) {
  double complex *B = NULL;
  B = malloc(sizeof(double complex) * N * N);

  for (int i = 0; i < N * N; i++) {
    B[i] = CMPLX(gsl_ran_gaussian(rng, 1.0), gsl_ran_gaussian(rng, 1.0));
  }

#define MAT(C, k, l) C[(k)*N + (l)]
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      MAT(A, i, j) = 0.0;
      for (int k = 0; k < N; k++) {
        MAT(A, i, j) += conj(MAT(B, k, i)) * MAT(B, k, j);
      }
    }
  }
#undef MAT

  free(B);
}

double complex *_A_ = NULL;
int _N_ = 0;

void set_test_A(gsl_rng *rng, int N) {
  if (_A_ != NULL)
    free(_A_);
  _N_ = N;
  _A_ = malloc(sizeof(double complex) * _N_ * _N_);
  generate_positive_definite_matrix(rng, _A_, _N_);
}

void apply_test_A(double complex *out, double complex *in) {
  for (int i = 0; i < _N_; i++) {
    out[i] = 0.0;
    for (int j = 0; j < _N_; j++) {
      out[i] += _A_[i * _N_ + j] * in[j];
    }
  }
}
