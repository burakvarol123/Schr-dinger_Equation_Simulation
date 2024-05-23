#include "conjgrad.h"
#include "linalg.h"
#include "utils.h"
#include <complex.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <math.h>

/******************************************************************************
Test function for conjugate gradient
******************************************************************************/

void test_conjugate_gradient(gsl_rng *rng, int N, double epsilon) {
  double complex *x = NULL, *b = NULL, *Ax = NULL;

  set_test_A(rng, N);

  x = malloc(sizeof(double complex) * N);
  b = malloc(sizeof(double complex) * N);
  Ax = malloc(sizeof(double complex) * N);

  for (int i = 0; i < N; i++) {
    b[i] = CMPLX(gsl_ran_gaussian(rng, 1.0), gsl_ran_gaussian(rng, 1.0));
  }

  int niters;
  niters = conjugate_gradient(x, b, &apply_test_A, N, epsilon, 1000000);

  double res;
  apply_test_A(Ax, x);
  for (int i = 0; i < N; i++) {
    Ax[i] -= b[i];
  }
  res = sqrt(norm2(Ax, N) / norm2(b, N));

  printf("CG\tN= %d\tniters= %d\teps= %e\tres= %e\tTest: ", N, niters, epsilon,
         res);
  if (res <= epsilon)
    printf("passed\n");
  else
    printf("failed\n");

  free(x);
  free(Ax);
  free(b);
}
