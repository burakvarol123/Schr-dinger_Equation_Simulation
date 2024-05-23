#include "linalg.h"
#include "powermethod.h"
#include "utils.h"
#include <complex.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <math.h>

/******************************************************************************
Test function for power method
******************************************************************************/

void test_power_method(gsl_rng *rng, int N, double epsilon) {
  double complex *v = NULL, *Av = NULL;
  double mu;

  set_test_A(rng, N);

  v = malloc(sizeof(double complex) * N);
  Av = malloc(sizeof(double complex) * N);

  int niters;
  niters = power_method(&mu, v, &apply_test_A, N, epsilon, 1000000);

  double res;
  apply_test_A(Av, v);
  for (int i = 0; i < N; i++) {
    Av[i] -= mu * v[i];
  }
  res = sqrt(norm2(Av, N));

  printf("PM\tN= %d\tniters= %d\tmu= %e\teps= %e\tres= %e\tTest: ", N, niters,
         mu, epsilon, res);
  if (res <= epsilon)
    printf("passed\n");
  else
    printf("failed\n");

  free(V);
  free(Av);
}
