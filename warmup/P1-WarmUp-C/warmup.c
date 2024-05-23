#include "conjgrad.h"
#include <complex.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/******************************************************************************
Main function
******************************************************************************/

int main(int argc, char *argv[]) {
  /* Initialize GSL random number generator  */
  gsl_rng *rng = NULL;
  gsl_rng_env_setup();
  rng = gsl_rng_alloc(gsl_rng_ranlux);
  gsl_rng_set(rng, 12345);

  test_power_method(rng, 5, 1e-8);
  test_power_method(rng, 10, 1e-4);
  test_power_method(rng, 18, 1e-12);
  test_power_method(rng, 17, 1e-10);
  test_power_method(rng, 30, 1e-8);

  test_conjugate_gradient(rng, 5, 1e-8);
  test_conjugate_gradient(rng, 10, 1e-4);
  test_conjugate_gradient(rng, 18, 1e-12);
  test_conjugate_gradient(rng, 17, 1e-10);
  test_conjugate_gradient(rng, 30, 1e-8);

  return 0;
}
