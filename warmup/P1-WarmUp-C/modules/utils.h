#include <complex.h>
#include <gsl/gsl_rng.h>

void set_test_A(gsl_rng *rng, int N);
void apply_test_A(double complex *out, double complex *in);
