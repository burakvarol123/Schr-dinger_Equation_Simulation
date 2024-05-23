#include <complex.h>
#include <gsl/gsl_rng.h>

int conjugate_gradient(double complex *x, double complex *b,
                       void (*A)(double complex *, double complex *), int N,
                       double eps, int maxit);

void test_conjugate_gradient(gsl_rng *rng, int N, double epsilon);
