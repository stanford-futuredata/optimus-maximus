/* mkl.h is required for dsecnd and SGEMM */
#ifdef MKL_ILP64
#include <mkl.h>
#else
#include <stdlib.h>
#include <cblas.h>
#include <sys/time.h>
#endif
#include <stdio.h>
/* initialization code is skipped for brevity (do a dummy dsecnd() call to improve accuracy of timing) */

void init_matrix(double *A, const long long rows, const long long cols) {
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      A[i*cols + j] = rand() / 1E9;
    }
  }
}

int main() {
  double alpha = 1.0, beta = 1.0;

  const long long m = 20000;
  const long long n = 40000;
  const long long k = 200;

  const int LOOP_COUNT = 100;

  double *A = (double *) malloc(m*k*sizeof(double));
  init_matrix(A, m, k);
  double *B = (double *) malloc(k*n*sizeof(double));
  init_matrix(B, k, n);
  double *C = (double *) malloc(m*n*sizeof(double));
  init_matrix(C, m, n);

  /* first call which does the thread/buffer initialization */
#ifdef MKL_ILP64
  DGEMM("N", "N", &m, &n, &k, &alpha, A, &m, B, &k, &beta, C, &m);
#else
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, alpha,
              A, k, B, k, beta, C, n);
#endif
  /* start timing after the first GEMM call */

#ifdef MKL_ILP64
  double time_st = dsecnd();
#else
  struct timeval tv0;
  gettimeofday(&tv0, 0);
#endif

  for (int i=0; i < LOOP_COUNT; ++i) {
#ifdef MKL_ILP64
    DGEMM("N", "N", &m, &n, &k, &alpha, A, &m, B, &k, &beta, C, &m);
#else
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, alpha,
              A, k, B, k, beta, C, n);
#endif
  }

#ifdef MKL_ILP64
  double time_end = dsecnd();
  double time_avg = (time_end - time_st)/LOOP_COUNT;
#else
  struct timeval tv1;
  gettimeofday(&tv1, 0);
  printf("%f\n", (tv1.tv_usec - tv0.tv_usec) /1E6);
  double time_avg = ((tv1.tv_usec - tv0.tv_usec) / 1E6)/LOOP_COUNT;

#endif
  double gflop = (2.0*m*n*k)*1E-9;
  printf("Average time: %e secs\n", time_avg);
  printf("GFlop       : %.5f\n", gflop);
  printf("GFlop/sec   : %.5f\n", gflop/time_avg);
}
