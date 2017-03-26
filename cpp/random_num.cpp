//
//  random_num.cpp
//  fomo_preproc
//
//  Created by Geet Sethi on 2/27/17.
//  Copyright © 2017 Geet Sethi. All rights reserved.
//

#include "random_num.hpp"
#include <mkl.h>
#include <stdio.h>

/* The function Next_Uniform_Int returns next random integer uniformly
 * distributed on the needed interval */
__inline unsigned int Next_Uniform_Int(
    VSLStreamStatePtr stream, float *D_UNIFORM01_BUF,
    unsigned int *D_UNIFORM01_IDX, int M,
    int N) {/* Return integer uniformly distributed on {0,...,m-1}. */
#define I_RNG_BUF ((unsigned int *)D_UNIFORM01_BUF)
  unsigned int i, j, k;
  int RNGBUFSIZE = M * 255;
  if ((*D_UNIFORM01_IDX) == 0) {
    /* Here if this is the first call to Next_Uniform_Int */
    /* or if D_UNIFORM01_BUF has been completely used */

    /* Generate float-precision uniforms from [0;1) */
    vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, RNGBUFSIZE,
                 D_UNIFORM01_BUF, 0.0, 1.0);
/* Integer scaling phase */
#pragma simd
    for (i = 0; i < RNGBUFSIZE / M; i++)
      for (k = 0; k < M; k++)
        I_RNG_BUF[i * M + k] =
            k + (unsigned int)(D_UNIFORM01_BUF[i * M + k] * (float)(N - k));
  }

  /* Return next integer from buffer */
  j = I_RNG_BUF[*D_UNIFORM01_IDX];
  (*D_UNIFORM01_IDX) = (*D_UNIFORM01_IDX) + 1;

  /* Check if buffer has been completely used */
  if ((*D_UNIFORM01_IDX) == RNGBUFSIZE) (*D_UNIFORM01_IDX) = 0;

  return j;
}

/* Fisher-Yates shuffle */
__inline void Fisher_Yates_shuffle(VSLStreamStatePtr stream,
                                   float *D_UNIFORM01_BUF,
                                   unsigned int *D_UNIFORM01_IDX,
                                   unsigned int *PERMUT_BUF, int M, int N) {
  unsigned int i, j;
  unsigned int tmp;

#pragma novector /* Vectorization is useless here */
  /* A2.2: for i from 0 to M-1 do */
  for (i = 0; i < M; i++) {

    /* A2.3: generate random natural number X from {i,...,N-1} */
    j = Next_Uniform_Int(stream, D_UNIFORM01_BUF, D_UNIFORM01_IDX, M,
                         N); /* Uniform integer from {i,...,N-1} */

    /* A2.4: interchange PERMUT_BUF[i] and PERMUT_BUF[X] */
    tmp = PERMUT_BUF[i];
    PERMUT_BUF[i] = PERMUT_BUF[j];
    PERMUT_BUF[j] = tmp;
  }
}

int *randomArray(int max, int size) {
  int *RESULTS_ARRAY;
  unsigned int i;
  float t;
  int N = max;
  int M = size;
  int ALIGNMENT_BYTES = 64;
  int RNGBUFSIZE = M * 255;
  int EXPERIM_NUM = 1;
  int sample_num;

  /* Allocate memory for results buffer */
  if (!(RESULTS_ARRAY = (int *)mkl_malloc(sizeof(int) * size, 64))) {
    printf("Error 1: Memory allocation failed!\n");
    exit(1);
  }

  /* Generation of all lottery results, with time measuring (without warm-up) */
  t = dsecnd(); /* Extra call to dsecnd for its initialization. */
  t = dsecnd(); /* Get elapsed time in seconds. */

  unsigned int seed;
  VSLStreamStatePtr stream; /* Each thread has its own RNG stream */
  unsigned int *PERMUT_BUF; /* Each thread has its own buffer of length N to
                               keep partial permutations */
  float *D_UNIFORM01_BUF; /* Each thread has its own buffer of intermediate
                             uniforms */
  unsigned int D_UNIFORM01_IDX = 0; /* Index of current uniform in the buffer
                                       (each thread has its own index) */

  seed = 777;

  /* Allocate memory for population buffer */
  if (!(PERMUT_BUF = (unsigned int *)mkl_malloc(sizeof(unsigned int) * N,
                                                ALIGNMENT_BYTES))) {
    printf("Error 2: Memory allocation failed!\n");
  } else {

    /* Allocate memory for intermediate uniforms */
    if (!(D_UNIFORM01_BUF = (float *)mkl_malloc(sizeof(float) * RNGBUFSIZE,
                                                ALIGNMENT_BYTES))) {
      printf("Error 3: Memory allocation failed!\n");

    } else {

      /* RNG stream initialization in this thread, */
      /* each thread will produce RNG sequence independent of other threads
       * sequencies, */
      /* inspite of same seed used */
      if ((vslNewStream(&stream, VSL_BRNG_MT2203, seed)) != VSL_STATUS_OK) {
        printf("Error 4: Stream initialization failed!\n");
      } else {

/*  A2.1: (Initialization step) let PERMUT_BUF contain natural numbers 1, 2,
 * ..., N */
#pragma simd
        for (i = 0; i < N; i++)
          PERMUT_BUF[i] = i + 1; /* we will use the set {1,...,N} */

/* Generation of experiment samples (in thread number thr). */
/*  The RESULTS_ARRAY is broken down into THREADS_NUM portions of size
 * ONE_THR_PORTION_SIZE, */
/*  and each thread generates results for its portion of RESULTS_ARRAY. */
#pragma simd /* No sence in vectorization */
        for (sample_num = 0; sample_num < EXPERIM_NUM; sample_num++) {

          /* Generate next lottery sample (steps A2.2, A2.3, A2.4): */
          Fisher_Yates_shuffle(stream, D_UNIFORM01_BUF, &D_UNIFORM01_IDX,
                               PERMUT_BUF, M, N);

/* A2.5: (Copy stage) for i from 0 to M-1 do RESULTS_ARRAY[i]=PERMUT_BUF[i] */
#pragma simd
          for (i = 0; i < M; i++)
            RESULTS_ARRAY[sample_num * M + i] = PERMUT_BUF[i];
        }

        vslDeleteStream(&stream);
      }

      mkl_free(D_UNIFORM01_BUF);
    }

    mkl_free(PERMUT_BUF);
  }
  t = dsecnd() - t; /* Time spent, measured in seconds. */
  printf("Performance: %.2f ms\n", t * 1e3); /* Convert to milliseconds. */

  //    /* Print 3 last lottery samples */
  //    #pragma simd /* No need for vectorization when printing results
  // sequentially */
  //    for( sample_num=0; sample_num<EXPERIM_NUM; sample_num++ ) {
  //        /* Print current generated sample */
  //        printf("Sample %2d of lottery %d of %d: ", (int)sample_num, M, N);
  //        #pragma simd /* No need for vectorization while printing */
  //        for( i=4563; i<4583; i++ ) printf("%2d, ",
  // (int)RESULTS_ARRAY[sample_num*M+i]);
  //        printf("\n");
  //    }

  //    mkl_free(RESULTS_ARRAY);
  return RESULTS_ARRAY;
}
