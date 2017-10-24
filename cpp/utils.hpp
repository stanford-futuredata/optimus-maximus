//
//  utils.hpp
//
//

#ifndef utils_hpp
#define utils_hpp

#include <stdio.h>
#include <sys/time.h>
#ifdef MKL_ILP64
#include <mkl.h>
#else
#include <stdlib.h>
#endif

typedef struct timeval bench_timer_t;

/** Starts the clock for a benchmark. */
static inline bench_timer_t time_start() {
  bench_timer_t t;
  gettimeofday(&t, NULL);
  return t;
}

/**
 * Stops the clock and returns time elapsed in seconds.
 * Throws an error if time_start() was not called first.
 ***/
static inline double time_stop(bench_timer_t start) {
  bench_timer_t end;
  bench_timer_t diff;
  gettimeofday(&end, NULL);
  timersub(&end, &start, &diff);
  return (double)diff.tv_sec + ((double)diff.tv_usec / 1000000.0);
}

inline void _free(void *ptr) {
#ifdef MKL_ILP64
  mkl_free(ptr);
#else
  free(ptr);
#endif
}

inline const void *_malloc(const size_t size) {
  void *ptr = NULL;
#ifdef MKL_ILP64
  ptr = mkl_malloc(size, 64);
#else
  ptr = malloc(size);
#endif
  if (ptr == NULL) {
    printf(
        "\n ERROR: Can't allocate memory. "
        "Aborting... \n\n");
    exit(1);
  }
  return ptr;
}

#endif /* utils_hpp */
