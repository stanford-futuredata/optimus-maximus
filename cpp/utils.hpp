//
//  utils.hpp
//
//

#ifndef utils_hpp
#define utils_hpp

#include <chrono>
#include <stdio.h>
#include <mkl.h>

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::milliseconds ms;
typedef std::chrono::duration<float> fsec;

namespace chrono = std::chrono;
typedef chrono::time_point<chrono::system_clock> sys_time;
typedef unsigned long long u64;

inline void _free(void *ptr) {
#ifdef ICC
  mkl_free(ptr);
#else
  free(ptr);
#endif
}

inline const void *_malloc(const size_t size) {
  void *ptr = NULL;
#ifdef ICC
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
