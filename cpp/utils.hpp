//
//  utils.hpp
//
//

#ifndef utils_hpp
#define utils_hpp

#include <boost/timer/timer.hpp>
#include <chrono>

#ifdef ICC
#include <mkl.h>
#endif

namespace timer = boost::timer;
namespace chrono = std::chrono;
typedef chrono::time_point<chrono::system_clock> sys_time;
typedef unsigned long long u64;

#define DECLARE_ARGS(val, low, high) unsigned low, high
#define EAX_EDX_VAL(val, low, high) ((low) | ((u64)(high) << 32))
#define EAX_EDX_ARGS(val, low, high) "a"(low), "d"(high)
#define EAX_EDX_RET(val, low, high) "=a"(low), "=d"(high)
static inline u64 _rdtsc2(void) {
  DECLARE_ARGS(val, low, high);
  asm volatile("rdtsc" : EAX_EDX_RET(val, low, high));
  return EAX_EDX_VAL(val, low, high);
}

template <typename F>
double time_function(F f) {
  const sys_time start = chrono::system_clock::now();
  f();
  const sys_time end = chrono::system_clock::now();
  const double time = (end - start).count();
  return time;
}

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
