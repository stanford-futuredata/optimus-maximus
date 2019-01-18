//
//  blocked_mm.hpp
//
//

#ifndef blocked_mm_hpp
#define blocked_mm_hpp

#include <iostream>
#include <queue>
#ifdef MKL_ILP64
#include <mkl.h>
#else
#include <cblas.h>
#include <omp.h>
#endif

inline void computeTopRating(double *ratings_matrix, int *top_K_items,
                             const int num_users, const int num_items) {
  #pragma omp parallel
  #pragma omp for
  for (int user_id = 0; user_id < num_users; user_id++) {
    unsigned long index = user_id;
    index *= num_items;
    int best_item_id = cblas_idamax(num_items, &ratings_matrix[index], 1);
    top_K_items[user_id] = best_item_id;
  }
}

inline void computeTopK(double *ratings_matrix, int *top_K_items,
                        const int num_users, const int num_items, const int K) {

  #pragma omp parallel
  #pragma omp for
  for (int i = 0; i < num_users; i++) {

    // TODO: allocate vector on the stack, reserve the size we need or use the
    // insertion-and-copy array strategy that Matei suggested
    std::priority_queue<std::pair<double, int>,
                        std::vector<std::pair<double, int> >,
                        std::greater<std::pair<double, int> > >
        q;

    unsigned long index = i;
    index *= num_items;

    for (int j = 0; j < K; j++) {
      q.push(std::make_pair(ratings_matrix[index + j], j));
    }

    for (int j = K; j < num_items; j++) {
      if (ratings_matrix[index + j] > q.top().first) {
        q.pop();
        q.push(std::make_pair(ratings_matrix[index + j], j));
      }
    }

    for (int j = 0; j < K; j++) {
      const std::pair<double, int> p = q.top();
      top_K_items[i * K + K - 1 - j] = p.second;
      q.pop();
    }
  }
}

#endif
