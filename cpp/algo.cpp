//
//  algo.cpp
//  Simdex
//
//

#include "algo.hpp"
#include "utils.hpp"
#include "parser.hpp"
#include "arith.hpp"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include <functional>
#include <utility>
#include <queue>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <iostream>
#include <fstream>

#include <mkl.h>
#include <ipps.h>

typedef unsigned long long u64;
#define DECLARE_ARGS(val, low, high) unsigned low, high
#define EAX_EDX_VAL(val, low, high) ((low) | ((u64)(high) << 32))
#define EAX_EDX_ARGS(val, low, high) "a"(low), "d"(high)
#define EAX_EDX_RET(val, low, high) "=a"(low), "=d"(high)

#define DATA_TYPE int64_t
static inline unsigned long long _rdtsc3(void) {
  DECLARE_ARGS(val, low, high);

  asm volatile("rdtsc" : EAX_EDX_RET(val, low, high));

  return EAX_EDX_VAL(val, low, high);
}

std::vector<float> linspace(const float start, const float end,
                            const size_t num) {
  float delta = (end - start) / num;

  std::vector<float> linspaced;
  // start from 1; omit start (we only care about upper bounds)
  for (size_t i = 0; i < num; ++i) {
    linspaced.push_back(start + delta * (i + 1));
  }
  return linspaced;
}

/**
 * Find index of smallest theta_b that is greater than theta_uc,
 * so we can find the right list of sorted upper bounds for a given
 * user
 **/
size_t find_theta_bin_index(const float theta_uc,
                            const std::vector<float> theta_bins,
                            const size_t num_bins) {
  for (size_t i = 0; i < num_bins; ++i) {
    if (theta_uc <= theta_bins[i]) {
      return i;
    }
  }
  return num_bins - 1;
}

void computeTopKForCluster(std::vector<size_t> *cluster_index,
                           const size_t clusterID, const size_t user_offset,
                           float *user_weights, float *centroids,
                           float *allItemWeights, float *allItemNorms,
                           size_t num_bins, const size_t K, float *theta_ics,
                           const size_t num_items,
                           const size_t num_latent_factors,
                           std::ofstream &user_stats_file) {

  double time_start, time_end, upperBoundCreation_time, sortUpperBound_time,
      computeTopK_time;

  const std::vector<size_t> user_ids_in_cluster = cluster_index[clusterID];
  const size_t numUsersInCluster = user_ids_in_cluster.size();

  userNormTuple_t *userNormTuple_array = (userNormTuple_t *)_malloc(
      sizeof(struct userNormTuple) * numUsersInCluster);
  // initialize userNormTuple_array with user ids that are assigned to this
  // cluster
  size_t i = 0;
  for (; i < numUsersInCluster; i++) {
    userNormTuple_array[i].userID = user_ids_in_cluster[i];
  }

  time_start = dsecnd();
  time_start = dsecnd();

  // compute theta_ucs for ever user assigned to this cluster,
  // and also initialize userNormTuple_array with the norms of every user
  // NOTE: theta_ucs are now already in the right order, i.e., you can access
  // them sequentially. This is because we reordered the user weights to be
  // in cluster order
  float *theta_ucs = compute_theta_ucs_for_centroid(
      &user_weights[user_offset * num_latent_factors],
      &centroids[clusterID * num_latent_factors], numUsersInCluster,
      num_latent_factors, userNormTuple_array);

  float theta_max = theta_ucs[cblas_isamax(numUsersInCluster, theta_ucs, 1)];
  // if (isnan(theta_max) != 0) {
  //   theta_max = 0;
  //   num_bins = 1;
  //   std::cout << "NaN detected." << std::endl;
  // }

  // ----------Theta Bin Creation Below------------------
  const std::vector<float> theta_bins = linspace(0.F, theta_max, num_bins);
  // theta_bins is correct
  float *const_vector_theta = (float *)_malloc(sizeof(float) * num_items);
  const_vector_theta[0] = 0.0;

  float **upperBounds = (float **)_malloc(sizeof(float *) * num_bins);
  for (i = 0; i < num_bins; i++) {
    upperBounds[i] = (float *)_malloc(num_items * sizeof(float));
  }

  size_t j;
  for (i = 0; i < num_bins; i++) {
    // TODO: inefficient copy value to all items in the array
    for (j = 0; j < num_items; j++) {
      const_vector_theta[j] = theta_bins[i];
    }
    // const_vector_theta = theta_b
    vsSub(num_items, theta_ics, const_vector_theta, const_vector_theta);
    // const_vector_theta = theta_ic - theta_b
    for (size_t l = 0; l < num_items; ++l) {
      // TODO: inefficient
      if (const_vector_theta[l] < 0) {
        const_vector_theta[l] = 0.F;
      }
    }
    vsCos(num_items, const_vector_theta, const_vector_theta);
    // const_vector_theta = cos(theta_ic - theta_b)
    vsMul(num_items, allItemNorms, const_vector_theta, upperBounds[i]);
    // upperBounds[i] = ||i|| * cos(theta_ic - theta_b)
  }

  // upperBounds are correct
  time_end = dsecnd();
  upperBoundCreation_time = (time_end - time_start);

  time_start = dsecnd();
  time_start = dsecnd();

  upperBoundItem_t **sortedUpperBounds =
      (upperBoundItem_t **)_malloc(num_bins * sizeof(upperBoundItem_t *));
  for (i = 0; i < num_bins; i++) {
    sortedUpperBounds[i] =
        (upperBoundItem_t *)_malloc(num_items * sizeof(struct upperBoundItem));
    for (j = 0; j < num_items; j++) {
      sortedUpperBounds[i][j].upperBound = upperBounds[i][j];
      sortedUpperBounds[i][j].itemID = j;
    }
  }
  IppSizeL *pBufSize = (IppSizeL *)malloc(sizeof(IppSizeL));
  ippsSortRadixGetBufferSize_L(num_items, ipp64s, pBufSize);
  Ipp8u *pBuffer = (Ipp8u *)malloc(*pBufSize * sizeof(Ipp8u));
  for (i = 0; i < num_bins; i++) {
    ippsSortRadixDescend_64s_I_L((Ipp64s *)sortedUpperBounds[i], num_items,
                                 pBuffer);
  }
  // sortedUpperBounds are correct

  time_end = dsecnd();
  sortUpperBound_time = (time_end - time_start);

  // ----------Computer Per User TopK Below------------------

  int **allTopK = (int **)_malloc(numUsersInCluster * sizeof(int *));
  for (i = 0; i < numUsersInCluster; i++) {
    allTopK[i] = (int *)malloc((K) * sizeof(int));
  }

  time_start = dsecnd();
  time_start = dsecnd();

  // const size_t num_users_to_compute =
  //     (numUsersInCluster < 10) ? numUsersInCluster : 10;
  // for (i = 0; i < num_users_to_compute; i++) {
  for (i = 0; i < numUsersInCluster; i++) {
    const size_t bin_index =
        find_theta_bin_index(theta_ucs[i], theta_bins, num_bins);

    std::priority_queue<std::pair<float, int>,
                        std::vector<std::pair<float, int> >,
                        std::greater<std::pair<float, int> > > q;

    float score = 0.F;
    size_t itemID = 0;

    for (j = 0; j < K; j++) {
      itemID = sortedUpperBounds[bin_index][j].itemID;
      score = cblas_sdot(
          num_latent_factors, &allItemWeights[itemID * num_latent_factors], 1,
          &user_weights[(user_offset + i) * num_latent_factors], 1);
      q.push(std::make_pair(score, itemID));
    }
    size_t itemsVisited = K;

    for (j = K; j < num_items; j++) {
      if (q.top().first > (userNormTuple_array[i].userNorm *
                           sortedUpperBounds[bin_index][j].upperBound)) {
        break;
      }
      itemID = sortedUpperBounds[bin_index][j].itemID;
      score = cblas_sdot(
          num_latent_factors, &allItemWeights[itemID * num_latent_factors], 1,
          &user_weights[(user_offset + i) * num_latent_factors], 1);
      itemsVisited++;
      if (q.top().first < score) {
        q.pop();
        q.push(std::make_pair(score, itemID));
      }
    }

    for (j = 0; j < K; j++) {
      std::pair<float, int> p = q.top();
      // dont need to store score
      allTopK[i][K - 1 - j] = p.second;  // store itemID
      q.pop();
    }

    user_stats_file << userNormTuple_array[i].userID << "," << clusterID << ","
                    << theta_ucs[i] << "," << itemsVisited << std::endl;
  }

  time_end = dsecnd();
  computeTopK_time = (time_end - time_start);

  // ----------Free Allocated Memory Below-------

  for (i = 0; i < numUsersInCluster; i++) {
    free(allTopK[i]);
  }
  MKL_free(allTopK);

  for (i = 0; i < num_bins; i++) {
    MKL_free(sortedUpperBounds[i]);
    MKL_free(upperBounds[i]);
  }
  MKL_free(upperBounds);
  MKL_free(userNormTuple_array);
  MKL_free(theta_ucs);
  MKL_free(const_vector_theta);
  MKL_free(sortedUpperBounds);
  MKL_Free_Buffers();

  // creationTime += upperBoundCreation_time;
  // sortTime += sortUpperBound_time;
  // computeKTime += computeTopK_time;
}
