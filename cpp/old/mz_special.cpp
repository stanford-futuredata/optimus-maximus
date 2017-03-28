//
//  mz_special.cpp
//  fomo_preproc
//
//  Created by Geet Sethi on 10/27/16.
//  Copyright © 2016 Geet Sethi. All rights reserved.
//

#include "mz_special.hpp"
#include "arith.hpp"
#include "parser.hpp"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <mkl.h>
#include <functional>
#include <utility>
#include <queue>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <iostream>
#include <fstream>
#include <math.h>
#include <ipps.h>

extern int numberUsers;
extern int numFeatures;
extern int numItems;

extern double creationTime;
extern double sortTime;
extern double computeKTime;

// TEST PERFORMANCE OF VS VS VMS MKL

typedef unsigned long long u64;
#define DECLARE_ARGS(val, low, high) unsigned low, high
#define EAX_EDX_VAL(val, low, high) ((low) | ((u64)(high) << 32))
#define EAX_EDX_ARGS(val, low, high) "a"(low), "d"(high)
#define EAX_EDX_RET(val, low, high) "=a"(low), "=d"(high)

#define DATA_TYPE int64_t
struct negrightshift {
  inline int64_t operator()(const int64_t &x, const unsigned offset) {
    return -(x >> offset);
  }
};

static inline unsigned long long _rdtsc3(void) {
  DECLARE_ARGS(val, low, high);

  asm volatile("rdtsc" : EAX_EDX_RET(val, low, high));

  return EAX_EDX_VAL(val, low, high);
}

int *indexPermutation(float *upperBounds) {
  int *indexes = (int *)malloc(sizeof(int) * numItems);

  std::iota(indexes, indexes + numItems, 0);
  std::sort(indexes, indexes + numItems, [&](size_t i1, size_t i2) {
    return upperBounds[i1] > upperBounds[i2];
  });

  return indexes;
}

float *applyPermutation(float *upperBounds, int *permutation) {
  float *sortedUpperBounds = (float *)malloc(sizeof(float) * numItems);
  std::transform(permutation, permutation + numItems, sortedUpperBounds,
                 [&](size_t i) { return upperBounds[i]; });
  return sortedUpperBounds;
}

bool compareByBound(const upperBoundItem_t &a, const upperBoundItem_t &b) {
  return a.upperBound > b.upperBound;
}

void computeTopKForCluster(fullCluster_t *fullClusterData, int clusterID,
                           float *fullUserData, float *clusterCentroids,
                           float *allItemWeights, float *allItemNorms,
                           int numBins, int k, float *vectorized_Acos_output_ic,
                           std::ofstream &user_stats_file) {
  int i = 0;
  int j = 0;

  double time_start, time_end, upperBoundCreation_time, sortUpperBound_time,
      computeTopK_time;

  int clusterOffset = fullClusterData->clusterOffset[clusterID];
  if (fullClusterData->clusterArray[clusterOffset] != clusterID) {
    printf(
        "In Cluster Prep: Cluster Array Cluster ID NOT EQUAL to input Cluster "
        "ID. Exiting.");
    exit(1);
  }

  int numUsersInCluster = fullClusterData->clusterArray[clusterOffset + 1];

  userNormTuple_t *userNormTuple_array = (userNormTuple_t *)mkl_malloc(
      sizeof(struct userNormTuple) * numUsersInCluster, 64);
  for (i = 0; i < numUsersInCluster; i++) {
    userNormTuple_array[i].userID =
        fullClusterData->clusterArray[clusterOffset + 2 + i];
  }

  time_start = dsecnd();
  time_start = dsecnd();

  float *vectorized_Acos_output_uc =
      (float *)mkl_malloc(sizeof(float) * numUsersInCluster, 64);

  float *vectorized_Acos_input_uc = computeCosineSimilarityUserCluster(
      &fullUserData[userNormTuple_array[0].userID * numFeatures],
      &clusterCentroids[clusterID * numFeatures], numUsersInCluster,
      userNormTuple_array);

  vsAcos(numUsersInCluster, vectorized_Acos_input_uc,
         vectorized_Acos_output_uc);
  // all acos -- ie theta_uc -- stored in output_array
  float theta_max = vectorized_Acos_output_uc
      [cblas_isamax(numUsersInCluster, vectorized_Acos_output_uc, 1)];
  if (isnan(theta_max) != 0) {
    theta_max = 0;
    numBins = 1;
    printf("NaN detected.\n");
  }

  // ----------Theta Bin Creation Below------------------

  // use vectorized sort
  float *theta_bins = (float *)malloc(sizeof(float) * numBins);
  float theta_bin_step = theta_max / (float)numBins;
  for (i = 1; i < (numBins + 1); i++) {
    theta_bins[i - 1] = theta_bin_step * i;
  }

  float *const_vector_theta =
      (float *)mkl_malloc(sizeof(float) * numItems, 64);
  const_vector_theta[0] = 0.0;

  float **upperBounds = (float **)mkl_malloc(sizeof(float *) * numBins, 64);
  for (i = 0; i < numBins; i++) {
    upperBounds[i] = (float *)mkl_malloc(numItems * sizeof(float), 64);
  }

  for (i = 0; i < numBins; i++) {
    upperBounds[i][0] = 0.0;
  }

  for (i = 0; i < numBins; i++) {
    for (j = 0; j < numItems; j++) {
      const_vector_theta[j] = theta_bins[i];
    }
    vsSub(numItems, vectorized_Acos_output_ic, const_vector_theta,
          const_vector_theta);
    vsCos(numItems, const_vector_theta, const_vector_theta);
    vsMul(numItems, allItemNorms, const_vector_theta, upperBounds[i]);
  }

  time_end = dsecnd();
  upperBoundCreation_time = (time_end - time_start);

  time_start = dsecnd();
  time_start = dsecnd();

  upperBoundItem_t **sortedUpperBounds =
      (upperBoundItem_t **)mkl_malloc(numBins * sizeof(upperBoundItem_t *), 64);
  for (i = 0; i < numBins; i++) {
    sortedUpperBounds[i] = (upperBoundItem_t *)mkl_malloc(
        numItems * sizeof(struct upperBoundItem), 64);
    for (j = 0; j < numItems; j++) {
      sortedUpperBounds[i][j].upperBound = upperBounds[i][j];
      sortedUpperBounds[i][j].itemID = j;
    }
  }
  IppSizeL *pBufSize = (IppSizeL *)malloc(sizeof(IppSizeL));
  ippsSortRadixGetBufferSize_L(numItems, ipp64s, pBufSize);
  Ipp8u *pBuffer = (Ipp8u *)malloc(*pBufSize * sizeof(Ipp8u));
  for (i = 0; i < numBins; i++) {
    ippsSortRadixDescend_64s_I_L((Ipp64s *)sortedUpperBounds[i], numItems,
                                 pBuffer);
  }

  time_end = dsecnd();
  sortUpperBound_time = (time_end - time_start);

  // ----------Computer Per User TopK Below------------------

  int currentUser = 0;
  int bucket_index = 0;  // will crash if not assigned later

  int **allTopK = (int **)mkl_malloc(numUsersInCluster * sizeof(int *), 64);
  for (i = 0; i < numUsersInCluster; i++) {
    allTopK[i] = (int *)malloc((k) * sizeof(int));
  }

  time_start = dsecnd();
  time_start = dsecnd();

  for (i = 0; i < numUsersInCluster; i++) {
    currentUser = userNormTuple_array[i].userID;
    // find user's bin
    for (j = 0; j < numBins; j++) {
      if (vectorized_Acos_output_uc[i] <= theta_bins[j]) {
        bucket_index = j;
        break;
      }
    }
    // found bin

    std::priority_queue<std::pair<float, int>,
                        std::vector<std::pair<float, int> >,
                        std::greater<std::pair<float, int> > > q;

    float score = 0.0;
    int itemID = 0;

    for (j = 0; j < k; j++) { 
      itemID = sortedUpperBounds[bucket_index][j].itemID;
      score = cblas_sdot(numFeatures, &allItemWeights[itemID * numFeatures], 1,
                         &fullUserData[currentUser * numFeatures], 1);
      q.push(std::make_pair(score, itemID));
    }
    int itemsVisited = k;

    for (j = k; j < numItems; j++) {
      if (q.top().first > (userNormTuple_array[i].userNorm *
                           sortedUpperBounds[bucket_index][j].upperBound)) {
        break;
      }
      itemID = sortedUpperBounds[bucket_index][j].itemID;
      score = cblas_sdot(numFeatures, &allItemWeights[itemID * numFeatures], 1,
                         &fullUserData[currentUser * numFeatures], 1);
      itemsVisited++;
      if (q.top().first < score) {
        q.pop();
        q.push(std::make_pair(score, itemID));
      }
    }

    for (j = 0; j < k; j++) {
      std::pair<float, int> p = q.top();
      // dont need to store score
      allTopK[i][k - 1 - j] = p.second;  // store itemID
      q.pop();
    }

    user_stats_file << k << "," << clusterID << "," << currentUser << ","
                    << itemsVisited << "," << vectorized_Acos_output_uc[i]
                    << std::endl;
  }

  time_end = dsecnd();
  computeTopK_time = (time_end - time_start);

  // ----------Free Allocated Memory Below-------

  for (i = 0; i < numUsersInCluster; i++) {
    free(allTopK[i]);
  }
  MKL_free(allTopK);

  for (i = 0; i < numBins; i++) {
    MKL_free(sortedUpperBounds[i]);
    MKL_free(upperBounds[i]);
  }
  MKL_free(upperBounds);
  MKL_free(userNormTuple_array);
  MKL_free(vectorized_Acos_input_uc);
  MKL_free(vectorized_Acos_output_uc);
  free(theta_bins);
  MKL_free(const_vector_theta);
  MKL_free(sortedUpperBounds);
  MKL_Free_Buffers();

  creationTime += upperBoundCreation_time;
  sortTime += sortUpperBound_time;
  computeKTime += computeTopK_time;
}
