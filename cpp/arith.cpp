//
//  arith.cpp
//  Simdex
//
//  Created by Geet Sethi on 10/27/16.
//  Copyright © 2016 Geet Sethi. All rights reserved.
//

#include "arith.hpp"
#include "algo.hpp"
#include "parser.hpp"
#include "utils.hpp"

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <mkl.h>

extern int numberUsers;
extern int numFeatures;
extern int numItems;
extern int numClusters;

float *computeAllItemNorms(float *allItemWeights) {
  float *allItemNorms = (float *)malloc(sizeof(float) * numItems);

  for (int i = 0; i < numItems; i++) {
    allItemNorms[i] =
        cblas_snrm2(numFeatures, &allItemWeights[i * numFeatures], 1);
  }

  return allItemNorms;
}

float *computeAllItemNormsForMM(float *allItemWeights) {
  float *allItemNorms =
      (float *)_malloc(sizeof(float) * numItems * numFeatures);
  for (int i = 0; i < numItems; i++) {
    allItemNorms[i * numFeatures] =
        cblas_snrm2(numFeatures, &allItemWeights[i * numFeatures], 1);
    for (int j = 0; j < numFeatures; j++) {
      allItemNorms[(i * numFeatures) + j] = allItemNorms[i * numFeatures];
    }
  }

  return allItemNorms;
}

float *computeAllCentroidNorms(float *clusterCentroids) {
  float *allCentroidNorms =
      (float *)_malloc(sizeof(float) * (numClusters * numFeatures));
  for (int i = 0; i < numClusters; i++) {
    allCentroidNorms[i * numFeatures] =
        cblas_snrm2(numFeatures, &clusterCentroids[i * numFeatures], 1);
    for (int j = 0; j < numFeatures; j++) {
      allCentroidNorms[(i * numFeatures) + j] =
          allCentroidNorms[i * numFeatures];
    }
  }

  return allCentroidNorms;
}

float *computeCosineSimilarityItemsCentroids(float *allItems,
                                             float *allCentroids) {

  int m, n, k, i;
  m = numClusters;
  k = numFeatures;
  n = numItems;
  float alpha = 1.0;
  float beta = 0.0;

  float *itemDOTcentroid =
      (float *)_malloc(sizeof(float) * numClusters * numItems);
  float *allItemsNew = (float *)_malloc(sizeof(float) * numItems * numFeatures);
  float *allCentroidsNew =
      (float *)_malloc(sizeof(float) * numClusters * numFeatures);

  float *allItemNorms = computeAllItemNormsForMM(allItems);
  float *allCentroidNorms = computeAllCentroidNorms(allCentroids);

  vsInv(numClusters * numFeatures, allCentroidNorms, allCentroidNorms);
  vsInv(numItems * numFeatures, allItemNorms, allItemNorms);

  for (i = 0; i < numItems; i++) {
    vsMul(numFeatures, &allItems[i * numFeatures],
          &allItemNorms[i * numFeatures], &allItemsNew[i * numFeatures]);
  }
  for (i = 0; i < numClusters; i++) {
    vsMul(numFeatures, &allCentroids[i * numFeatures],
          &allCentroidNorms[i * numFeatures],
          &allCentroidsNew[i * numFeatures]);
  }

  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, alpha,
              allCentroidsNew, k, allItemsNew, k, beta, itemDOTcentroid, n);

  MKL_free(allCentroidsNew);
  MKL_free(allItemsNew);
  MKL_free(allItemNorms);
  MKL_free(allCentroidNorms);

  return itemDOTcentroid;
}

float *computeClusterUserNorms(float *allUserWeights, int numUsers,
                               userNormTuple_t *userNormTuple_array) {
  float *allUserNorms =
      (float *)_malloc(sizeof(float) * (numUsers * numFeatures));

  for (int i = 0; i < numUsers; i++) {
    allUserNorms[i * numFeatures] =
        cblas_snrm2(numFeatures, &allUserWeights[i * numFeatures], 1);
    userNormTuple_array[i].userNorm = allUserNorms[i * numFeatures];
    for (int j = 0; j < numFeatures; j++) {
      allUserNorms[(i * numFeatures) + j] = allUserNorms[i * numFeatures];
    }
  }

  return allUserNorms;
}

float *computeCosineSimilarityUserCluster(
    float *users, float *centroid, int numUsers,
    userNormTuple_t *userNormTuple_array) {
  int m, k, i;
  m = numUsers;
  k = numFeatures;

  float alpha = 1.0;
  float beta = 0.0;
  int stride = 1;

  float *usersDOTcentroid = (float *)_malloc(sizeof(float) * numUsers);
  float *centroidNorm = (float *)_malloc(sizeof(float) * numFeatures);

  centroidNorm[0] = cblas_snrm2(numFeatures, centroid, 1);
  for (i = 1; i < numFeatures; i++) {
    centroidNorm[i] = centroidNorm[0];
  }

  float *userNorms =
      computeClusterUserNorms(users, numUsers, userNormTuple_array);

  vsInv(numFeatures, centroidNorm, centroidNorm);
  vsInv((numUsers) * numFeatures, userNorms, userNorms);

  vsMul(numFeatures, centroid, centroidNorm, centroidNorm);
  for (i = 0; i < numUsers; i++) {
    vsMul(numFeatures, &users[i * numFeatures], &userNorms[i * numFeatures],
          &userNorms[i * numFeatures]);
  }

  cblas_sgemv(CblasRowMajor, CblasNoTrans, m, k, alpha, userNorms, k,
              centroidNorm, stride, beta, usersDOTcentroid, stride);

  MKL_free(userNorms);
  MKL_free(centroidNorm);

  return usersDOTcentroid;
}

/**
 * Replace all NaNs in the array with zeroes
 **/
inline void remove_nans(float *arr, size_t num_elems) {
  // TODO: more efficient
  for (size_t i = 0; i < num_elems; ++i) {
    if (isnan(arr[i])) {
      arr[i] = 0.F;
    }
  }
}

// TODO: Combine compute_theta_ics and compute_theta_ucs into a single function

/**
 * Compute theta_ics: the angle between every centroid and every item in the
 * dataset
 **/
float *compute_theta_ics(float *all_items, float *all_centroids,
                         const size_t num_items, const size_t num_clusters,
                         const size_t num_latent_factors) {
  const int m = num_clusters;
  const int k = num_latent_factors;
  const int n = num_items;
  const float alpha = 1.0;
  const float beta = 0.0;

  float *itemDOTcentroid =
      (float *)_malloc(sizeof(float) * num_clusters * num_items);
  float *allItemsNew =
      (float *)_malloc(sizeof(float) * num_items * num_latent_factors);
  float *allCentroidsNew =
      (float *)_malloc(sizeof(float) * num_clusters * num_latent_factors);

  float *allItemNorms = computeAllItemNormsForMM(all_items);
  float *allCentroidNorms = computeAllCentroidNorms(all_centroids);

  vsInv(num_clusters * num_latent_factors, allCentroidNorms, allCentroidNorms);
  vsInv(num_items * num_latent_factors, allItemNorms, allItemNorms);

  size_t i = 0;
  for (; i < num_items; i++) {
    vsMul(num_latent_factors, &all_items[i * num_latent_factors],
          &allItemNorms[i * num_latent_factors],
          &allItemsNew[i * num_latent_factors]);
  }
  for (i = 0; i < num_clusters; i++) {
    vsMul(num_latent_factors, &all_centroids[i * num_latent_factors],
          &allCentroidNorms[i * num_latent_factors],
          &allCentroidsNew[i * num_latent_factors]);
  }

  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, alpha,
              allCentroidsNew, k, allItemsNew, k, beta, itemDOTcentroid, n);

  float *theta_ics = (float *)_malloc(sizeof(float) * num_clusters * num_items);
  vsAcos(num_clusters * num_items, itemDOTcentroid, theta_ics);
  remove_nans(theta_ics, num_clusters * num_items);

  MKL_free(allCentroidsNew);
  MKL_free(allItemsNew);
  MKL_free(allItemNorms);
  MKL_free(allCentroidNorms);
  MKL_free(itemDOTcentroid);

  return theta_ics;
}

/**
 * Compute theta_ucs: angle between _single_ centroid and every user in the
 * dataset
 **/
float *compute_theta_ucs_for_centroid(float *user_weights, float *centroid,
                                      const size_t num_users,
                                      const size_t num_latent_factors,
                                      userNormTuple_t *userNormTuple_array) {
  const size_t m = num_users;
  const size_t k = num_latent_factors;

  const float alpha = 1.0;
  const float beta = 0.0;
  const int stride = 1;

  float *usersDOTcentroid = (float *)_malloc(sizeof(float) * num_users);
  float *centroidNorm = (float *)_malloc(sizeof(float) * num_latent_factors);

  // TODO: inefficient
  centroidNorm[0] = cblas_snrm2(num_latent_factors, centroid, 1);
  size_t i = 1;
  for (i = 1; i < num_latent_factors; i++) {
    centroidNorm[i] = centroidNorm[0];
  }

  float *userNorms =
      computeClusterUserNorms(user_weights, num_users, userNormTuple_array);

  vsInv(num_latent_factors, centroidNorm, centroidNorm);
  vsInv((num_users) * num_latent_factors, userNorms, userNorms);

  vsMul(num_latent_factors, centroid, centroidNorm, centroidNorm);
  for (i = 0; i < num_users; i++) {
    vsMul(num_latent_factors, &user_weights[i * num_latent_factors],
          &userNorms[i * num_latent_factors],
          &userNorms[i * num_latent_factors]);
  }

  cblas_sgemv(CblasRowMajor, CblasNoTrans, m, k, alpha, userNorms, k,
              centroidNorm, stride, beta, usersDOTcentroid, stride);

  // now compute theta_uc's by taking arccosine
  float *theta_ucs = (float *)_malloc(sizeof(float) * num_users);
  vsAcos(num_users, usersDOTcentroid, theta_ucs);
  remove_nans(theta_ucs, num_users);
  MKL_free(userNorms);
  MKL_free(centroidNorm);
  MKL_free(usersDOTcentroid);
  return theta_ucs;
}
