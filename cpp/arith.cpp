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

float *computeAllItemNorms(float *item_weights, const size_t num_items,
                           const size_t num_latent_factors) {
  float *allItemNorms = (float *)malloc(sizeof(float) * num_items);

  for (size_t i = 0; i < num_items; i++) {
    allItemNorms[i] = cblas_snrm2(num_latent_factors,
                                  &item_weights[i * num_latent_factors], 1);
  }

  return allItemNorms;
}

float *computeClusterUserNorms(float *user_weights, const size_t num_users,
                               const size_t num_latent_factors,
                               userNormTuple_t *userNormTuple_array) {
  float *user_norms =
      (float *)_malloc(sizeof(float) * (num_users * num_latent_factors));

  for (size_t i = 0; i < num_users; i++) {
    user_norms[i * num_latent_factors] = cblas_snrm2(
        num_latent_factors, &user_weights[i * num_latent_factors], 1);
    userNormTuple_array[i].userNorm = user_norms[i * num_latent_factors];
    for (size_t j = 0; j < num_latent_factors; j++) {
      user_norms[(i * num_latent_factors) + j] =
          user_norms[i * num_latent_factors];
    }
  }

  return user_norms;
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

/**
 * Compute L2 norm per row for a given matrix. This preserves the dimensions
 * of the input matrix_weights which is num_rows x num_cols;
 * output matrix_norms[i, :] = norm(matrix_weights[i, :]) _for the entire row_
 * (Entire row contains the same value in matrix_norms.)
 **/
float *compute_norms_matrix(float *matrix_weights, const size_t num_rows,
                            const size_t num_cols) {
  float *matrix_norms = (float *)_malloc(sizeof(float) * num_rows * num_cols);
  for (size_t i = 0; i < num_rows; i++) {
    matrix_norms[i * num_cols] =
        cblas_snrm2(num_cols, &matrix_weights[i * num_cols], 1);
    for (size_t j = 1; j < num_cols; j++) {
      matrix_norms[(i * num_cols) + j] = matrix_norms[i * num_cols];
    }
  }

  return matrix_norms;
}

/**
 * Compute theta_ics: the angle between every centroid and every item in the
 * dataset
 **/
float *compute_theta_ics(float *item_weights, float *centroids,
                         const size_t num_items,
                         const size_t num_latent_factors,
                         const size_t num_clusters) {
  const int m = num_clusters;
  const int k = num_latent_factors;
  const int n = num_items;
  const float alpha = 1.0;
  const float beta = 0.0;

  float *cos_theta_ics =
      (float *)_malloc(sizeof(float) * num_clusters * num_items);
  float *normalized_item_weights =
      (float *)_malloc(sizeof(float) * num_items * num_latent_factors);
  float *normalized_centroids =
      (float *)_malloc(sizeof(float) * num_clusters * num_latent_factors);

  // item_norms_matrix: a num_items x num_latent_factors matrix.
  // item_norms_matrix[i, :] = ||i|| across entire row
  float *item_norms_matrix =
      compute_norms_matrix(item_weights, num_items, num_latent_factors);
  // centroid_norms_matrix: a num_clusters x num_latent_factors matrix.
  // centroid_norms_matrix[c, 0] = ||c|| across entire row
  float *centroid_norms_matrix =
      compute_norms_matrix(centroids, num_clusters, num_latent_factors);
  // centroid_norms_matrix now has 1/||c||

  vsInv(num_clusters * num_latent_factors, centroid_norms_matrix,
        centroid_norms_matrix);
  vsInv(num_items * num_latent_factors, item_norms_matrix, item_norms_matrix);
  // item_norms_matrix now has 1/||i||

  size_t i = 0;
  for (; i < num_items; i++) {
    vsMul(num_latent_factors, &item_weights[i * num_latent_factors],
          &item_norms_matrix[i * num_latent_factors],
          &normalized_item_weights[i * num_latent_factors]);
  }
  // normalized_item_weights[i, j] = item_weights[i, j]/||i||
  for (i = 0; i < num_clusters; i++) {
    vsMul(num_latent_factors, &centroids[i * num_latent_factors],
          &centroid_norms_matrix[i * num_latent_factors],
          &normalized_centroids[i * num_latent_factors]);
  }
  // normalized_centroids[i, j] = centroids[i, j]/||i||

  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, alpha,
              normalized_centroids, k, normalized_item_weights, k, beta,
              cos_theta_ics, n);
  // cos_theta_ics =
  // normalized_centroids^T (num_clusters x  num_latent_factors)
  // * normalized_item_weights (num_latent_factors * num_items)
  // cos_theta_ics[i, j] = item_weights[i, j] * centroids[i, j] / ||i|| ||c||

  float *theta_ics = (float *)_malloc(sizeof(float) * num_clusters * num_items);
  vsAcos(num_clusters * num_items, cos_theta_ics, theta_ics);
  remove_nans(theta_ics, num_clusters * num_items);

  MKL_free(normalized_centroids);
  MKL_free(normalized_item_weights);
  MKL_free(item_norms_matrix);
  MKL_free(centroid_norms_matrix);
  MKL_free(cos_theta_ics);

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

  float *users_dot_centroid = (float *)_malloc(sizeof(float) * num_users);
  float *centroidNorm = (float *)_malloc(sizeof(float) * num_latent_factors);

  // TODO: inefficient
  centroidNorm[0] = cblas_snrm2(num_latent_factors, centroid, 1);
  size_t i = 1;
  for (i = 1; i < num_latent_factors; i++) {
    centroidNorm[i] = centroidNorm[0];
  }

  float *userNorms = computeClusterUserNorms(
      user_weights, num_users, num_latent_factors, userNormTuple_array);

  vsInv(num_latent_factors, centroidNorm, centroidNorm);
  vsInv((num_users) * num_latent_factors, userNorms, userNorms);

  vsMul(num_latent_factors, centroid, centroidNorm, centroidNorm);
  for (i = 0; i < num_users; i++) {
    vsMul(num_latent_factors, &user_weights[i * num_latent_factors],
          &userNorms[i * num_latent_factors],
          &userNorms[i * num_latent_factors]);
  }

  cblas_sgemv(CblasRowMajor, CblasNoTrans, m, k, alpha, userNorms, k,
              centroidNorm, stride, beta, users_dot_centroid, stride);

  // now compute theta_uc's by taking arccosine
  float *theta_ucs = (float *)_malloc(sizeof(float) * num_users);
  vsAcos(num_users, users_dot_centroid, theta_ucs);
  remove_nans(theta_ucs, num_users);
  MKL_free(userNorms);
  MKL_free(centroidNorm);
  MKL_free(users_dot_centroid);
  return theta_ucs;
}
