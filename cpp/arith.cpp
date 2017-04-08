//
//  arith.cpp
//  SimDex
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

/**
 * Replace all NaNs in the array with zeroes
 **/
inline void remove_nans(float *arr, int num_elems) {
  // TODO: more efficient
  for (int i = 0; i < num_elems; ++i) {
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
float *compute_norms_matrix(const float *matrix_weights, const int num_rows,
                            const int num_cols) {
  float *matrix_norms = (float *)_malloc(sizeof(float) * num_rows * num_cols);
  for (int i = 0; i < num_rows; i++) {
    matrix_norms[i * num_cols] =
        cblas_snrm2(num_cols, &matrix_weights[i * num_cols], 1);
    for (int j = 1; j < num_cols; j++) {
      matrix_norms[(i * num_cols) + j] = matrix_norms[i * num_cols];
    }
  }

  return matrix_norms;
}

/**
 * Compute L2 norm per row for a given matrix. If the input matrix_weights is
 * num_rows x num_cols, the output will be a num_rows x 1 vector
 **/
float *compute_norms_vector(const float *matrix_weights, const int num_rows,
                            const int num_cols) {
  float *norms = (float *)_malloc(sizeof(float) * num_rows);

  for (int i = 0; i < num_rows; i++) {
    norms[i] = cblas_snrm2(num_cols, &matrix_weights[i * num_cols], 1);
  }
  return norms;
}

/**
 * Compute theta_ics: the angle between every centroid and every item in the
 * dataset
 **/
float *compute_theta_ics(const float *item_weights, const float *centroids,
                         const int num_items, const int num_latent_factors,
                         const int num_clusters) {
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

  vsInv(num_clusters * num_latent_factors, centroid_norms_matrix,
        centroid_norms_matrix);
  // centroid_norms_matrix now has 1/||c||
  vsInv(num_items * num_latent_factors, item_norms_matrix, item_norms_matrix);
  // item_norms_matrix now has 1/||i||

  int i = 0;
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

  _free(normalized_centroids);
  _free(normalized_item_weights);
  _free(item_norms_matrix);
  _free(centroid_norms_matrix);
  _free(cos_theta_ics);

  return theta_ics;
}

/**
 * Compute theta_ucs: angle between _single_ centroid and every user in the
 * dataset
 **/
float *compute_theta_ucs_for_centroid(const float *user_weights,
                                      const float *user_norms,
                                      const float *centroid,
                                      const int num_users,
                                      const int num_latent_factors) {
  const int m = num_users;
  const int k = num_latent_factors;

  const float alpha = 1.0;
  const float beta = 0.0;
  const int stride = 1;

  float *users_dot_centroid = (float *)_malloc(sizeof(float) * num_users);
  float centroid_norm[num_latent_factors];

  centroid_norm[0] = cblas_snrm2(num_latent_factors, centroid, 1);
  int i = 1;
  // TODO: inefficient
  for (; i < num_latent_factors; i++) {
    centroid_norm[i] = centroid_norm[0];
  }

  float *user_norms_matrix =
      (float *)_malloc(sizeof(float) * num_users * num_latent_factors);
  for (int i = 0; i < num_users; i++) {
    for (int j = 0; j < num_latent_factors; j++) {
      user_norms_matrix[(i * num_latent_factors) + j] = user_norms[i];
    }
  }

  vsInv(num_latent_factors, centroid_norm, centroid_norm);
  vsInv(num_users * num_latent_factors, user_norms_matrix, user_norms_matrix);

  vsMul(num_latent_factors, centroid, centroid_norm, centroid_norm);
  for (i = 0; i < num_users; i++) {
    vsMul(num_latent_factors, &user_weights[i * num_latent_factors],
          &user_norms_matrix[i * num_latent_factors],
          &user_norms_matrix[i * num_latent_factors]);
  }

  cblas_sgemv(CblasRowMajor, CblasNoTrans, m, k, alpha, user_norms_matrix, k,
              centroid_norm, stride, beta, users_dot_centroid, stride);

  // now compute theta_uc's by taking arccosine
  float *theta_ucs = (float *)_malloc(sizeof(float) * num_users);
  vsAcos(num_users, users_dot_centroid, theta_ucs);
  remove_nans(theta_ucs, num_users);
  _free(user_norms_matrix);
  _free(users_dot_centroid);
  return theta_ucs;
}
