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
                         const int num_clusters, const float *item_norms, const float *centroid_norms,
                         const float **normalized_centroids_ptr) {
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

  *normalized_centroids_ptr = normalized_centroids;


  cblas_scopy(num_items*num_latent_factors, item_weights, 1, normalized_item_weights, 1);
  float inv_item_norms[num_items];
  cblas_scopy(num_items, item_norms, 1, inv_item_norms, 1);
  vsInv(num_items, inv_item_norms, inv_item_norms);

  for (int i = 0; i < num_items; i++){
    cblas_sscal(num_latent_factors, inv_item_norms[i], &normalized_item_weights[i*num_latent_factors], 1);
  }

  cblas_scopy(num_clusters*num_latent_factors, centroids, 1, normalized_centroids, 1);
  float inv_centroid_norms[num_clusters];
  cblas_scopy(num_clusters, centroid_norms, 1, inv_centroid_norms, 1);
  vsInv(num_clusters, inv_centroid_norms, inv_centroid_norms);

  for (int i = 0; i < num_clusters; i++){
    cblas_sscal(num_latent_factors, inv_centroid_norms[i], &normalized_centroids[i*num_latent_factors], 1);
  }

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

  // _free(normalized_centroids);
  _free(normalized_item_weights);
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
                                      const int num_latent_factors, const float *centroid_norm) {
  const int m = num_users;
  const int k = num_latent_factors;

  const float alpha = 1.0;
  const float beta = 0.0;
  const int stride = 1;

  float *users_dot_centroid = (float *)_malloc(sizeof(float) * num_users);
  // float centroid_norm = cblas_snrm2(num_latent_factors, centroid, 1);

  float inv_centroid_norm = 1.0f / *centroid_norm;
  float new_centroid[num_latent_factors];
  cblas_scopy(num_latent_factors, centroid, 1, new_centroid, 1); 
  cblas_sscal(num_latent_factors, inv_centroid_norm, new_centroid, 1);

  float *user_norms_matrix =
      (float *)_malloc(sizeof(float) * num_users * num_latent_factors);
  cblas_scopy(num_users*num_latent_factors, user_weights, 1, user_norms_matrix, 1);
  float inv_user_norms[num_users];
  cblas_scopy(num_users, user_norms, 1, inv_user_norms, 1); 
  vsInv(num_users, inv_user_norms, inv_user_norms); 

  for (int i = 0; i < num_users; i++) {
      cblas_sscal(num_latent_factors, inv_user_norms[i], &user_norms_matrix[i*num_latent_factors], 1);
  }

  cblas_sgemv(CblasRowMajor, CblasNoTrans, m, k, alpha, user_norms_matrix, k,
              new_centroid, stride, beta, users_dot_centroid, stride);

  // now compute theta_uc's by taking arccosine
  float *theta_ucs = (float *)_malloc(sizeof(float) * num_users);
  vsAcos(num_users, users_dot_centroid, theta_ucs);
  remove_nans(theta_ucs, num_users);
  _free(user_norms_matrix);
  _free(users_dot_centroid);
  return theta_ucs;
}

float *compute_all_theta_ucs(const float *user_weights, const float *user_norms, const float *centroids,
                           const float *centroid_norms, const int num_latent_factors,
                           const int num_users, const int num_clusters, const std::vector<int>* cluster_index,
                           const int* num_users_so_far_arr,
                           float *normalized_centroids){
    
//    const int m = num_users;
    int m;
    const int k = num_latent_factors;
    
    const float alpha = 1.0;
    const float beta = 0.0;
    const int stride = 1;
    
    int i;
    
    float *theta_ucs = (float *)_malloc(sizeof(float) * num_users);
    float *user_norms_matrix = (float *)_malloc(sizeof(float) * num_users * num_latent_factors);
    // float *centroids_norm_matrix = (float *)_malloc(sizeof(float) * num_clusters * num_latent_factors);
    
    float *centroids_norm_matrix = normalized_centroids;

    // float *users_dot_centroid = (float *)_malloc(sizeof(float) * max_cluster_users);

    
    cblas_scopy(num_users*num_latent_factors, user_weights, 1, user_norms_matrix, 1);
    float inv_user_norms[num_users];
    cblas_scopy(num_users, user_norms, 1, inv_user_norms, 1);
    vsInv(num_users, inv_user_norms, inv_user_norms);

    for (i = 0; i < num_users; i++) {
        cblas_sscal(num_latent_factors, inv_user_norms[i], &user_norms_matrix[i*num_latent_factors], 1);
    }
    
    // cblas_scopy(num_clusters*num_latent_factors, centroids, 1, centroids_norm_matrix, 1);
    // float inv_cluster_norms[num_clusters];
    // cblas_scopy(num_clusters, centroid_norms, 1, inv_cluster_norms, 1);
    // vsInv(num_clusters, inv_cluster_norms, inv_cluster_norms);

    // for (i = 0; i < num_clusters; i++) {
    //     cblas_sscal(num_latent_factors, inv_cluster_norms[i], &centroids_norm_matrix[i*num_latent_factors], 1);
    // }
    
    
    for (i = 0; i < num_clusters; i++) {
        m = cluster_index[i].size();
        float users_dot_centroid[m];
        const int num_users_so_far = num_users_so_far_arr[i];
        
        cblas_sgemv(CblasRowMajor, CblasNoTrans, m, k, alpha, &user_norms_matrix[num_users_so_far*num_latent_factors], k,
                    &centroids_norm_matrix[i*num_latent_factors], stride, beta, users_dot_centroid, stride);
        
        vsAcos(m, users_dot_centroid, &theta_ucs[num_users_so_far]);

    }
    
    _free(user_norms_matrix);
    // _free(users_dot_centroid);
    _free(centroids_norm_matrix);
    return theta_ucs;
    
}
