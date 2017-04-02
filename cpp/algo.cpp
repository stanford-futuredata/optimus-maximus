//
//  algo.cpp
//  SimDex
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

#include <queue>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <numeric>

#include <mkl.h>
#include <ipps.h>

std::vector<float> linspace(const float start, const float end, const int num) {
  float delta = (end - start) / num;

  std::vector<float> linspaced;
  // start from 1; omit start (we only care about upper bounds)
  for (int i = 0; i < num; ++i) {
    linspaced.push_back(start + delta * (i + 1));
  }
  return linspaced;
}

/**
 * Find index of smallest theta_b that is greater than theta_uc,
 * so we can find the right list of sorted upper bounds for a given
 * user
 **/
int find_theta_bin_index(const float theta_uc,
                         const std::vector<float> theta_bins,
                         const int num_bins) {
  for (int i = 0; i < num_bins; ++i) {
    if (theta_uc <= theta_bins[i]) {
      return i;
    }
  }
  return num_bins - 1;
}

#ifdef DEBUG
void check_against_naive(const float *user_weight, const float *item_weights,
                         const int num_items, const int num_latent_factors,
                         const int *computed_top_K,
                         const float *computed_scores, const int K) {
  const int m = num_items;
  const int k = num_latent_factors;

  const float alpha = 1.0;
  const float beta = 0.0;
  const int stride = 1;

  float scores[num_items];
  cblas_sgemv(CblasRowMajor, CblasNoTrans, m, k, alpha, item_weights, k,
              user_weight, stride, beta, scores, stride);

  // Sort item ids by their associated scores in descending order
  std::vector<int> v(num_items);
  std::iota(v.begin(), v.end(), 0);
  std::sort(v.begin(), v.end(),
            [&scores](int i1, int i2) { return scores[i1] > scores[i2]; });
  // Sort scores in descending order, too
  std::sort(scores, scores + num_items, std::greater<float>());
  // First compare scores
  for (int i = 0; i < K; ++i) {
    if (scores[i] != computed_scores[i]) {
      std::cout << "scores[i] = " << scores[i]
                << " does not equal computed_scores[i] = " << computed_scores[i]
                << ", i = " << i << std::endl;
      exit(1);
    }
  }
  for (int i = 0; i < K; ++i) {
    if (v[i] != computed_top_K[i]) {
      std::cout << "v[i] = " << v[i]
                << " does not equal computed_top_K[i] = " << computed_top_K[i]
                << ", i = " << i << std::endl;
    }
  }
}
#endif

void computeTopKForCluster(const int cluster_id, const float *centroid,
                           const std::vector<int> &user_ids_in_cluster,
                           const float *user_weights, const float *item_weights,
                           const float *item_norms, const float *theta_ics,
                           const int num_items, const int num_latent_factors,
                           const int num_bins, const int K,
                           std::ofstream &user_stats_file,
                           const int batch_size, const float *centroid_norm) {

  double time_start, time_end, upperBoundCreation_time, sortUpperBound_time,
      computeTopK_time;

  const int num_users_in_cluster = user_ids_in_cluster.size();

  time_start = dsecnd();
  time_start = dsecnd();

  // compute user_norms and theta_ucs for every user assigned to this cluster

  float *user_norms = compute_norms_vector(user_weights, num_users_in_cluster,
                                           num_latent_factors);
  float *theta_ucs =
      compute_theta_ucs_for_centroid(user_weights, user_norms, centroid,
                                     num_users_in_cluster, num_latent_factors, centroid_norm);
  
  // NOTE: both are now already in the right order, i.e., you can access
  // them sequentially. This is because we reordered the user weights to be
  // in cluster order in main.cpp (see build_cluster_index)

  const float theta_max =
      theta_ucs[cblas_isamax(num_users_in_cluster, theta_ucs, 1)];
  const std::vector<float> theta_bins = linspace(0.F, theta_max, num_bins);
  // theta_bins is correct
  float temp_upper_bounds[num_items];
  float upper_bounds[num_bins][num_items];

  int i, j;
  for (i = 0; i < num_bins; i++) {
    // TODO: inefficient copy value to all items in the array
    for (j = 0; j < num_items; j++) {
      temp_upper_bounds[j] = theta_bins[i];
    }
    // temp_upper_bounds = theta_b
    vsSub(num_items, theta_ics, temp_upper_bounds, temp_upper_bounds);
    // temp_upper_bounds = theta_ic - theta_b
    for (int l = 0; l < num_items; ++l) {
      // TODO: inefficient
      if (temp_upper_bounds[l] < 0) {
        temp_upper_bounds[l] = 0.F;
      }
    }
    vsCos(num_items, temp_upper_bounds, temp_upper_bounds);
    // temp_upper_bounds = cos(theta_ic - theta_b)
    vsMul(num_items, item_norms, temp_upper_bounds, upper_bounds[i]);
    // upper_bounds[i] = ||i|| * cos(theta_ic - theta_b)
  }

  // upper_bounds are correct
  time_end = dsecnd();
  upperBoundCreation_time = (time_end - time_start);

  time_start = dsecnd();
  time_start = dsecnd();

  int sorted_upper_bounds_indices[num_bins][num_items];
  int *pBufSize = (int *)malloc(sizeof(int));
  ippsSortRadixIndexGetBufferSize(num_items, ipp32f, pBufSize);
  Ipp8u *pBuffer = (Ipp8u *)malloc(*pBufSize * sizeof(Ipp8u));
  int srcStrideBytes = 4;
  for (i = 0; i < num_bins; i++) {
    ippsSortRadixIndexDescend_32f((Ipp32f *)upper_bounds[i], srcStrideBytes,
                                  (Ipp32s *)sorted_upper_bounds_indices[i],
                                  num_items, pBuffer);
  }

  time_end = dsecnd();
  sortUpperBound_time = (time_end - time_start);

  int batch_counter[num_bins];
  std::memset(batch_counter, 0, sizeof batch_counter);
  float sorted_upper_bounds[num_bins][num_items];
  float *sorted_item_weights = (float *)_malloc(sizeof(float) * num_bins *
                                                num_items * num_latent_factors);
  const int bin_offset = num_items * num_latent_factors;
  int item_id;

  // compute initial batches
  for (i = 0; i < num_bins; i++) {
    for (j = 0; j < batch_size; j++) {
      item_id = sorted_upper_bounds_indices[i][batch_counter[i]];
      sorted_upper_bounds[i][batch_counter[i]] = upper_bounds[i][item_id];
      cblas_scopy(
          num_latent_factors, &item_weights[item_id * num_latent_factors], 1,
          &sorted_item_weights
               [(i * bin_offset) + (batch_counter[i] * num_latent_factors)],
          1);
      batch_counter[i]++;
    }
  }

  int mod = (batch_size) - 1;

  // ----------Computer Per User TopK Below------------------
  int top_K_items[num_users_in_cluster][K];

  time_start = dsecnd();
  time_start = dsecnd();

#ifdef DEBUG
  const int num_users_to_compute =
      num_users_in_cluster < 30 ? num_users_in_cluster : 30;
#else
  const int num_users_to_compute = num_users_in_cluster;
#endif

  float *user_dot_items = (float *)_malloc(sizeof(float) * batch_size);
  float *user_times_upper_bound = (float*)_malloc(sizeof(float)*batch_size);

  for (i = 0; i < num_users_to_compute; i++) {
    const int bin_index =
        find_theta_bin_index(theta_ucs[i], theta_bins, num_bins);

    std::priority_queue<std::pair<float, int>,
                        std::vector<std::pair<float, int> >,
                        std::greater<std::pair<float, int> > > q;

    float score = 0.F;
    int itemID = 0;

    int m = batch_size;  // may be adjusted later
    const int k = num_latent_factors;

    const float alpha = 1.0;
    const float beta = 0.0;
    const int stride = 1;

    cblas_sgemv(CblasRowMajor, CblasNoTrans, m, k, alpha,
                &sorted_item_weights[(bin_index * bin_offset)], k,
                &user_weights[i * num_latent_factors], stride, beta,
                user_dot_items, stride);
    cblas_scopy(batch_size, sorted_upper_bounds[bin_index], 1, user_times_upper_bound, 1 );
    cblas_sscal(batch_size, user_norms[i], user_times_upper_bound, 1);

    for (j = 0; j < K; j++) {
      itemID = sorted_upper_bounds_indices[bin_index][j];
      score = user_dot_items[j];
      q.push(std::make_pair(score, itemID));
    }
    int num_items_visited = K;

    for (j = K; j < num_items; j++) {
      if (batch_counter[bin_index] == j) {
        // previous batches exhausted, need to add an additional batch
        const int true_batch_size =
            std::min(batch_size, num_items - batch_counter[bin_index]);
        // if we're at the very end, we may not need a full batch
        m = true_batch_size;  // change for upcoming sgemv op
        for (int l = 0; l < true_batch_size; l++) {
          item_id =
              sorted_upper_bounds_indices[bin_index][batch_counter[bin_index]];
          sorted_upper_bounds[bin_index][batch_counter[bin_index]] =
              upper_bounds[bin_index][item_id];
          cblas_scopy(num_latent_factors,
                      &item_weights[item_id * num_latent_factors], 1,
                      &sorted_item_weights
                           [(bin_index * bin_offset) +
                            (batch_counter[bin_index] * num_latent_factors)],
                      1);
          batch_counter[bin_index]++;
        }
      }

      if ((j & mod) == 0) {
        cblas_sgemv(CblasRowMajor, CblasNoTrans, m, k, alpha,
                    &sorted_item_weights
                         [(bin_index * bin_offset) + (j * num_latent_factors)],
                    k, &user_weights[i * num_latent_factors], stride, beta,
                    user_dot_items, stride);
        cblas_scopy(batch_size, &sorted_upper_bounds[bin_index][j], 1, user_times_upper_bound, 1 );
        cblas_sscal(batch_size, user_norms[i], user_times_upper_bound, 1);
      }

      if (q.top().first > user_times_upper_bound[j & mod]) {
        break;
      }
      itemID = sorted_upper_bounds_indices[bin_index][j];
      score = user_dot_items[j & mod];
      num_items_visited++;
      if (q.top().first < score) {
        q.pop();
        q.push(std::make_pair(score, itemID));
      }
    }
#ifdef DEBUG
    float top_K_scores[K];
#endif
    for (j = 0; j < K; j++) {
      std::pair<float, int> p = q.top();
#ifdef DEBUG
      top_K_scores[K - 1 - j] = p.first;
#endif
      // don't need to store score
      top_K_items[i][K - 1 - j] = p.second;  // store item ID
      q.pop();
    }

#ifdef DEBUG
    std::cout << "User ID " << user_ids_in_cluster[i] << std::endl;
    check_against_naive(&user_weights[i * num_latent_factors], item_weights,
                        num_items, num_latent_factors, top_K_items[i],
                        top_K_scores, K);

    user_stats_file << user_ids_in_cluster[i] << "," << cluster_id << ","
                    << theta_ucs[i] << "," << num_items_visited << std::endl;
#endif
  }

  time_end = dsecnd();
  computeTopK_time = (time_end - time_start);

  // ----------Free Allocated Memory Below-------

  _free(user_norms);
  _free(theta_ucs);
  _free(sorted_item_weights);
  _free(user_dot_items);

  MKL_Free_Buffers();

  // printf("upper bound time: %f secs \n", upperBoundCreation_time);
  // printf("sort time: %f secs \n", sortUpperBound_time);
  // printf("compute top K time: %f secs \n", computeTopK_time);
  // creationTime += upperBoundCreation_time;
  // sortTime += sortUpperBound_time;
  // computeKTime += computeTopK_time;
}
