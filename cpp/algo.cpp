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
void check_against_naive(const double *user_weight, const double *item_weights,
                         const int num_items, const int num_latent_factors,
                         const int *computed_top_K,
                         const double *computed_scores, const int K) {
  const int m = num_items;
  const int k = num_latent_factors;

  const float alpha = 1.0;
  const float beta = 0.0;
  const int stride = 1;

  double scores[num_items];
  cblas_dgemv(CblasRowMajor, CblasNoTrans, m, k, alpha, item_weights, k,
              user_weight, stride, beta, scores, stride);

  // Sort item ids by their associated scores in descending order
  std::vector<int> v(num_items);
  std::iota(v.begin(), v.end(), 0);
  std::sort(v.begin(), v.end(),
            [&scores](int i1, int i2) { return scores[i1] > scores[i2]; });
  // Sort scores in descending order, too
  std::sort(scores, scores + num_items, std::greater<double>());
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

void computeTopKForCluster(
    int *top_K_items, const int cluster_id, const double *centroid,
    const std::vector<int> &user_ids_in_cluster, const double *user_weights,
    const double *item_weights, const float *item_norms, const float *theta_ics,
    const float &centroid_norm, const int num_items,
    const int num_latent_factors, const int num_bins, const int K,
    const int batch_size, float *upper_bounds, int *sorted_upper_bounds_indices,
    float *sorted_upper_bounds, double *sorted_item_weights,
    std::ofstream &user_stats_file) {

#ifdef STATS
  double time_start, time_end;

  dsecnd();
  time_start = dsecnd();
#endif

  const int num_users_in_cluster = user_ids_in_cluster.size();

  double *user_dot_items = (double *)_malloc(sizeof(double) * batch_size);
  float *user_norm_times_upper_bound =
      (float *)_malloc(sizeof(float) * batch_size);
  const int mod = batch_size - 1;  // assumes batch_size is a power of 2, so
  // mod is all 1's in binary, therefore
  // ind % batch_size == ind & mod

  // compute user_norms and theta_ucs for every user assigned to this cluster
  float *user_norms = compute_norms_vector(user_weights, num_users_in_cluster,
                                           num_latent_factors);
  float *theta_ucs = compute_theta_ucs_for_centroid(
      user_weights, user_norms, centroid, num_users_in_cluster,
      num_latent_factors, centroid_norm);
  // NOTE: both are now already in the right order, i.e., you can access
  // them sequentially. This is because we reordered the user weights to be
  // in cluster order in main (see build_cluster_index)

  const float theta_max =
      theta_ucs[cblas_isamax(num_users_in_cluster, theta_ucs, 1)];
  const std::vector<float> theta_bins = linspace(0.F, theta_max, num_bins);
  // theta_bins is correct

  int i, j, l;

  for (i = 0; i < num_bins; i++) {
    std::fill(&upper_bounds[i * num_items], &upper_bounds[(i + 1) * num_items],
              theta_bins[i]);
    vsSub(num_items, theta_ics, &upper_bounds[i * num_items],
          &upper_bounds[i * num_items]);
  }
  // upper_bounds[i] = theta_ic - theta_b
  ippsThreshold_32f_I((Ipp32f *)upper_bounds, num_bins * num_items,
                      (Ipp32f)0.0f, ippCmpLess);
  // all values less than 0 in upper_bounds[i] now set to 0
  // Same as:
  // for (i = 0; i < num_bins; ++i) {
  //   for (j = 0; j < num_items; ++j) {
  //     if (upper_bounds[i][j] < 0) {
  //       upper_bounds[i][j] = 0.F;
  //     }
  //   }
  // }
  vsCos(num_bins * num_items, upper_bounds, upper_bounds);
  // upper_bounds[i] = cos(theta_ic - theta_b)
  for (i = 0; i < num_bins; i++) {
    vsMul(num_items, item_norms, &upper_bounds[i * num_items],
          &upper_bounds[i * num_items]);
  }
// upper_bounds[i] = ||i|| * cos(theta_ic - theta_b)

#ifdef STATS
  time_end = dsecnd();
  const double upper_bounds_time = time_end - time_start;

  dsecnd();
  time_start = dsecnd();
#endif

  int *pBufSize = (int *)_malloc(sizeof(int));
  ippsSortRadixIndexGetBufferSize(num_items, ipp32f, pBufSize);
  Ipp8u *pBuffer = (Ipp8u *)_malloc(*pBufSize * sizeof(Ipp8u));
  const int src_stride_bytes = 4;
  for (i = 0; i < num_bins; i++) {
    ippsSortRadixIndexDescend_32f(
        (Ipp32f *)&upper_bounds[i * num_items], src_stride_bytes,
        (Ipp32s *)&sorted_upper_bounds_indices[i * num_items], num_items,
        pBuffer);
  }

#ifdef STATS
  time_end = dsecnd();
  const double sort_time = time_end - time_start;

  dsecnd();
  time_start = dsecnd();
#endif

  int batch_counter[num_bins];
  std::memset(batch_counter, 0, sizeof batch_counter);
  const int bin_offset = num_items * num_latent_factors;
  int item_id;

  // prepare initial batches
  for (i = 0; i < num_bins; i++) {
    for (j = 0; j < batch_size; j++) {
      item_id = sorted_upper_bounds_indices[i * num_items + batch_counter[i]];
      sorted_upper_bounds[i * num_items + batch_counter[i]] =
          upper_bounds[i * num_items + item_id];
      // copy item_weights into sorted_item_weights based on order of
      // sorted_upper_bounds_indices
      cblas_dcopy(
          num_latent_factors, &item_weights[item_id * num_latent_factors], 1,
          &sorted_item_weights
               [(i * bin_offset) + (batch_counter[i] * num_latent_factors)],
          1);
      batch_counter[i]++;
    }
  }

#ifdef STATS
  time_end = dsecnd();
  const double batch_time = time_end - time_start;
#endif

// ----------Computer Per User TopK Below------------------
#ifdef DEBUG
  const int num_users_to_compute =
      num_users_in_cluster < 30 ? num_users_in_cluster : 30;
#else
  const int num_users_to_compute = num_users_in_cluster;
#endif

  for (i = 0; i < num_users_to_compute; i++) {
#ifdef STATS
    dsecnd();
    time_start = dsecnd();
#endif

    const int bin_index =
        find_theta_bin_index(theta_ucs[i], theta_bins, num_bins);

    std::priority_queue<std::pair<double, int>,
                        std::vector<std::pair<double, int> >,
                        std::greater<std::pair<double, int> > > q;

    double score = 0.0;

    int m = std::min(batch_size,
                     num_items);  // not const, because it may be adjusted later
    const int k = num_latent_factors;

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const int stride = 1;

    cblas_dgemv(CblasRowMajor, CblasNoTrans, m, k, alpha,
                &sorted_item_weights[(bin_index * bin_offset)], k,
                &user_weights[i * num_latent_factors], stride, beta,
                user_dot_items, stride);

#pragma simd
    for (j = 0; j < batch_size; j++) {
      user_norm_times_upper_bound[j] =
          user_norms[i] * sorted_upper_bounds[bin_index * num_items + j];
    }

    for (j = 0; j < K; j++) {
      item_id = sorted_upper_bounds_indices[bin_index * num_items + j];
      score = user_dot_items[j];
      q.push(std::make_pair(score, item_id));
    }

    int num_items_visited = K;

    for (j = K; j < num_items; j++) {
      if (batch_counter[bin_index] == j) {
        // previous batches exhausted, need to add an additional batch
        const int true_batch_size =
            std::min(batch_size, num_items - batch_counter[bin_index]);
        // if we're at the very end, we may not need a full batch
        m = true_batch_size;  // change for upcoming sgemv operation
        for (int l = 0; l < true_batch_size; l++) {
          item_id = sorted_upper_bounds_indices
              [bin_index * num_items + batch_counter[bin_index]];
          sorted_upper_bounds
              [bin_index * num_items + batch_counter[bin_index]] =
                  upper_bounds[bin_index * num_items + item_id];
          cblas_dcopy(num_latent_factors,
                      &item_weights[item_id * num_latent_factors], 1,
                      &sorted_item_weights
                           [(bin_index * bin_offset) +
                            (batch_counter[bin_index] * num_latent_factors)],
                      1);
          batch_counter[bin_index]++;
        }
      }

      if ((j & mod) == 0) {
        cblas_dgemv(CblasRowMajor, CblasNoTrans, m, k, alpha,
                    &sorted_item_weights
                         [(bin_index * bin_offset) + (j * num_latent_factors)],
                    k, &user_weights[i * num_latent_factors], stride, beta,
                    user_dot_items, stride);

#pragma simd
        for (l = 0; l < batch_size; l++) {
          user_norm_times_upper_bound[l] =
              user_norms[i] *
              sorted_upper_bounds[bin_index * num_items + j + l];
        }
      }

      if (q.top().first > user_norm_times_upper_bound[j & mod]) {
        break;
      }
      item_id = sorted_upper_bounds_indices[bin_index * num_items + j];
      score = user_dot_items[j & mod];
      num_items_visited++;
      if (q.top().first < score) {
        q.pop();
        q.push(std::make_pair(score, item_id));
      }
    }
#ifdef DEBUG
    double top_K_scores[K];
#endif
    for (j = 0; j < K; j++) {
      std::pair<double, int> p = q.top();
#ifdef DEBUG
      top_K_scores[K - 1 - j] = p.first;
#endif
      // don't need to store score
      top_K_items[i * K + K - 1 - j] = p.second;  // store item ID
      q.pop();
    }
#ifdef STATS
    time_end = dsecnd();
    const double user_top_K_time = time_end - time_start;
#endif

#ifdef DEBUG
    std::cout << "User ID " << user_ids_in_cluster[i] << std::endl;
    check_against_naive(&user_weights[i * num_latent_factors], item_weights,
                        num_items, num_latent_factors, &top_K_items[i * K],
                        top_K_scores, K);
#endif
#ifdef STATS
    const double total_user_time_ms =
        1000 * (user_top_K_time + batch_time + sort_time + upper_bounds_time);
    user_stats_file << cluster_id << "," << theta_ucs[i] << ","
                    << theta_bins[bin_index] << "," << num_items_visited << ","
                    << total_user_time_ms << std::endl;
#endif
  }

  // ----------Free Allocated Memory Below-------
  _free(pBufSize);
  _free(pBuffer);
  _free(user_norms);
  _free(theta_ucs);
  _free(user_dot_items);
  _free(user_norm_times_upper_bound);
  MKL_Free_Buffers();
}
