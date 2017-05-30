//
//  algo.hpp
//  SimDex
//
//

#ifndef algo_hpp
#define algo_hpp

#include "types.hpp"
#include <iostream>
#include <vector>

std::vector<float> linspace(const float start, const float end, const int num);
int find_theta_bin_index(const float theta_uc,
                         const std::vector<float> theta_bins,
                         const int num_bins);
void computeTopKForCluster(
    int *top_K_items, const size_t cluster_id, const double *centroid,
    const std::vector<int> &user_ids_in_cluster, const double *user_weights,
    const double *item_weights, const float *item_norms, const float *theta_ics,
    const float &centroid_norm, const size_t num_items,
    const size_t num_latent_factors, const size_t K, const size_t item_batch_size,
    float *upper_bounds, int *sorted_upper_bounds_indices,
    float *sorted_upper_bounds, double *sorted_item_weights,
    std::ofstream &user_stats_file);

#endif /* algo_hpp */
