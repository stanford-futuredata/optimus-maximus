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

void computeTopKForCluster(
    const int cluster_id, const float *centroid,
    const std::vector<int> &user_ids_in_cluster, const float *user_weights,
    const float *item_weights, const float *item_norms, const float *theta_ics,
    const float &centroid_norm, const int num_items,
    const int num_latent_factors, const int num_bins, const int K,
    const int batch_size, float *upper_bounds, int *sorted_upper_bounds_indices,
    float *sorted_upper_bounds, float *sorted_item_weights,
    std::ofstream &user_stats_file);

#endif /* algo_hpp */
