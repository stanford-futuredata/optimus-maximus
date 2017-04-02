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

void computeTopKForCluster(const int cluster_id, const float *centroid,
                           const std::vector<int> &user_ids_in_cluster,
                           const float *user_weights, const float *item_weights,
                           const float *item_norms, const float *theta_ics,
                           const int num_items, const int num_latent_factors,
                           const int num_bins, const int K,
                           std::ofstream &user_stats_file,
                           const int batch_size,
                           const float *centroid_norm,
                           const float *user_norms,
                           const float *theta_ucs);

#endif /* algo_hpp */
