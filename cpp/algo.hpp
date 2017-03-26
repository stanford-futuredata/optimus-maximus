//
//  algo.hpp
//  Simdex
//
//

#ifndef algo_hpp
#define algo_hpp

#include <stdio.h>
#include <iostream>
#include <vector>
#include "types.hpp"

void computeTopKForCluster(
    std::vector<size_t> *cluster_index, const size_t clusterID,
    const size_t user_offset, float *user_weights, float *centroids,
    float *allItemWeights, float *allItemNorms, size_t numBins, const size_t K,
    float *vectorized_Acos_output_ic, const size_t num_items,
    const size_t num_latent_factors, std::ofstream &user_stats_file);

#endif /* algo_hpp */
