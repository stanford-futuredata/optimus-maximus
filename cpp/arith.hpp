//
//  arith.hpp
//  SimDex
//
//  Created by Geet Sethi on 10/27/16.
//  Copyright © 2016 Geet Sethi. All rights reserved.
//

#ifndef arith_hpp
#define arith_hpp

#include "types.hpp"

float *compute_norms_vector(const float *matrix_weights, const int num_rows,
                            const int num_cols);
float *compute_theta_ics(const float *item_weights, const float *centroids,
                         const int num_items, const int num_latent_factors,
                         const int num_clusters);
float *compute_theta_ucs_for_centroid(const float *user_weights,
                                      const float *user_norms,
                                      const float *centroid,
                                      const int num_users,
                                      const int num_latent_factors);

#endif /* arith_hpp */
