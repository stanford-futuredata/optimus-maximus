//
//  arith.hpp
//  fomo_preproc
//
//  Created by Geet Sethi on 10/27/16.
//  Copyright © 2016 Geet Sethi. All rights reserved.
//

#ifndef arith_hpp
#define arith_hpp

#include <stdio.h>
#include "mz_special.hpp"

float *computeAllItemNorms(float *allItemWeights);
float *computeCosineSimilarityItemsCentroids(float *allItems, float *allCentroids);
float *computeCosineSimilarityUserCluster(float *users, float *centroid, int numUsers, userNormTuple_t *userNormTuple_array);

#endif /* arith_hpp */
