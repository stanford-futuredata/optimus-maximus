//
//  mz_special.hpp
//  fomo_preproc
//
//  Created by Geet Sethi on 10/27/16.
//  Copyright © 2016 Geet Sethi. All rights reserved.
//

#ifndef mz_special_hpp
#define mz_special_hpp

#include <stdio.h>
#include "parser.hpp"

typedef struct userNormTuple {
  int userID;
  float userNorm;
} userNormTuple_t;

typedef struct userThetaUcTuple {
  int userID;
  float theta_uc;
} userThetaUcTuple_t;

typedef struct upperBoundItem {
  int itemID;
  float upperBound;
} upperBoundItem_t;

void computeTopKForCluster(fullCluster_t *fullClusterData, int clusterID,
                           float *fullUserData, float *clusterCentroids,
                           float *allItemWeights, float *allItemNorms,
                           int numBins, int k, float *vectorized_Acos_output_ic,
                           std::ofstream &misc_file);

#endif /* mz_special_hpp */
