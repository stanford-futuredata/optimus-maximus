//
//  parser.hpp
//  fomo_preproc
//
//  Created by Geet Sethi on 10/24/16.
//  Copyright © 2016 Geet Sethi. All rights reserved.
//

#ifndef parser_hpp
#define parser_hpp

#include "types.hpp"
#include <stdio.h>
#include <unordered_map>

void setNumFeatures(int x);
void setNumItems(int x);
void setNumClusters(int x);

float *parse_weights_csv(const std::string filename, const size_t num_rows,
                         const size_t num_cols);
int *parse_ids_csv(const std::string filename, const size_t num_rows);

fullCluster_t *parseClusters(FILE *cluserIDsInverted, FILE *cluserUserCounts);
float *parseAllUsers(FILE *userWeights, fullCluster_t *allClusters,
                     std::unordered_map<int, int> userToClusterMap);
float *parseClusterCentroids(FILE *clusterCentroids);
std::unordered_map<int, int> userToClusterMap(FILE *userToClusterFile);
float *reorderUsersInClusterOrder(float *allUsers,
                                  fullCluster_t *fullClusterData);

#endif /* parser_hpp */
