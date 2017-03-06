//
//  parser.hpp
//  fomo_preproc
//
//  Created by Geet Sethi on 10/24/16.
//  Copyright © 2016 Geet Sethi. All rights reserved.
//

#ifndef parser_hpp
#define parser_hpp

#include <stdio.h>
#include <unordered_map>

typedef struct fullCluster{
    int *clusterArray;
    int *clusterOffset;
}fullCluster_t;

typedef struct fullUser{
    float *userArray;
    int *userOffset;
}fullUser_t;

void setNumFeatures(int x);
void setNumItems(int x);
void setNumClusters(int x);

fullCluster_t *parseClusers(FILE *cluserIDsInverted, FILE *cluserUserCounts);
float *parseAllUsers(FILE *userWeights, fullCluster_t *allClusters, std::unordered_map<int, int> userToClusterMap);
float *parseAllItems(FILE *itemWeights);
float *parseClusterCentroids(FILE *clusterCentroids);
std::unordered_map<int, int> userToClusterMap(FILE *userToClusterFile);
float *reorderUsersInClusterOrder(float *allUsers, fullCluster_t *fullClusterData);

#endif /* parser_hpp */
