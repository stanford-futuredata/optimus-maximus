//
//  types.hpp
//  Simdex
//
//

#ifndef types_hpp
#define types_hpp

typedef struct fullCluster {
  int *clusterArray;
  int *clusterOffset;
} fullCluster_t;

typedef struct fullUser {
  float *userArray;
  int *userOffset;
} fullUser_t;

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

#endif /* types_hpp */
