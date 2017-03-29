//
//  parser.cpp
//  fomo_preproc
//
//  Created by Geet Sethi on 10/24/16.
//  Copyright © 2016 Geet Sethi. All rights reserved.
//

#include "parser.hpp"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unordered_map>
#include <iostream>
#include <fstream>
#include <mkl.h>

int numberUsers;
int numFeatures;
int numItems;
int numClusters;

void setNumFeatures(int x) { numFeatures = x; }

void setNumItems(int x) { numItems = x; }

void setNumClusters(int x) { numClusters = x; }

// returns a struct containing an array of all cluster info (stored as
// |cluster_id|numUsers|users|) and an offset array
// which is indexed by clusterID and returns the starting index of the cluster
// in the data array.

fullCluster_t *parseClusers(FILE *cluserIDsInverted, FILE *cluserUserCounts) {
  rewind(cluserIDsInverted);
  rewind(cluserUserCounts);

  int cluserUserCounts_array[numClusters];

  char *lineptr;
  size_t length;

  lineptr = NULL;
  length = 0;
  ssize_t char_read;
  char *data;
  char *data2;
  char *saveptr1;
  char *saveptr2;

  char *saveptr3;
  int currentCluster = 0;
  int temp = 0;

  char *token;

  int numUsers = 0;

  while ((char_read = getline(&lineptr, &length, cluserUserCounts)) != -1) {
    if ((token = strtok_r(lineptr, ",", &saveptr3)) != NULL) {
      int index = atoi(token);
      token = strtok_r(NULL, ",", &saveptr3);
      int userCount = atoi(token);
      numUsers += userCount;
      cluserUserCounts_array[index] = userCount;
    }

    while ((token = strtok_r(NULL, ",", &saveptr3)) != NULL) {
      int index = atoi(token);
      token = strtok_r(NULL, ",", &saveptr3);
      int userCount = atoi(token);
      numUsers += userCount;
      cluserUserCounts_array[index] = userCount;
    }
  }

  int *allClusters =
      (int *)malloc(sizeof(int) * (numUsers + (2 * numClusters)));
  int *clusterOffset = (int *)malloc(sizeof(int) * numClusters);
  int offset = 0;

  while ((char_read = getline(&lineptr, &length, cluserIDsInverted)) != -1) {
    if ((data = strtok_r(lineptr, ":", &saveptr1)) != NULL) {
      currentCluster = atoi(data);
      allClusters[offset] = currentCluster;
      clusterOffset[currentCluster] = offset;
      offset++;
      allClusters[offset] = cluserUserCounts_array[currentCluster];
      offset++;
    }
    if ((data = strtok_r(NULL, ":", &saveptr1)) != NULL) {
      if ((data2 = strtok_r(data, ",", &saveptr2)) != NULL) {
        data2 += 2;
        temp = atoi(data2);
        allClusters[offset] = temp;
        offset++;
      }
      while (((data2 = strtok_r(NULL, ",", &saveptr2)) != NULL)) {
        if (data2[strlen(data) - 1] == ']') {
          data2[strlen(data) - 1] = '\0';
        }
        temp = atoi(data2);
        allClusters[offset] = temp;
        offset++;
      }
    }
  }

  fullCluster_t *fullClusterData =
      (fullCluster_t *)malloc(sizeof(struct fullCluster));
  fullClusterData->clusterArray = allClusters;
  fullClusterData->clusterOffset = clusterOffset;  // offset points to start of
                                                   // cluster block (i.e. the
                                                   // cluster_id slot)

  numberUsers = numUsers;

  return fullClusterData;
}

// returns a struct containing an array of all user vectors (stored as array of
// floats) and an offset array
// which is indexed by userID and returns the starting index of the user vector
// in the data array.

float *parseAllUsers(FILE *userWeights, fullCluster_t *allClusters,
                     std::unordered_map<int, int> userToClusterMap) {
  rewind(userWeights);

  //    std::ofstream myfile;
  //    std::string outputFile;
  //    outputFile = "newUserMappingSTAGE2_" + std::to_string(numberUsers) +
  // ".txt";
  //
  //    myfile.open (outputFile);
  //    myfile << "Original User ID, New User ID\n";

  float *allUsers = (float *)malloc(sizeof(float) * numberUsers * numFeatures);
  char *lineptr;
  size_t length;

  lineptr = NULL;
  length = 0;
  ssize_t char_read;
  char *data;

  int offset = 0;
  int tempUser = 0;
  int cluster = 0;
  int usersInCluster = 0;
  int clusterOffset = 0;
  int i = 0;

  int runningCount = 0;
  int count = 0;

  while ((char_read = getline(&lineptr, &length, userWeights)) != -1) {
    if ((data = strtok(lineptr, ",")) != NULL) {
      data++;
      tempUser = atoi(data);
      count = 0;
      auto search = userToClusterMap.find(tempUser);
      if (search == userToClusterMap.end()) {
        printf("user lookup in map failed. exiting.\n");
        exit(1);
      } else
        cluster = search->second;

      clusterOffset = allClusters->clusterOffset[cluster];
      usersInCluster = allClusters->clusterArray[clusterOffset + 1];

      for (i = 0; i < usersInCluster; i++) {
        if (tempUser == allClusters->clusterArray[clusterOffset + 2 + i]) {
          allClusters->clusterArray[clusterOffset + 2 + i] = runningCount;
        }
      }
    }
    while (((data = strtok(NULL, ",")) != NULL) && count < numFeatures) {
      if (data[strlen(data) - 1] == ')') {
        data[strlen(data) - 1] = '\0';
      }
      allUsers[offset] = atof(data);
      offset++;
      count++;
    }
    //        myfile << tempUser << "," << runningCount << "\n";
    runningCount++;
  }

  return allUsers;
};

// returns an array of size numberItems*numFeature floats. Index into the array
// by
// itemID*numFeatures -- this points to first feature of item.
// ASSUMES ITEMIDS ARE SEQUENTIALLY ASSIGNED IN FILE.

float *parseAllItems(FILE *itemWeights) {
  rewind(itemWeights);

  //    float *allItems = (float
  // *)malloc(sizeof(float)*(numItems+1)*numFeatures);

  float *allItems =
      (float *)mkl_malloc(sizeof(float) * (numItems + 1) * numFeatures, 64);
  if (allItems == NULL) {
    printf("\n ERROR: Can't allocate memory for allItems. Aborting... \n\n");
    mkl_free(allItems);
    exit(1);
  }

  for (int i = 0; i < numFeatures; i++) {
    allItems[i] = 0.0;
  }

  char *lineptr;
  size_t length;

  lineptr = NULL;
  length = 0;
  ssize_t char_read;
  char *data;

  int offset = 0;
  int tempItem = 0;
  int count = 0;

  while ((char_read = getline(&lineptr, &length, itemWeights)) != -1) {
    if ((data = strtok(lineptr, ",")) != NULL) {
      data++;
      tempItem = atoi(data);
      offset = tempItem * numFeatures;
      count = 0;
    }
    while (((data = strtok(NULL, ",")) != NULL) && count < numFeatures) {
      if (data[strlen(data) - 1] == ')') {
        data[strlen(data) - 1] = '\0';
      }
      allItems[offset] = atof(data);
      offset++;
      count++;
    }
  }

  return allItems;
}

float *parseClusterCentroids(FILE *clusterCentroids) {
  rewind(clusterCentroids);

  //    float *allCentroids = (float
  // *)malloc(sizeof(float)*(numClusters)*numFeatures);

  float *allCentroids =
      (float *)mkl_malloc(sizeof(float) * (numClusters) * numFeatures, 64);
  if (allCentroids == NULL) {
    printf(
        "\n ERROR: Can't allocate memory for allCentroids. Aborting... \n\n");
    mkl_free(allCentroids);
    exit(1);
  }

  char *lineptr;
  size_t length;

  lineptr = NULL;
  length = 0;
  ssize_t char_read;
  char *data;

  int offset = 0;
  int tempCluster = 0;
  int count = 0;

  while ((char_read = getline(&lineptr, &length, clusterCentroids)) != -1) {
    if ((data = strtok(lineptr, ",")) != NULL) {
      data++;
      tempCluster = atoi(data);
      offset = tempCluster * numFeatures;
      count = 0;
    }
    while (((data = strtok(NULL, ",")) != NULL) && count < numFeatures) {
      if (data[strlen(data) - 1] == ')') {
        data[strlen(data) - 1] = '\0';
      }
      allCentroids[offset] = atof(data);
      offset++;
      count++;
    }
  }

  return allCentroids;
}

std::unordered_map<int, int> userToClusterMap(FILE *userToClusterFile) {
  rewind(userToClusterFile);
  std::unordered_map<int, int> userClusterMap;

  char *lineptr;
  size_t length;

  lineptr = NULL;
  length = 0;
  ssize_t char_read;

  char *saveptr3;

  char *token;

  int user = 0;
  int cluster = 0;

  while ((char_read = getline(&lineptr, &length, userToClusterFile)) != -1) {
    if ((token = strtok_r(lineptr, ":", &saveptr3)) != NULL) {
      user = atoi(token);
      token = strtok_r(NULL, ",", &saveptr3);
      cluster = atoi(token);
      userClusterMap.insert({user, cluster});
    }
  }

  return userClusterMap;
}

// Going to make a new user array where users are arranged in cluster order.
// Needed for usersInClusters X Centroid Matrix-Vector Prod.
float *reorderUsersInClusterOrder(float *allUsers,
                                  fullCluster_t *fullClusterData) {

  //    float *allUsers = (float
  // *)malloc(sizeof(float)*numberUsers*numFeatures);
  float *allUsersNew =
      (float *)mkl_malloc(sizeof(float) * (numberUsers) * numFeatures, 64);
  if (allUsersNew == NULL) {
    printf("\n ERROR: Can't allocate memory for allUsersNew. Aborting... \n\n");
    mkl_free(allUsersNew);
    exit(1);
  }

  //    std::ofstream myfile;
  //    std::string outputFile;
  //    outputFile = "newUserMappingSTAGE1_" + std::to_string(numberUsers) +
  // ".txt";
  //
  //    myfile.open (outputFile);
  //    myfile << "Original User ID, New User ID\n";

  int clusterOffset;
  int numUsersInCluster;
  int newUserNumber = 0;
  int oldUserNumber;

  int i, j, k;

  for (i = 0; i < numClusters; i++) {
    clusterOffset = fullClusterData->clusterOffset[i];
    numUsersInCluster = fullClusterData->clusterArray[clusterOffset + 1];
    for (j = 0; j < numUsersInCluster; j++) {
      oldUserNumber = fullClusterData->clusterArray[clusterOffset + 2 + j];
      for (k = 0; k < numFeatures; k++) {
        allUsersNew[(newUserNumber * numFeatures) + k] =
            allUsers[(oldUserNumber * numFeatures) + k];
      }
      fullClusterData->clusterArray[clusterOffset + 2 + j] = newUserNumber;
      //            myfile << oldUserNumber << "," << newUserNumber << "\n";
      newUserNumber++;
    }
  }

  free(allUsers);
  return allUsersNew;
}
