//
//  naive.cpp
//  fomo_preproc
//
//  Created by Geet Sethi on 10/28/16.
//  Copyright © 2016 Geet Sethi. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/stat.h>
#include <mkl.h>
#include <mkl_scalapack.h>
#include <functional>
#include <utility>
#include <queue>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <iostream>
#include <fstream>

int numFeatures;
int numItems;
int numUsers;

typedef unsigned long long u64;
#define DECLARE_ARGS(val, low, high) unsigned low, high
#define EAX_EDX_VAL(val, low, high) ((low) | ((u64)(high) << 32))
#define EAX_EDX_ARGS(val, low, high) "a"(low), "d"(high)
#define EAX_EDX_RET(val, low, high) "=a"(low), "=d"(high)
static inline unsigned long long _rdtsc2(void) {
  DECLARE_ARGS(val, low, high);

  asm volatile("rdtsc" : EAX_EDX_RET(val, low, high));

  return EAX_EDX_VAL(val, low, high);
}

int file_exist(const char *path) {
  struct stat buffer;
  return (stat(path, &buffer) == 0);
}

float *parseAllItems(FILE *itemWeights) {
  rewind(itemWeights);

  float *allItems =
      (float *)malloc(sizeof(float) * (numItems + 1) * numFeatures);
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
  int counter = 0;

  while ((char_read = getline(&lineptr, &length, itemWeights)) != -1) {
    if ((data = strtok(lineptr, ",")) != NULL) {
      data++;
      tempItem = atoi(data);
      offset = tempItem * numFeatures;
      counter = 0;
    }
    while (((data = strtok(NULL, ",")) != NULL) && counter < numFeatures) {
      if (data[strlen(data) - 1] == ')') {
        data[strlen(data) - 1] = '\0';
      }
      allItems[offset] = atof(data);
      offset++;
      counter++;
    }
  }

  return allItems;
}

float **parseAllUsers(FILE *userWeights) {
  rewind(userWeights);

  float **allUsers = (float **)malloc(numUsers * sizeof(float *));
  for (int i = 0; i < numUsers; i++)
    allUsers[i] = (float *)malloc((numFeatures + 1) * sizeof(float));
  char *lineptr;
  size_t length;

  lineptr = NULL;
  length = 0;
  ssize_t char_read;
  char *data;

  int offset = 0;
  int tempUser = 0;
  int userCount = 0;
  int counter = 0;

  while ((char_read = getline(&lineptr, &length, userWeights)) != -1) {
    if ((data = strtok(lineptr, ",")) != NULL) {
      data++;
      tempUser = atoi(data);
      allUsers[userCount][0] = tempUser;
      offset = 1;
      counter = 0;
    }
    while (((data = strtok(NULL, ",")) != NULL) && counter < numFeatures) {
      if (data[strlen(data) - 1] == ')') {
        data[strlen(data) - 1] = '\0';
      }
      allUsers[userCount][offset] = atof(data);
      offset++;
      counter++;
    }
    userCount++;
  }

  return allUsers;
}

int main(int argc, const char *argv[]) {
  // insert code here...

  if (argc != 7) {
    printf(
        "Usage: naive <numItems> <numLatentFactors> <numUsers> <K> <UserWeight "
        "File Name> <ItemWeight File Name> \n");
    exit(1);
  }

  for (int i = 5; i < 7; i++) {
    if (!file_exist(argv[i])) {
      printf("Could not open %s. Perhaps incorrect filename?\n", argv[i]);
      exit(1);
    }
  }

  numItems = atoi(argv[1]);
  numFeatures = atoi(argv[2]);
  numUsers = atoi(argv[3]);

  int k = atoi(argv[4]);

  FILE *userWeights;
  FILE *itemWeights;

  userWeights = fopen(argv[5], "r");
  itemWeights = fopen(argv[6], "r");

  float **allUsers = parseAllUsers(userWeights);
  float *allItems = parseAllItems(itemWeights);

  //    std::ofstream myfile;
  //    myfile.open ("naive_output.txt", std::ofstream::out |
  // std::ofstream::app);

  unsigned long long call_start, call_stop, call_run_time = 0;
  call_start = _rdtsc2();

  for (int i = 0; i < numUsers; i++) {
    std::vector<std::pair<float, int>> userItemScores(numItems);
    for (int j = 0; j < numItems; j++) {
      userItemScores.push_back({cblas_sdot(numFeatures, &allUsers[i][1], 1,
                                           &allItems[j * numFeatures], 1),
                                j});
    }
    std::sort(
        userItemScores.begin(), userItemScores.end(),
        [](const std::pair<int, int> &left, const std::pair<int, int> &right) {
          return left.first > right.first;
        });

    //        myfile << (int)allUsers[i][0] << " : [";
    //        myfile << userItemScores[0].second;
    //        for (int j = 1; j < k; j++) {
    //            myfile << ", " << userItemScores[j].second;
    //        }
    //        myfile << "]\n";
  }

  call_stop = _rdtsc2();
  call_run_time = call_stop - call_start;
  printf("run time: %llu\n", call_run_time);

  return 0;
}
