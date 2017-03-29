//
//  cross_product.cpp
//  fomo_preproc
//
//  Created by Geet Sethi on 10/30/16.
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
#include <math.h>
#include <fstream>

int numFeatures;
int numItems;
int numUsers;
int topk;

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

float *parseAllItems(FILE *itemWeights) {
  rewind(itemWeights);

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

float *parseAllUsers(FILE *userWeights) {
  rewind(userWeights);

  //    std::ofstream myfile;
  //    myfile.open ("newUserMapping.txt");
  //    myfile << "Original User ID, New User ID\n";

  float *allUsers =
      (float *)mkl_malloc(sizeof(float) * numUsers * numFeatures, 64);
  if (allUsers == NULL) {
    printf("\n ERROR: Can't allocate memory for allUsers. Aborting... \n\n");
    mkl_free(allUsers);
    exit(1);
  }
  char *lineptr;
  size_t length;

  lineptr = NULL;
  length = 0;
  ssize_t char_read;
  char *data;

  int offset = 0;
  int tempUser = 0;

  int runningCount = 0;
  int count = 0;

  while ((char_read = getline(&lineptr, &length, userWeights)) != -1) {
    if ((data = strtok(lineptr, ",")) != NULL) {
      data++;
      tempUser = atoi(data);
      count = 0;
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

// float *computeAllItemNorms(float *allItemWeights){
//    float *allItemNorms = (float *)mkl_malloc(sizeof(float)*(numItems+1), 64);
//
//    for (int i = 0; i < (numItems+1); i++) {
//        allItemNorms[i] = cblas_snrm2(numFeatures,
// &allItemWeights[i*numFeatures], 1);
//    }
//
//    return allItemNorms;
//}

// float *computeAllUserNorms(float *allUserWeights){
//    float *allUserNorms = (float *)mkl_malloc(sizeof(float)*(numUsers), 64);
//
//    for (int i = 0; i < numUsers; i++) {
//        allUserNorms[i] = cblas_snrm2(numFeatures,
// &allUserWeights[i*numFeatures], 1);
//    }
//
//    return allUserNorms;
//}

float *computeAllItemNorms(float *allItemWeights) {
  float *allItemNorms =
      (float *)mkl_malloc(sizeof(float) * (numItems + 1) * numFeatures, 64);
  if (allItemNorms == NULL) {
    printf(
        "\n ERROR: Can't allocate memory for allItemNorms. Aborting... \n\n");
    mkl_free(allItemNorms);
    exit(1);
  }

  for (int i = 0; i < (numItems + 1); i++) {
    allItemNorms[i * numFeatures] =
        cblas_snrm2(numFeatures, &allItemWeights[i * numFeatures], 1);
    for (int j = 0; j < numFeatures; j++) {
      allItemNorms[(i * numFeatures) + j] = allItemNorms[i * numFeatures];
    }
  }

  return allItemNorms;
}

float *computeAllUserNorms(float *allUserWeights) {
  float *allUserNorms =
      (float *)mkl_malloc(sizeof(float) * (numUsers * numFeatures), 64);
  if (allUserNorms == NULL) {
    printf(
        "\n ERROR: Can't allocate memory for allUserNorms. Aborting... \n\n");
    mkl_free(allUserNorms);
    exit(1);
  }

  for (int i = 0; i < numUsers; i++) {
    allUserNorms[i * numFeatures] =
        cblas_snrm2(numFeatures, &allUserWeights[i * numFeatures], 1);
    for (int j = 0; j < numFeatures; j++) {
      allUserNorms[(i * numFeatures) + j] = allUserNorms[i * numFeatures];
    }
  }

  return allUserNorms;
}

void computeTopK(float *matrix, int users) {
  int log = 0;
  for (int i = 0; i < users; i++) {

    //        if (i > 120840) {
    //            log = 1;
    //        }

    std::priority_queue<std::pair<float, int>,
                        std::vector<std::pair<float, int> >,
                        std::greater<std::pair<float, int> > > q;
    //        if (log == 1) {
    //            printf("here 1: %d\n", i);
    //        }
    int j = 0;

    unsigned long index = i;
    int n = numItems + 1;
    index = index * n;

    for (j = 0; j < topk; j++) {
      q.push(std::make_pair(matrix[index + j], j));
      //            if (log == 1) {
      //                printf("here 2: %d\n", j);
      //            }
    }

    for (int j = topk; j < (numItems + 1); j++) {
      if (matrix[index + j] > q.top().first) {
        //                if (log == 1) {
        //                    printf("here 3: %d\n", j);
        //                }
        q.pop();
        //                if (log == 1) {
        //                    printf("here 4: %d\n", j);
        //                }
        q.push(std::make_pair(matrix[index + j], j));
        //                if (log == 1) {
        //                    printf("here 5: %d\n", j);
        //                }
      }
    }

    int topK[topk];

    for (int j = 0; j < topk; j++) {
      std::pair<float, int> p = q.top();
      topK[topk - 1 - j] = p.second;
      q.pop();
    }
    //        if (log == 1) {
    //            printf("user: %d\n", i);
    //        }
    //        printf("user: %d\n", i);
  }
}

int main(int argc, const char *argv[]) {

  if (argc != 7) {
    printf(
        "Usage: cross <numItems> <numUsers> <numLatentFactors> <k> <Weight "
        "Directory> <mkl_cores> \n");
    exit(1);
  }

  std::vector<std::string> allArgs(argv, argv + argc);
  std::string userWeightPath;
  std::string itemWeightPath;

  numItems = atoi(argv[1]);
  numFeatures = atoi(argv[3]);
  numUsers = atoi(argv[2]);
  topk = atoi(argv[4]);

  int maxThreads = atoi(argv[6]);
  MKL_Set_Num_Threads(maxThreads);
  printf("max num threads: %d\n", mkl_get_max_threads());

  if (((char)allArgs[5].back()) == '/') {
    userWeightPath = allArgs[5] + "userWeights.txt";
    itemWeightPath = allArgs[5] + "itemWeights.txt";

  } else {
    userWeightPath = allArgs[5] + "/userWeights.txt";
    itemWeightPath = allArgs[5] + "/itemWeights.txt";
  }

  FILE *userWeights;
  FILE *itemWeights;

  userWeights = fopen(userWeightPath.c_str(), "r");
  itemWeights = fopen(itemWeightPath.c_str(), "r");

  int i = 0;
  int m, n, k;
  m = numUsers;
  k = numFeatures;
  n = numItems + 1;

  float *allUsers = parseAllUsers(userWeights);
  printf("All user weights read in.\n");
  float *allItems = parseAllItems(itemWeights);
  printf("All item weights read in.\n");

  printf("All weights read in.\n");
  printf("m(users)=%d, n(items)=%d, k(features)=%d\n", m, n, k);
  mkl_free_buffers();

  unsigned long long call_start, call_stop, call_run_time = 0;
  //    call_start = _rdtsc2();

  //    float *allItemNorms = computeAllItemNorms(allItems);
  //    float *allUserNorms = computeAllUserNorms(allUsers);
  //
  //
  //    vsInv(numUsers*numFeatures, allUserNorms, allUserNorms);
  //    vsInv((numItems+1)*numFeatures, allItemNorms, allItemNorms);

  float alpha = 1.0;
  float beta = 0.0;

  //    for (i = 0; i < (numItems+1); i++) {
  //        vsMul(numFeatures, &allItems[i*numFeatures],
  // &allItemNorms[i*numFeatures], &allItems[i*numFeatures]);
  //    }
  //    for (i = 0; i < numUsers; i++) {
  //        vsMul(numFeatures, &allUsers[i*numFeatures],
  // &allUserNorms[i*numFeatures], &allUsers[i*numFeatures]);
  //    }

  //    call_stop = _rdtsc2();
  //    call_run_time = call_stop - call_start;
  //    printf("stage 1 run time: %llu\n", call_run_time);

  unsigned long tb = 1024 * 1024;
  tb = tb * 1024;
  tb = tb * 800;

  double time_st, time_end, time_avg1 = 0, time_avg2 = 0, time_avg3 = 0;

  unsigned long needed = 1;
  needed = needed * m;
  needed = needed * n;
  needed = needed * 4;

  if (needed > tb) {
    int per_instance = (int)floor((numUsers) / 3.0);
    unsigned long mem_needed = 1;
    mem_needed = mem_needed * per_instance;
    mem_needed = mem_needed * n;
    mem_needed = mem_needed * sizeof(float);
    printf("bytes needed: %lu\n", mem_needed);
    printf("first path.\n");
    float *cosineSimilarity = (float *)mkl_malloc(mem_needed, 64);
    if (cosineSimilarity == NULL) {
      printf(
          "\n ERROR: Can't allocate memory for cosineSimilarity. Aborting... "
          "\n\n");
      mkl_free(cosineSimilarity);
      exit(1);
    }

    //        call_start = _rdtsc2();

    //        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
    //                    (2), n, k, alpha, allUsers, k, allItems, k, beta,
    // cosineSimilarity, n);

    //        std::ofstream myfile;
    //        myfile.open ("naive_output.txt", std::ofstream::out |
    // std::ofstream::app);

    //        for (i = 0; i < 2; i++) {
    //            std::priority_queue<std::pair<float, int>,
    // std::vector<std::pair<float, int> >, std::greater<std::pair<float, int> >
    // > q;
    //
    //
    //
    //            for (int j = 0; j < (numItems+1); j++) {
    //                if (j < 100) {
    //                    q.push(std::make_pair(cosineSimilarity[(i*(numItems+1))+j],
    // j));
    //                    continue;
    //                }
    //                if (cosineSimilarity[(i*(numItems+1))+j] > q.top().first)
    // {
    //                    q.pop();
    //                    q.push(std::make_pair(cosineSimilarity[(i*(numItems+1))+j],
    // j));
    //                }
    //            }
    //
    //
    //            int topK[100];
    //
    //            for (int j = 0; j < 100; j++) {
    //                std::pair<float, int> p = q.top();
    //                topK[100-1-j] = p.second;
    //                q.pop();
    //            }
    //
    //            myfile << i << " : [";
    //            myfile << topK[0];
    //            for (int j = 1; j < k; j++) {
    //                myfile << ", " << topK[j];
    //            }
    //            myfile << "]\n";
    //
    //        }

    time_st = dsecnd();
    time_st = dsecnd();

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (per_instance), n, k,
                alpha, allUsers, k, allItems, k, beta, cosineSimilarity, n);

    time_end = dsecnd();
    time_avg1 += (time_end - time_st);

    time_st = dsecnd();
    time_st = dsecnd();
    computeTopK(cosineSimilarity, per_instance);
    time_end = dsecnd();
    time_avg2 += (time_end - time_st);

    time_st = dsecnd();
    time_st = dsecnd();
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (per_instance), n, k,
                alpha, &allUsers[per_instance], k, allItems, k, beta,
                cosineSimilarity, n);
    time_end = dsecnd();
    time_avg1 += (time_end - time_st);

    time_st = dsecnd();
    time_st = dsecnd();
    computeTopK(cosineSimilarity, per_instance);
    time_end = dsecnd();
    time_avg2 += (time_end - time_st);

    time_st = dsecnd();
    time_st = dsecnd();
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (per_instance), n, k,
                alpha, &allUsers[per_instance * 2], k, allItems, k, beta,
                cosineSimilarity, n);
    time_end = dsecnd();
    time_avg1 += (time_end - time_st);

    time_st = dsecnd();
    time_st = dsecnd();
    computeTopK(cosineSimilarity, per_instance);
    time_end = dsecnd();
    time_avg2 += (time_end - time_st);

    //        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
    //                    (per_instance), n, k, alpha,
    // &allUsers[per_instance*3], k, allItems, k, beta, cosineSimilarity, n);
    //
    //        computeTopK(cosineSimilarity, per_instance);

    //        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
    //                    (per_instance), n, k, alpha,
    // &allUsers[per_instance*4], k, allItems, k, beta, cosineSimilarity, n);
    //
    //        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
    //                    (per_instance), n, k, alpha,
    // &allUsers[per_instance*5], k, allItems, k, beta, cosineSimilarity, n);
    //
    //        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
    //                    (per_instance), n, k, alpha,
    // &allUsers[per_instance*6], k, allItems, k, beta, cosineSimilarity, n);
    //
    //        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
    //                    (per_instance), n, k, alpha,
    // &allUsers[per_instance*7], k, allItems, k, beta, cosineSimilarity, n);
    //
    //        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
    //                    (per_instance), n, k, alpha,
    // &allUsers[per_instance*8], k, allItems, k, beta, cosineSimilarity, n);
    //
    //        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
    //                    (per_instance), n, k, alpha,
    // &allUsers[per_instance*9], k, allItems, k, beta, cosineSimilarity, n);

    time_avg3 = time_avg1 + time_avg2;
    printf("gemm time: %f secs \n", time_avg1);
    printf("heap time: %f secs \n", time_avg2);
    printf("time taken: %f secs \n", time_avg3);

    //        call_stop = _rdtsc2();
    //        call_run_time = call_stop - call_start;
    //        printf("stage 2 run time: %llu\n", call_run_time);

    MKL_free(allItems);
    MKL_free(allUsers);
    //        MKL_free(allItemNorms);
    //        MKL_free(allUserNorms);
    MKL_free(cosineSimilarity);

  } else {

    //        float *cosineSimilarity = (float
    // *)mkl_malloc(sizeof(float)*(numUsers)*(numItems+1), 64);

    float *cosineSimilarity = (float *)mkl_malloc(needed, 64);
    if (cosineSimilarity == NULL) {
      printf(
          "\n ERROR: Can't allocate memory for cosineSimilarity. Aborting... "
          "\n\n");
      mkl_free(cosineSimilarity);
      //        cosineSimilarity = (float
      // *)malloc(sizeof(float)*numUsers*(numItems+1));
      //        if (cosineSimilarity == NULL) {
      //            printf( "\n ERROR: Can't allocate memory for
      // cosineSimilarity. Aborting... \n\n");
      //            free(cosineSimilarity);
      //            exit(1);
      //        }
      exit(1);
    }

    printf("second path.\n");

    unsigned long test = m - 1000;
    test = test * n;
    test = test + (n - 1000);

    cosineSimilarity[test] = 5.0;
    printf("%f\n", cosineSimilarity[test]);

    //        call_start = _rdtsc2();

    time_st = dsecnd();
    time_st = dsecnd();
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, alpha,
                allUsers, k, allItems, k, beta, cosineSimilarity, n);
    time_end = dsecnd();
    time_avg1 = (time_end - time_st);
    printf("gemm time: %f secs \n", time_avg1);

    time_st = dsecnd();
    time_st = dsecnd();
    computeTopK(cosineSimilarity, m);
    time_end = dsecnd();
    time_avg2 = (time_end - time_st);
    printf("heap time: %f secs \n", time_avg2);

    time_avg3 = time_avg1 + time_avg2;

    printf("total time: %f secs \n", (time_avg3));

    //        call_stop = _rdtsc2();
    //        call_run_time = call_stop - call_start;
    //        printf("stage 2 run time: %llu\n", call_run_time);

    //        std::ofstream myfile;
    //        myfile.open ("naive_output.txt", std::ofstream::out |
    // std::ofstream::app);
    //
    //        for (i = 0; i < (numUsers-1); i++) {
    //            std::priority_queue<std::pair<float, int>,
    // std::vector<std::pair<float, int> >, std::greater<std::pair<float, int> >
    // > q;
    //
    //
    //
    //            for (int j = 0; j < (numItems); j++) {
    //                if (j < 50) {
    //                    q.push(std::make_pair(cosineSimilarity[(i*(numItems+1))+j],
    // j));
    //                    continue;
    //                }
    //                if (cosineSimilarity[(i*(numItems+1))+j] > q.top().first)
    // {
    //                    q.pop();
    //                    q.push(std::make_pair(cosineSimilarity[(i*(numItems+1))+j],
    // j));
    //                }
    //            }
    //
    //
    //            int topK[50];
    //
    //            for (int j = 0; j < 50; j++) {
    //                std::pair<float, int> p = q.top();
    //                topK[100-1-j] = p.second;
    //                q.pop();
    //            }
    //
    ////            myfile << i << " : [";
    ////            myfile << topK[0];
    ////            for (int j = 1; j < k; j++) {
    ////                myfile << ", " << topK[j];
    ////            }
    ////            myfile << "]\n";
    ////
    ////            printf("user: %d\n", i);
    //
    //        }
    //

    MKL_free(allItems);
    MKL_free(allUsers);
    //        MKL_free(allItemNorms);
    //        MKL_free(allUserNorms);
    MKL_free(cosineSimilarity);
  }

  std::ofstream myfile;
  std::string outputFile;
  outputFile = "blocked_u" + std::to_string(numUsers) + "_f" +
               std::to_string(numFeatures) + "_k" + std::to_string(topk) +
               ".txt";
  myfile.open(outputFile, std::ofstream::out | std::ofstream::app);
  myfile << "Users: " << numUsers << " Items: " << numItems
         << " Features: " << numFeatures << " K: " << topk << "\t"
         << "GEMM Time: " << time_avg1 << " Heap Time: " << time_avg2
         << " Total Time: " << time_avg3 << "\n";
  myfile.close();

  return 0;
}