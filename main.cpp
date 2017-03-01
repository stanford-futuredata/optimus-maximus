//
//  main.cpp
//  fomo_preproc
//
//  Created by Geet Sethi on 10/24/16.
//  Copyright © 2016 Geet Sethi. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/stat.h>
#include <mkl.h>
#include <mkl_scalapack.h>
#include <functional>
#include <utility>
#include <queue>
#include <algorithm>
#include <numeric>
#include "mz_special.hpp"
#include "parser.hpp"
#include "arith.hpp"
#include <unordered_map>
#include <string>
#include <iostream>
#include <fstream>
#include <math.h>

extern int numberUsers;
extern int numFeatures;
extern int numItems;
extern int numClusters;

double creationTime;
double sortTime;
double computeKTime;

typedef unsigned long long u64;
#define DECLARE_ARGS(val, low, high)    unsigned low, high
#define EAX_EDX_VAL(val, low, high)     ((low) | ((u64)(high) << 32))
#define EAX_EDX_ARGS(val, low, high)    "a" (low), "d" (high)
#define EAX_EDX_RET(val, low, high)     "=a" (low), "=d" (high)
static inline unsigned long long
_rdtsc2(void)
{
    DECLARE_ARGS(val, low, high);
    
    asm volatile("rdtsc" : EAX_EDX_RET(val, low, high));
    
    return EAX_EDX_VAL(val, low, high);
}


int file_exist (const char *path){
    struct stat buffer;
    return (stat(path, &buffer) == 0);
}


int main(int argc, const char * argv[]) {
    
    // insert code here...
    
    if (argc != 10) { //MAKE SURE TO FIX FOR USER TO CLUSTER
        printf("Usage: fomo <numItems> <numLatentFactors> <numClusters> <numBins> <K> <numClusters to Time> <Weight Directory> <Cluster Directory> <MaxThreads>\n");
        exit(1);
    }
    
//    for (int i = 6; i < 11; i++) {
//        if (!file_exist(argv[i])) {
//            printf("Could not open %s. Perhaps incorrect filename?\n", argv[i]);
//            exit(1);
//        }
//    }
    
    std::vector<std::string> allArgs(argv, argv + argc);
    std::string userWeightPath;
    std::string itemWeightPath;
    std::string clusterUserCountsPath;
    std::string clustersToUsersPath;
    std::string centroidsPath;
    std::string userToClustersPath;
    
    int numberItems = atoi(argv[1]);
    int numberFeatures = atoi(argv[2]);
    int numberClusters = atoi(argv[3]);
    int maxThreads = atoi(argv[9]);
    
    MKL_Set_Num_Threads(maxThreads);
    
    
    if (((char)allArgs[7].back()) == '/') {
        userWeightPath = allArgs[7] + "userWeights.txt";
        itemWeightPath = allArgs[7] + "itemWeights.txt";

    }
    else{
        userWeightPath = allArgs[7] + "/userWeights.txt";
        itemWeightPath = allArgs[7] + "/itemWeights.txt";

    }
    
    if (((char)allArgs[8].back()) == '/') {
        clusterUserCountsPath = allArgs[8] + allArgs[3] + "_user_cluster_ids_counts";
        clustersToUsersPath = allArgs[8] + allArgs[3] + "_user_cluster_ids_inverted";
        centroidsPath = allArgs[8] + allArgs[3] + "_user_clusters";
        userToClustersPath = allArgs[8] + allArgs[3] + "_user_cluster_ids";
    }
    else{
        clusterUserCountsPath = allArgs[8] + "/" + allArgs[3] + "_user_cluster_ids_counts";
        clustersToUsersPath = allArgs[8] + "/" + allArgs[3] + "_user_cluster_ids_inverted";
        centroidsPath = allArgs[8] + "/" + allArgs[3] + "_user_clusters";
        userToClustersPath = allArgs[8] + "/" + allArgs[3] + "_user_cluster_ids";
        
    }
    
    
    int numberBins = atoi(argv[4]);
    int k = atoi(argv[5]);
    
    int numClustersToTime = atoi(argv[6]);
    
    setNumFeatures(numberFeatures);
    setNumItems(numberItems);
    setNumClusters(numberClusters);
    
    FILE *userWeights;
    FILE *itemWeights;
    FILE *clusterUserCounts;
    FILE *clustersToUsers;
    FILE *centroids;
    
    userWeights = fopen(userWeightPath.c_str(), "r");
    itemWeights = fopen(itemWeightPath.c_str(), "r");
    clusterUserCounts = fopen(clusterUserCountsPath.c_str(), "r");
    clustersToUsers = fopen(clustersToUsersPath.c_str(), "r");
    centroids = fopen(centroidsPath.c_str(), "r");
    
    FILE *userToClusters;
    userToClusters = fopen(userToClustersPath.c_str(), "r");
    std::unordered_map<int, int> userClusterMap = userToClusterMap(userToClusters);
    
    double time_start, time_end, time_diff1, time_diff2, time_diff3;
    
    time_start = dsecnd();
    time_start = dsecnd();
    fullCluster_t *allClusters = parseClusers(clustersToUsers, clusterUserCounts);
    float *allCentroids = parseClusterCentroids(centroids);
    time_end = dsecnd();
    time_diff3 = (time_end - time_start);

    
    time_start = dsecnd();
    time_start = dsecnd();

    float *allUsers = parseAllUsers(userWeights, allClusters, userClusterMap);
    time_end = dsecnd();
    time_diff1 = (time_end - time_start);
    
    
    float *allItems = parseAllItems(itemWeights);
 //   float *allItemNorms = computeAllItemNorms(allItems);
    
    allUsers = reorderUsersInClusterOrder(allUsers, allClusters);
    
    unsigned long long call_start, call_stop, call_run_time = 0;
    call_start = _rdtsc2();
    
    float *cosineSimilarityItemCentroid = computeCosineSimilarityItemsCentroids(allItems, allCentroids);
    float *arcCosineSimilarityItemCentroid = (float *)mkl_malloc(sizeof(float)*numberClusters*(numberItems+1), 64);
    if (arcCosineSimilarityItemCentroid == NULL) {
        printf( "\n ERROR: Can't allocate memory for arcCosineSimilarityItemCentroid. Aborting... \n\n");
        mkl_free(arcCosineSimilarityItemCentroid);
        exit(1);
    }

    uint half = ceilf(numberClusters/2);
    uint len = numberClusters*(numberItems + 1);
    uint first = half*(numberItems+1);
    uint remaining = len - first;
    //printf("here %ud %d %d\n", len, first, remaining);
    vsAcos(first, cosineSimilarityItemCentroid, arcCosineSimilarityItemCentroid);
    vsAcos(remaining, &cosineSimilarityItemCentroid[first], &arcCosineSimilarityItemCentroid[first]);
    //printf("here\n");
    MKL_free(cosineSimilarityItemCentroid);
    
    float *allItemNorms = computeAllItemNorms(allItems);
    
    call_stop = _rdtsc2();
    call_run_time = call_stop - call_start;
//    printf("pre_cluster_prep run time: %llu\n", call_run_time);
    
//    int clustersToVisit[10] = {3017, 2981, 346, 1996, 3745, 675, 937, 3516, 1958, 1497};
    
    std::ofstream misc_file;
    std::string misc_out_file;
    misc_out_file = "output_U-" + std::to_string(numberUsers) + "_F-" + std::to_string(numberFeatures) + "_C-" + std::to_string(numberClusters) + "_B-" + std::to_string(numberBins) + ".csv";
    misc_file.open (misc_out_file, std::ofstream::out | std::ofstream::app);
    
    misc_file << "K,Cluster,User,ItemsVisited,ThetaUC" << std::endl;
    
    std::ofstream run_file;
    std::string run_out_file;
    run_out_file = "runtime_U-" + std::to_string(numberUsers) + "_F-" + std::to_string(numberFeatures) + "_C-" + std::to_string(numberClusters) + "_B-" + std::to_string(numberBins) + ".txt";
//    run_file.open (run_out_file, std::ofstream::out | std::ofstream::app);
//    
//    run_file << "PreComp Time:" << call_run_time << "\n";
//    run_file << "Cluster,K,UpperBoundCreation,SortUpperBound,TopK,FreqMap\n";
    
    call_start = _rdtsc2();
    
    creationTime = 0;
    sortTime = 0;
    computeKTime = 0;

    
    time_start = dsecnd();
    time_start = dsecnd();
    for (int i = 0; i < numberClusters; i++) {
        clusterPrep(allClusters, i, allUsers, allCentroids, allItems, allItemNorms, numberBins, k, &arcCosineSimilarityItemCentroid[i*(numberItems+1)], misc_file, run_file);
    }
    time_end = dsecnd();
    time_diff2 = (time_end - time_start);
    
    call_stop = _rdtsc2();
    call_run_time = call_stop - call_start;
    //printf("total run time: %llu\n", call_run_time);
//    run_file << "Total Run Time:" << call_run_time << "\n";
    printf("comp time: %f secs \n", time_diff2);
    
    std::ofstream myfile;
    std::string outputFile;
    outputFile = "simdex2_u" + std::to_string(numberUsers) + "_f" + std::to_string(numFeatures) + "_c" + std::to_string(numClusters) + "_k" + std::to_string(k) + ".csv";
    myfile.open (outputFile, std::ofstream::out | std::ofstream::app);
    myfile << "Users,Clusters,Features,K,NumBins,ClusterLoadTime,UserLoadTime,CompTime,UpperBoundCreationTime,UpperBoundSortTime,WalkTime" << std::endl;
    myfile << numberUsers << "," << numClusters << "," << numFeatures << "," << k << "," << numberBins << "," << time_diff3 << "," << time_diff1 << "," << time_diff2 << "," << creationTime << "," << sortTime << "," << computeKTime << std::endl;
    
//    myfile << "Users: " << numberUsers << " Clusters: " << numClusters << " Features: " << numFeatures << " K: " << k << " NumBins: " << numberBins << "  " << " Cluster Load Time: " << time_diff3 << " User Load Time: " << time_diff1 << " Comp Time: " << time_diff2 << " UpperBoundCreationTime: " << creationTime << " UpperBoundSortTime: " << sortTime << " WalkTime: " << computeKTime << "\n";
    myfile.close();

    
    return 0;
    
}

















