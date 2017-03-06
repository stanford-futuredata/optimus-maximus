//
//  mz_special.cpp
//  fomo_preproc
//
//  Created by Geet Sethi on 10/27/16.
//  Copyright © 2016 Geet Sethi. All rights reserved.
//

#include "mz_special.hpp"
#include "arith.hpp"
#include "parser.hpp"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <mkl.h>
#include <functional>
#include <utility>
#include <queue>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <iostream>
#include <fstream>
#include <math.h>
#include <ipps.h>

extern int numberUsers;
extern int numFeatures;
extern int numItems;

extern double creationTime;
extern double sortTime;
extern double computeKTime;

//TEST PERFORMANCE OF VS VS VMS MKL


typedef unsigned long long u64;
#define DECLARE_ARGS(val, low, high)    unsigned low, high
#define EAX_EDX_VAL(val, low, high)     ((low) | ((u64)(high) << 32))
#define EAX_EDX_ARGS(val, low, high)    "a" (low), "d" (high)
#define EAX_EDX_RET(val, low, high)     "=a" (low), "=d" (high)

#define DATA_TYPE int64_t
struct negrightshift {
    inline int64_t operator()(const int64_t &x, const unsigned offset) {
        return -(x >> offset);
    }
};

static inline unsigned long long
_rdtsc3(void)
{
    DECLARE_ARGS(val, low, high);
    
    asm volatile("rdtsc" : EAX_EDX_RET(val, low, high));
    
    return EAX_EDX_VAL(val, low, high);
}



int *indexPermutation(float *upperBounds){
    int *indexes = (int *)malloc(sizeof(int)*(numItems+1));
    
    std::iota(indexes, indexes+(numItems+1), 0);
    std::sort(indexes, indexes+(numItems+1), [&](size_t i1, size_t i2) {return upperBounds[i1] > upperBounds[i2];});
    
    return indexes;
}

float *applyPermutation(float *upperBounds, int *permutation){
    float *sortedUpperBounds = (float *)malloc(sizeof(float)*(numItems+1));
    std::transform(permutation, permutation+(numItems+1), sortedUpperBounds, [&](size_t i){ return upperBounds[i]; });
    return sortedUpperBounds;
}


bool compareByBound(const upperBoundItem_t &a, const upperBoundItem_t &b)
{
    return a.upperBound > b.upperBound;
}

void clusterPrep(fullCluster_t *fullClusterData, int clusterID, float *fullUserData, float *clusterCentroids, float *allItemWeights, float *allItemNorms, int numBins, int k, float *vectorized_Acos_output_ic, std::ofstream& misc_file, std::ofstream& run_file){
    
    int i = 0;
    int j = 0;
//    unsigned long long call_start, call_stop, call_run_time = 0;
//    
//    unsigned long long upperBoundCreation_time, sortUpperBound_time, computeTopK_time, makeFreqUnion_time = 0;
    
    double time_start, time_end, upperBoundCreation_time, sortUpperBound_time, computeTopK_time;


    
    int clusterOffset = fullClusterData->clusterOffset[clusterID];
    if (fullClusterData->clusterArray[clusterOffset] != clusterID) {
        printf("In Cluster Prep: Cluster Array Cluster ID NOT EQUAL to input Cluster ID. Exiting.");
        exit(1);
    }
    
    int numUsersInCluster = fullClusterData->clusterArray[clusterOffset+1];
    
    
    //userNormTuple_t *userNormTuple_array = (userNormTuple_t *)malloc(sizeof(struct userNormTuple) * numUsersInCluster);
    //switched to stack allocation-BACK TO HEAP
    //userNormTuple_t userNormTuple_array[numUsersInCluster];
    
    
    userNormTuple_t *userNormTuple_array = (userNormTuple_t *)mkl_malloc(sizeof(struct userNormTuple) * numUsersInCluster, 64);
    if (userNormTuple_array == NULL) {
        printf( "\n ERROR: Can't allocate memory for userNormTuple_array. Aborting... \n\n");
        mkl_free(userNormTuple_array);
        exit(1);
    }
    
    for (i = 0; i < numUsersInCluster; i++) {
        userNormTuple_array[i].userID = fullClusterData->clusterArray[clusterOffset+2+i];
//        userNormTuple_array[i].userNorm = cblas_snrm2(numFeatures, &fullUserData[userNormTuple_array[i].userID*numFeatures], 1);
        // computed when doing userClusterSimilarity
    }
    
// START TIMING FOR UPPERBOUND COMPUTATION
//
//    call_start = _rdtsc3();
    
//        float *vectorized_Acos_input_uc2 = computeCosineSimilarityUserCluster(&fullUserData[userNormTuple_array[0].userID*numFeatures], &clusterCentroids[clusterID*numFeatures], numUsersInCluster, userNormTuple_array);
//    
//    float centroidNorm = cblas_snrm2(numFeatures, &clusterCentroids[clusterID*numFeatures], 1);
////    
////    //float vectorized_Acos_input_uc[numUsersInCluster];
//    float *vectorized_Acos_input_uc = (float *)malloc(sizeof(float)*numUsersInCluster);
//    //this is computing the 'x' in acos(x) for all users in the clusters - stored in the same order as the userNormTuple
//    for (i = 0; i < numUsersInCluster; i++) {
//        vectorized_Acos_input_uc[i] = cblas_sdot(numFeatures, &clusterCentroids[clusterID*numFeatures], 1, &fullUserData[userNormTuple_array[i].userID*numFeatures], 1) / (centroidNorm*userNormTuple_array[i].userNorm);
////        if (vectorized_Acos_input_uc[i] > 1) {
////            vectorized_Acos_input_uc[i] = 1.0;
////        }
//    }
    
    //float vectorized_Acos_output_uc[numUsersInCluster];
    //float *vectorized_Acos_output_uc = (float *)malloc(sizeof(float)*numUsersInCluster);
    
    time_start = dsecnd();
    time_start = dsecnd();

    
    float *vectorized_Acos_output_uc = (float *)mkl_malloc(sizeof(float)*numUsersInCluster, 64);
    if (vectorized_Acos_output_uc == NULL) {
        printf( "\n ERROR: Can't allocate memory for usersDOTcentroid. Aborting... \n\n");
        mkl_free(vectorized_Acos_output_uc);
        exit(1);
    }
    
    float *vectorized_Acos_input_uc = computeCosineSimilarityUserCluster(&fullUserData[userNormTuple_array[0].userID*numFeatures], &clusterCentroids[clusterID*numFeatures], numUsersInCluster, userNormTuple_array);
    
//    for (int r = 0; r < numUsersInCluster; r++) {
//        printf("%d: %f \t %f\n", r, vectorized_Acos_input_uc[r], vectorized_Acos_input_uc[r]);
//    }
//    
//    if (clusterID == 3) {
//        exit(1);
//    }

    
    vsAcos(numUsersInCluster, vectorized_Acos_input_uc, vectorized_Acos_output_uc);
    //all acos -- ie theta_uc -- stored in output_array
    float theta_max = vectorized_Acos_output_uc[cblas_isamax(numUsersInCluster, vectorized_Acos_output_uc, 1)];
    if (isnan(theta_max) != 0) {
        theta_max = 0;
        numBins = 1;
        printf("NaN detected.\n");
    }
//    printf("Cluster ID: %d \t theta_max: %f\n", clusterID, theta_max);
    
    
// ----------Theta Bin Creation Below------------------
    
    //use vectorized sort
    //float theta_bins[numBins];
    float *theta_bins = (float *)malloc(sizeof(float)*numBins);
    float theta_bin_step = theta_max / (float)numBins;
    for (i = 1; i < (numBins+1); i++) {
        theta_bins[i-1] = theta_bin_step*i;
    }
    
    //need to compute theta_ic's
    //float vectorized_Acos_input_ic[numItems+1];
    
//    float *vectorized_Acos_input_ic = (float *)malloc(sizeof(float)*(numItems+1));
//    vectorized_Acos_input_ic[0] = 0.0;
//    //this is computing the 'x' in acos(x) for all items - stored in the order
//    for (i = 0; i < (numItems+1); i++) {
//        vectorized_Acos_input_ic[i] = cblas_sdot(numFeatures, &clusterCentroids[clusterID*numFeatures], 1, &allItemWeights[i*numFeatures], 1) / (centroidNorm*allItemNorms[i]);
//    }
//    //float vectorized_Acos_output_ic[numItems+1];
//    float *vectorized_Acos_output_ic = (float *)malloc(sizeof(float)*(numItems+1));
//
//    
//    vsAcos((numItems+1), vectorized_Acos_input_ic, vectorized_Acos_output_ic);
    
//    for (i = 0; i < numItems+1; i++) {
//        printf("ClusterID: %d \t ItemID: %d \t theta_ic: %f\n", clusterID, i, vectorized_Acos_output_ic[i]);
//    }
//    return;
    
    //float const_vector_theta[numItems+1]; //will be the const theta_uc in each bucket as well as intermediary theta_ic - theta_uc, and then cos(theta_ic-theta_uc)
    //float *const_vector_theta = (float *)malloc(sizeof(float)*(numItems+1));
    float *const_vector_theta = (float *)mkl_malloc(sizeof(float)*(numItems+1), 64);
    if (const_vector_theta == NULL) {
        printf( "\n ERROR: Can't allocate memory for const_vector_theta. Aborting... \n\n");
        mkl_free(const_vector_theta);
        exit(1);
    }
    
    
    const_vector_theta[0] = 0.0;
    
    
    //float *upperBounds = (float *)malloc(sizeof(float)*numBins*(numItems+1));
    //float upperBounds[numBins][(numItems+1)];
    
    //float **upperBounds = (float **)malloc(numBins * sizeof(float *));
    float **upperBounds = (float **)mkl_malloc(sizeof(float*)*numBins, 64);
    if (upperBounds == NULL) {
        printf( "\n ERROR: Can't allocate memory for upperBounds. Aborting... \n\n");
        mkl_free(upperBounds);
        exit(1);
    }
    
    
    for (i = 0; i < numBins; i++){
        upperBounds[i] = (float *)mkl_malloc((numItems+1) * sizeof(float), 64);
        if (upperBounds[i] == NULL) {
            printf( "\n ERROR: Can't allocate memory for upperBounds[i]. Aborting... \n\n");
            mkl_free(upperBounds[i]);
            exit(1);
        }
    }
    
    
    for (i = 0; i < numBins; i++) {
        upperBounds[i][0] = 0.0;
    }
    
    for (i = 0; i < numBins; i++) {
        for (j = 0; j < (numItems+1); j++) {
            const_vector_theta[j] = theta_bins[i];
        }
        vsSub((numItems+1), vectorized_Acos_output_ic, const_vector_theta, const_vector_theta);
        vsCos((numItems+1), const_vector_theta, const_vector_theta);
        vsMul((numItems+1), allItemNorms, const_vector_theta, upperBounds[i]);
    }
    
    time_end = dsecnd();
    upperBoundCreation_time = (time_end - time_start);
    
//    call_stop = _rdtsc3();
//    call_run_time = call_stop - call_start;
////    printf("Cluster: %d \t time to create upperbounds: %llu\n", clusterID, call_run_time);
//    upperBoundCreation_time = call_run_time;

//  END OF UPPER BOUND COMPUTATION
//
    //sort upperbounds
    //still deciding best way to sort...
    //scalapack not working ....
    //will have two arrays - one of sorted upperbounds -- and one of corresponding index (i.e. itemID)
    
//    call_start = _rdtsc3();
    
    time_start = dsecnd();
    time_start = dsecnd();
    
    upperBoundItem_t **sortedUpperBounds = (upperBoundItem_t **)mkl_malloc(numBins * sizeof(upperBoundItem_t *),64);
    if (sortedUpperBounds == NULL) {
        printf( "\n ERROR: Can't allocate memory for newUpperBounds. Aborting... \n\n");
        mkl_free(sortedUpperBounds);
        exit(1);
    }
    for (i = 0; i < numBins; i++){
        sortedUpperBounds[i] = (upperBoundItem_t *)mkl_malloc((numItems+1) * sizeof(struct upperBoundItem), 64);
        if (sortedUpperBounds[i] == NULL) {
            printf( "\n ERROR: Can't allocate memory for newUpperBounds[i]. Aborting... \n\n");
            mkl_free(sortedUpperBounds[i]);
            exit(1);
        }
        for (j = 0; j < (numItems+1); j++) {
            sortedUpperBounds[i][j].upperBound = upperBounds[i][j];
            sortedUpperBounds[i][j].itemID = j;
        }
    }
    IppSizeL *pBufSize = (IppSizeL*)malloc(sizeof(IppSizeL));
    ippsSortRadixGetBufferSize_L((numItems+1), ipp64s, pBufSize);
    Ipp8u *pBuffer = (Ipp8u *)malloc(*pBufSize*sizeof(Ipp8u));
    //Ipp64s **upperBound64 = (Ipp64s **)newUpperBounds;
    for (i = 0; i < numBins; i++) {
        ippsSortRadixDescend_64s_I_L((Ipp64s*)sortedUpperBounds[i], (numItems+1), pBuffer);
    }
    
    time_end = dsecnd();
    sortUpperBound_time = (time_end - time_start);
    
//    for (i = 0; i < numItems+1; i++) {
//        printf("%f\n",sortedUpperBounds[0][i].upperBound);
//    }
//    exit(1);
    
    
//    float **NewupperBounds = (float **)mkl_malloc(sizeof(float*)*numBins, 64);
//    if (NewupperBounds == NULL) {
//        printf( "\n ERROR: Can't allocate memory for NewupperBounds. Aborting... \n\n");
//        mkl_free(NewupperBounds);
//        exit(1);
//    }
//    
//    
//    for (i = 0; i < numBins; i++){
//        NewupperBounds[i] = (float *)mkl_malloc((numItems+1) * sizeof(float), 64);
//        if (NewupperBounds[i] == NULL) {
//            printf( "\n ERROR: Can't allocate memory for NewupperBounds[i]. Aborting... \n\n");
//            mkl_free(NewupperBounds[i]);
//            exit(1);
//        }
//    }
//    
//    int *indexes = (int *)malloc(sizeof(int)*(numItems+1));
//    for (i = 0; i < numBins; i++) {
//        ippsSortIndexDescend_32f_I((Ipp32f*) upperBounds[i], indexes, (numItems+1));
//        for (j = 0; j < (numItems+1); j++) {
//            NewupperBounds[i][j] = upperBounds[i][indexes[j]];
//        }
//    }
    
//    call_stop = _rdtsc3();
//    call_run_time = call_stop - call_start;
//    sortUpperBound_time = call_run_time;
//    printf("Cluster: %d \t time to sort upperbounds: %llu\n", clusterID, sortUpperBound_time);
//    
////
//    call_start = _rdtsc3();
//    
//    //int *indexPermutations[numBins];
//    int **indexPermutations = (int **)malloc(sizeof(int *)*numBins);
//    //float *sortedUpperBounds[numBins];
//    float **sortedUpperBounds = (float **)malloc(sizeof(float *)*numBins);
//    //need to free at the end of function call
//    for (i = 0; i < numBins; i++) {
//        indexPermutations[i] = indexPermutation(upperBounds[i]);
//        sortedUpperBounds[i] = applyPermutation(upperBounds[i], indexPermutations[i]);
//    }
//    
//    call_stop = _rdtsc3();
//    call_run_time = call_stop - call_start;
////    printf("Cluster: %d \t time to sort upperbounds: %llu\n", clusterID, call_run_time);
//    sortUpperBound_time = call_run_time;
    
//    call_start = _rdtsc3();
//    upperBoundItem_t **sortedUpperBounds = (upperBoundItem_t **)malloc(numBins * sizeof(upperBoundItem_t *));
//    //printf("%lu\n", sizeof(struct upperBoundItem));
//    if (sortedUpperBounds == NULL) {
//        printf( "\n ERROR: Can't allocate memory for newUpperBounds. Aborting... \n\n");
//        free(sortedUpperBounds);
//        exit(1);
//    }
//    for (i = 0; i < numBins; i++){
//        sortedUpperBounds[i] = (upperBoundItem_t *)malloc((numItems+1) * sizeof(struct upperBoundItem));
//        if (sortedUpperBounds[i] == NULL) {
//            printf( "\n ERROR: Can't allocate memory for newUpperBounds[i]. Aborting... \n\n");
//            free(sortedUpperBounds[i]);
//            exit(1);
//        }
//        for (j = 0; j < (numItems+1); j++) {
//            sortedUpperBounds[i][j].upperBound = upperBounds[i][j];
//            sortedUpperBounds[i][j].itemID = j;
//        }
//    }
//    
//    //    int64_t **toSort = (int64_t**)sortedUpperBounds;
//    for (i = 0; i < numBins; i++) {
//        std::sort(sortedUpperBounds[i], sortedUpperBounds[i]+(numItems+1), compareByBound);
//        
//    }
//    call_stop = _rdtsc3();
//    call_run_time = call_stop - call_start;
//    sortUpperBound_time = call_run_time;

    
// ----------Computer Per User TopK Below------------------
    
    int currentUser = 0;
    int bucket_index = 0; //will crash if not assigned later
    //int allTopK[numUsersInCluster][k];
    
//    int **allTopK = (int **)malloc(numUsersInCluster * sizeof(int *));
//    for (i = 0; i < numUsersInCluster; i++){
//        allTopK[i] = (int *)malloc((k) * sizeof(int));
//    }
    
//    float *scoreA = (float *)mkl_malloc(sizeof(float)*1000, 64);
//    if (scoreA == NULL) {
//        printf( "\n ERROR: Can't allocate memory for scoreA. Aborting... \n\n");
//        mkl_free(scoreA);
//        exit(1);
//    }
//    
//    int m, n;
//    m = 1000;
//    n = numFeatures;
//    float alpha = 1.0;
//    float beta = 0.0;
//    int stride = 1;
    
//    int kList[4] = {1,5,10,50};
    int **allTopK;
//
//    for (int r = 0; r < 4; r++) {
//        k = kList[r];

        allTopK = (int **)mkl_malloc(numUsersInCluster * sizeof(int *),64);
        for (i = 0; i < numUsersInCluster; i++){
            allTopK[i] = (int *)malloc((k) * sizeof(int));
        }
    
//        call_start = _rdtsc3();
    
    time_start = dsecnd();
    time_start = dsecnd();
    
        for (i = 0; i < numUsersInCluster; i++) {
            currentUser = userNormTuple_array[i].userID;
            //find user's bin
            for (j = 0; j < numBins; j++) {
                if (vectorized_Acos_output_uc[i] <= theta_bins[j]) {
                    bucket_index = j;
                    break;
                }
            }
            //found bin
            
            
            std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int> >, std::greater<std::pair<float, int> > > q;
            
    //        cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, alpha, userNorms, n, &fullUserData[currentUser*numFeatures], stride, beta, usersDOTcentroid, stride);

            
            float score = 0.0;
            int itemID = 0;
            
            for (j = 0; j < k+1; j++) { //add 1 because first item is junk
                itemID = sortedUpperBounds[bucket_index][j].itemID;
                score = cblas_sdot(numFeatures, &allItemWeights[itemID*numFeatures], 1, &fullUserData[currentUser*numFeatures], 1);
    //            float true_ui = acosf(score / (userNormTuple_array[i].userNorm*allItemNorms[itemID]));
                q.push(std::make_pair(score, itemID));
    //            printf("ClusterID: %d \t UserID: %d \t ItemID: %d \t theta_ic: %f \t theta_uc: %f \t ic-uc: %f \t ui: %f\n", clusterID, currentUser, itemID, vectorized_Acos_output_ic[itemID], vectorized_Acos_output_uc[i], vectorized_Acos_output_ic[itemID] - vectorized_Acos_output_uc[i], true_ui);
            }
            int itemsVisited = k;
            
            for (j = k+1; j < (numItems+1); j++) {
                if (q.top().first > (userNormTuple_array[i].userNorm*sortedUpperBounds[bucket_index][j].upperBound)) {
                    break;
                }
                itemID = sortedUpperBounds[bucket_index][j].itemID;
                score = cblas_sdot(numFeatures, &allItemWeights[itemID*numFeatures], 1, &fullUserData[currentUser*numFeatures], 1);
    //            float true_ui = acosf(score / (userNormTuple_array[i].userNorm*allItemNorms[itemID]));
    //            printf("ClusterID: %d \t UserID: %d \t ItemID: %d \t theta_ic: %f \t theta_uc: %f \t ic-uc: %f \t ui: %f\n", clusterID, currentUser, itemID, vectorized_Acos_output_ic[itemID], vectorized_Acos_output_uc[i], vectorized_Acos_output_ic[itemID] - vectorized_Acos_output_uc[i], true_ui);
                itemsVisited++;
                if (q.top().first < score) {
                    q.pop();
                    q.push(std::make_pair(score, itemID));
                }
            }
            
            for (j = 0; j < k; j++) {
                std::pair<float, int> p = q.top();
                //dont need to store score
                allTopK[i][k-1-j] = p.second; //store itemID
                q.pop();
            }
            
    //        printf("Cluster: %d \t UserID: %d \t ItemsVisited: %d\n", clusterID, currentUser, itemsVisited);
    //        printf("%d,%d,%d,%f\n", clusterID, currentUser, itemsVisited, vectorized_Acos_output_uc[i]);
            
//            misc_file << k << "," << clusterID << "," << currentUser << "," << itemsVisited << "," << vectorized_Acos_output_uc[i] << "\n";

            
    //        if (clusterID == 2) {
    //            printf("Cluster: %d \t UserID: %d \t ItemsVisited: %d \t theta_uc: %f\n", clusterID, currentUser, itemsVisited, vectorized_Acos_output_uc[i]);
    //        }
    //        
    //        if (clusterID == 7) {
    //            printf("Cluster: %d \t UserID: %d \t ItemsVisited: %d\n", clusterID, currentUser, itemsVisited);
    //        }
    //        if (clusterID == 8) {
    //            printf("Cluster: %d \t UserID: %d \t ItemsVisited: %d\n", clusterID, currentUser, itemsVisited);
    //
    //        }
            
        }
    
    time_end = dsecnd();
    computeTopK_time = (time_end - time_start);
    
//        call_stop = _rdtsc3();
//        call_run_time = call_stop - call_start;
//        //printf("Cluster: %d \t time to compute all TopKs [K=%d]: %llu\n", clusterID, k, call_run_time);
//        computeTopK_time = call_run_time;
    
        
//        call_start = _rdtsc3();
    
/////----------------------freq map code removed
    
//        std::unordered_map<int, int> freqMap;
//        
//        for (i = 0; i < numUsersInCluster; i++) {
//            //        myfile << "Top 100 items for User " << userNormTuple_array[i].userID << ": [";
//            for (j = 0; j < k; j++) {
//                //            if (j == 0) {
//                //                myfile << allTopK[i][j];
//                //            }
//                //            else
//                //                myfile << "," << allTopK[i][j];
//                auto search = freqMap.find(allTopK[i][j]);
//                if (search == freqMap.end()) {
//                    freqMap.insert({allTopK[i][j], 1});
//                }
//                else
//                    search->second = search->second + 1;
//            }
//            //        myfile << "]\n";
//        }
//        
//        
//        
//        
//        //printf("size of map: %lu\n", freqMap.size());
//        
//        
//        std::vector<std::pair<int, int>> tuples(freqMap.begin(), freqMap.end());
//        std::sort(tuples.begin(), tuples.end(), [](const std::pair<int,int> &left, const std::pair<int,int> &right) {
//            return left.second > right.second;
//        });
//        
//        call_stop = _rdtsc3();
//        call_run_time = call_stop - call_start;
//        //printf("Cluster: %d \t time to compute union sorted by freq [K=%d]: %llu\n", clusterID, k, call_run_time);
//        makeFreqUnion_time = call_run_time;
//        
////        run_file << clusterID << "," << k << "," << upperBoundCreation_time << "," << sortUpperBound_time << "," << computeTopK_time << "," << makeFreqUnion_time << "\n";
//    
        for (i = 0; i < numUsersInCluster; i++){
            free(allTopK[i]);
        }
        
        MKL_free(allTopK);
        
//    }
    
// ----------All TopK computed, union Below---POTENTIALLY TOO SLOW------------------
    
//    std::vector<int> v(numUsersInCluster*k);
//    std::vector<int>::iterator it;
//    
//    std::sort (allTopK[0],allTopK[0]+k);
//    std::sort (allTopK[1],allTopK[1]+k);
//    
//    it=std::set_union (allTopK[0],allTopK[0]+k, allTopK[1],allTopK[1]+k, v.begin());
//    v.resize(it-v.begin());
//    
//    for (i = 2; i < numUsersInCluster; i++) {
//        std::sort (allTopK[i],allTopK[i]+k);
//        it=std::set_union (v.begin(), v.end(), allTopK[i],allTopK[i]+k, v.begin());
//        v.resize(it-v.begin());
//    }
//    //v is the union set. i feel like this will be pretty inefficient.
    
// ----------All TopK computed, Switched Freq Map Below------------------
    
//    std::ofstream myfile;
//    std::string outputFile;
//    outputFile = "output_" + std::to_string(numberUsers) + ".txt";
//    myfile.open (outputFile, std::ofstream::out | std::ofstream::app);
    
    
//    call_start = _rdtsc3();
//    
//    std::unordered_map<int, int> freqMap;
//    
//    for (i = 0; i < numUsersInCluster; i++) {
////        myfile << "Top 100 items for User " << userNormTuple_array[i].userID << ": [";
//        for (j = 0; j < k; j++) {
////            if (j == 0) {
////                myfile << allTopK[i][j];
////            }
////            else
////                myfile << "," << allTopK[i][j];
//            auto search = freqMap.find(allTopK[i][j]);
//            if (search == freqMap.end()) {
//                freqMap.insert({allTopK[i][j], 1});
//            }
//            else
//                search->second = search->second + 1;
//        }
////        myfile << "]\n";
//    }
//
//    
//
//        
//    //printf("size of map: %lu\n", freqMap.size());
//
//    
//    std::vector<std::pair<int, int>> tuples(freqMap.begin(), freqMap.end());
//    std::sort(tuples.begin(), tuples.end(), [](const std::pair<int,int> &left, const std::pair<int,int> &right) {
//        return left.second > right.second;
//    });
//    
//    call_stop = _rdtsc3();
//    call_run_time = call_stop - call_start;
//    printf("Cluster: %d \t time to compute union sorted by freq: %llu\n", clusterID, call_run_time);


// ----------Save Cluster TopK List (sorted by frequency) Below------------------
    
//    std::ofstream myfile;
//    std::string outputFile;
//    outputFile = "output_" + std::to_string(numberUsers) + ".txt";
//    myfile.open (outputFile, std::ofstream::out | std::ofstream::app);
    
//    myfile << "L List for Cluster " << clusterID << ": [";
//    myfile << tuples[0].first;
//    int listSize = (int)tuples.size();
//    for (i = 1; i < listSize; i++) {
//        myfile << "," << tuples[i].first;
//    }
//    myfile << "]\n";
    
    
// ----------Free Allocated Memory Below------------------
    
    for (i = 0; i < numBins; i++) {
        //free(indexPermutations[i]);
        MKL_free(sortedUpperBounds[i]);
        MKL_free(upperBounds[i]);
    }
//    for (i = 0; i < numUsersInCluster; i++){
//        free(allTopK[i]);
//    }
    MKL_free(upperBounds);
//    free(allTopK);
    MKL_free(userNormTuple_array);
    MKL_free(vectorized_Acos_input_uc);
    MKL_free(vectorized_Acos_output_uc);
    free(theta_bins);
    MKL_free(const_vector_theta);
    //free(indexPermutations);
    MKL_free(sortedUpperBounds);
    MKL_Free_Buffers();
    
    creationTime += upperBoundCreation_time;
    sortTime += sortUpperBound_time;
    computeKTime += computeTopK_time;
//    MKL_free(scoreA);

}






















