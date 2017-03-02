//
//  arith.cpp
//  fomo_preproc
//
//  Created by Geet Sethi on 10/27/16.
//  Copyright © 2016 Geet Sethi. All rights reserved.
//

#include "arith.hpp"
#include "parser.hpp"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <mkl.h>
#include "mz_special.hpp"

extern int numberUsers;
extern int numFeatures;
extern int numItems;
extern int numClusters;

float *computeAllItemNorms(float *allItemWeights){
    float *allItemNorms = (float *)malloc(sizeof(float)*(numItems+1));
    
    for (int i = 0; i < (numItems+1); i++) {
        allItemNorms[i] = cblas_snrm2(numFeatures, &allItemWeights[i*numFeatures], 1);
    }
    
    return allItemNorms;
}

float *computeAllItemNormsForMM(float *allItemWeights){
    float *allItemNorms = (float *)mkl_malloc(sizeof(float)*(numItems+1)*numFeatures, 64);
    if (allItemNorms == NULL) {
        printf( "\n ERROR: Can't allocate memory for allItemNorms. Aborting... \n\n");
        mkl_free(allItemNorms);
        exit(1);
    }
    
    for (int i = 0; i < (numItems+1); i++) {
        allItemNorms[i*numFeatures] = cblas_snrm2(numFeatures, &allItemWeights[i*numFeatures], 1);
        for (int j = 0; j < numFeatures; j++) {
            allItemNorms[(i*numFeatures)+j] = allItemNorms[i*numFeatures];
        }
    }
    
    return allItemNorms;
}

float *computeAllCentroidNorms(float *clusterCentroids){
    float *allCentroidNorms = (float *)mkl_malloc(sizeof(float)*(numClusters*numFeatures), 64);
    if (allCentroidNorms == NULL) {
        printf( "\n ERROR: Can't allocate memory for allUserNorms. Aborting... \n\n");
        mkl_free(allCentroidNorms);
        exit(1);
    }
    
    for (int i = 0; i < numClusters; i++) {
        allCentroidNorms[i*numFeatures] = cblas_snrm2(numFeatures, &clusterCentroids[i*numFeatures], 1);
        for (int j = 0; j < numFeatures; j++) {
            allCentroidNorms[(i*numFeatures)+j] = allCentroidNorms[i*numFeatures];
        }
    }
    
    return allCentroidNorms;
}

float *computeCosineSimilarityItemsCentroids(float *allItems, float *allCentroids){
    
    int m, n, k, i;
    m = numClusters;
    k = numFeatures;
    n = numItems+1;
    float alpha = 1.0;
    float beta = 0.0;
    
    float *itemDOTcentroid = (float *)mkl_malloc(sizeof(float)*numClusters*(numItems+1), 64);
    if (itemDOTcentroid == NULL) {
        printf( "\n ERROR: Can't allocate memory for itemDOTcentroid. Aborting... \n\n");
        mkl_free(itemDOTcentroid);
        exit(1);
    }
    float *allItemsNew = (float *)mkl_malloc(sizeof(float)*(numItems+1)*numFeatures, 64);
    if (allItemsNew == NULL) {
        printf( "\n ERROR: Can't allocate memory for allItems. Aborting... \n\n");
        mkl_free(allItemsNew);
        exit(1);
    }
    float *allCentroidsNew = (float *)mkl_malloc(sizeof(float)*(numClusters)*numFeatures, 64);
    if (allCentroidsNew == NULL) {
        printf( "\n ERROR: Can't allocate memory for allCentroids. Aborting... \n\n");
        mkl_free(allCentroidsNew);
        exit(1);
    }
    
    
    float *allItemNorms = computeAllItemNormsForMM(allItems);
    float *allCentroidNorms = computeAllCentroidNorms(allCentroids);
    
    vsInv(numClusters*numFeatures, allCentroidNorms, allCentroidNorms);
    vsInv((numItems+1)*numFeatures, allItemNorms, allItemNorms);
    
    for (i = 0; i < (numItems+1); i++) {
        vsMul(numFeatures, &allItems[i*numFeatures], &allItemNorms[i*numFeatures], &allItemsNew[i*numFeatures]);
    }
    for (i = 0; i < numClusters; i++) {
        vsMul(numFeatures, &allCentroids[i*numFeatures], &allCentroidNorms[i*numFeatures], &allCentroidsNew[i*numFeatures]);
    }
    
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                m, n, k, alpha, allCentroidsNew, k, allItemsNew, k, beta, itemDOTcentroid, n);
    
    MKL_free(allCentroidsNew);
    MKL_free(allItemsNew);
    MKL_free(allItemNorms);
    MKL_free(allCentroidNorms);
    
    return itemDOTcentroid;
    
}

float *computeClusterUserNorms(float *allUserWeights, int numUsers, userNormTuple_t *userNormTuple_array){
    float *allUserNorms = (float *)mkl_malloc(sizeof(float)*(numUsers*numFeatures), 64);
    if (allUserNorms == NULL) {
        printf( "\n ERROR: Can't allocate memory for allUserNorms. Aborting... \n\n");
        mkl_free(allUserNorms);
        exit(1);
    }
    
    for (int i = 0; i < numUsers; i++) {
        allUserNorms[i*numFeatures] = cblas_snrm2(numFeatures, &allUserWeights[i*numFeatures], 1);
        userNormTuple_array[i].userNorm = allUserNorms[i*numFeatures];
        for (int j = 0; j < numFeatures; j++) {
            allUserNorms[(i*numFeatures)+j] = allUserNorms[i*numFeatures];
        }
    }
    
    return allUserNorms;
}

float *computeCosineSimilarityUserCluster(float *users, float *centroid, int numUsers, userNormTuple_t *userNormTuple_array){
    int m, k, i;
    m = numUsers;
    k = numFeatures;

    float alpha = 1.0;
    float beta = 0.0;
    int stride = 1;
    
    float *usersDOTcentroid = (float *)mkl_malloc(sizeof(float)*numUsers, 64);
    if (usersDOTcentroid == NULL) {
        printf( "\n ERROR: Can't allocate memory for usersDOTcentroid. Aborting... \n\n");
        mkl_free(usersDOTcentroid);
        exit(1);
    }
    
    float *centroidNorm = (float *)mkl_malloc(sizeof(float)*numFeatures, 64);
    if (centroidNorm == NULL) {
        printf( "\n ERROR: Can't allocate memory for centroidNorm. Aborting... \n\n");
        mkl_free(centroidNorm);
        exit(1);
    }
    
    centroidNorm[0] = cblas_snrm2(numFeatures, centroid, 1);
    for (i = 1; i < numFeatures; i++) {
        centroidNorm[i] = centroidNorm[0];
    }
    
    float *userNorms = computeClusterUserNorms(users, numUsers, userNormTuple_array);
    
    vsInv(numFeatures, centroidNorm, centroidNorm);
    vsInv((numUsers)*numFeatures, userNorms, userNorms);
    
    vsMul(numFeatures, centroid, centroidNorm, centroidNorm);
    for (i = 0; i < numUsers; i++) {
        vsMul(numFeatures, &users[i*numFeatures], &userNorms[i*numFeatures], &userNorms[i*numFeatures]);
    }
    
    cblas_sgemv(CblasRowMajor, CblasNoTrans, m, k, alpha, userNorms, k, centroidNorm, stride, beta, usersDOTcentroid, stride);
    
    MKL_free(userNorms);
    MKL_free(centroidNorm);
    
    return usersDOTcentroid;
    
}

void userDOTitemBLOCK(); // for topK calc
