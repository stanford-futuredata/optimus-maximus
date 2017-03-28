/* file: kmeans_dense_batch.cpp */
/*******************************************************************************
 * Copyright 2014-2017 Intel Corporation All Rights Reserved.
 *
 * The source code,  information  and material  ("Material") contained  herein is
 * owned by Intel Corporation or its  suppliers or licensors,  and  title to such
 * Material remains with Intel  Corporation or its  suppliers or  licensors.  The
 * Material  contains  proprietary  information  of  Intel or  its suppliers  and
 * licensors.  The Material is protected by  worldwide copyright  laws and treaty
 * provisions.  No part  of  the  Material   may  be  used,  copied,  reproduced,
 * modified, published,  uploaded, posted, transmitted,  distributed or disclosed
 * in any way without Intel's prior express written permission.  No license under
 * any patent,  copyright or other  intellectual property rights  in the Material
 * is granted to  or  conferred  upon  you,  either   expressly,  by implication,
 * inducement,  estoppel  or  otherwise.  Any  license   under such  intellectual
 * property rights must be express and approved by Intel in writing.
 *
 * Unless otherwise agreed by Intel in writing,  you may not remove or alter this
 * notice or  any  other  notice   embedded  in  Materials  by  Intel  or Intel's
 * suppliers or licensors in any way.
 *******************************************************************************/

/*
 !  Content:
 !    C++ example of dense K-Means clustering in the batch processing mode
 !******************************************************************************/

#include <daal.h>
#include "service.h"
#include <mkl.h>
//#include "random_num.hpp"

using namespace std;
//using namespace daal;
//using namespace daal::algorithms;
//using namespace data_management;

/* Input data set parameters */
string datasetFileName     = "s2_userWeights.csv";

/* K-Means algorithm parameters */
int nClusters;
const size_t nIterations = 10;
const size_t nThreads    = 4;
size_t nThreadsInit;
size_t nThreadsNew;

typedef struct kmeans_data{
    float *assignments;
    float *centroids;
    
}kmeans_t;

float *computeCentroids(float *sample_users, size_t nCols, int sampleSize){
    using namespace std;
    using namespace daal;
    using namespace daal::algorithms;
    using namespace data_management;
    
    double time_st, time_end, time_avg;
    
    kmeans::init::Batch<float,kmeans::init::randomDense> init(nClusters);
    
//        init.input.set(kmeans::init::data, dataSource.getNumericTable());
//
    services::SharedPtr<NumericTable> sampleTablePtr = services::SharedPtr<NumericTable>(new HomogenNumericTable<float>(sample_users, nCols, sampleSize));
    
    
    
    init.input.set(kmeans::init::data, sampleTablePtr);
    
    
//    time_end = dsecnd();
//    time_avg = (time_end - time_st);
//    printf("initilization time taken: %f secs \n", time_avg);
    
    time_st = dsecnd();
    time_st = dsecnd();
    init.compute();
    time_end = dsecnd();
    time_avg = (time_end - time_st);
    printf("time taken to compute initial centroids: %f secs \n", time_avg);
    
    services::SharedPtr<NumericTable> centroids = init.getResult()->get(kmeans::init::centroids);
    
    /* Create an algorithm object for the K-Means algorithm */
    kmeans::Batch<float> algorithm(nClusters, nIterations);
    
    algorithm.input.set(kmeans::data,           sampleTablePtr);
    algorithm.input.set(kmeans::inputCentroids, centroids);
    
    
    
    
    
    time_st = dsecnd();
    time_st = dsecnd();
    algorithm.compute();
    time_end = dsecnd();
    time_avg = (time_end - time_st);
    printf("time taken to compute clusters: %f secs \n", time_avg);
    
    /* Print the clusterization results */
    printNumericTable(algorithm.getResult()->get(kmeans::assignments), "First 10 cluster assignments:", 10);
    printNumericTable(algorithm.getResult()->get(kmeans::centroids  ), "First 10 dimensions of centroids:", 20, 10);
    printNumericTable(algorithm.getResult()->get(kmeans::goalFunction), "Goal function value:");
    printNumericTable(algorithm.getResult()->get(kmeans::nIterations), "iteration values:", nIterations);
    
    services::SharedPtr<NumericTable> endCentroids = algorithm.getResult()->get(kmeans::centroids);
    size_t nRows = endCentroids->getNumberOfRows();
    if (nRows != nClusters) {
        printf("ERROR!! first round of kmeans centroids rows dont match num clusters.\n");
    }
    //    size_t nCols = dataTable->getNumberOfColumns();
    BlockDescriptor<float> cent_block;
    endCentroids->getBlockOfRows(0, nRows, readOnly, cent_block);
    float *centArray = cent_block.getBlockPtr();
    
    float *returnArray = (float*)MKL_malloc(sizeof(float)*nCols*nClusters, 64);
    cblas_scopy(nCols*nClusters, centArray, 1, returnArray, 1);
    
    return returnArray;
}

void assignment(float *inputArray, float *centroids, size_t nCols, size_t nRows){
    using namespace std;
    using namespace daal;
    using namespace daal::algorithms;
    using namespace data_management;
    
    double time_st, time_end, time_avg;

    
    
    services::SharedPtr<NumericTable> centroidTablePtr = services::SharedPtr<NumericTable>(new HomogenNumericTable<float>(centroids, nCols, nClusters));
    
    services::SharedPtr<NumericTable> dataTablePtr = services::SharedPtr<NumericTable>(new HomogenNumericTable<float>(inputArray, nCols, nRows));
    
    
        printNumericTable(centroidTablePtr, "Input Centroids:", 20, 10);
    
    kmeans::Batch<> algorithm2(nClusters, 0);
    algorithm2.input.set(kmeans::data, dataTablePtr);
    algorithm2.input.set(kmeans::inputCentroids, centroidTablePtr);
    
    time_st = dsecnd();
    time_st = dsecnd();
    algorithm2.compute();
    time_end = dsecnd();
    time_avg = (time_end - time_st);
    printf("time taken to assign clusters: %f secs \n", time_avg);
    
    printNumericTable(algorithm2.getResult()->get(kmeans::assignments), "First 10 cluster assignments:", 10);
    printNumericTable(algorithm2.getResult()->get(kmeans::centroids  ), "First 10 dimensions of centroids:", 20, 10);
    printNumericTable(algorithm2.getResult()->get(kmeans::goalFunction), "Goal function value:");
    
    services::SharedPtr<NumericTable> endCentroids = algorithm2.getResult()->get(kmeans::centroids);
    size_t nRows2 = endCentroids->getNumberOfRows();
    if (nRows2 != nClusters) {
        printf("ERROR!! first round of kmeans centroids rows dont match num clusters.\n");
    }
    //    size_t nCols = dataTable->getNumberOfColumns();
    BlockDescriptor<float> cent_block;
    endCentroids->getBlockOfRows(0, nRows2, readOnly, cent_block);
    float *centArray = cent_block.getBlockPtr();
    
    //testing
    
}

int main(int argc, char *argv[])
{
    using namespace std;
    using namespace daal;
    using namespace daal::algorithms;
    using namespace data_management;
    MKL_Free_Buffers();
    
//    checkArguments(argc, argv, 1, &datasetFileName);
    
//    nClusters = atoi(argv[1]);
    
    daal::services::interface1::LibraryVersionInfo info_obj;
    cout << info_obj.majorVersion << "\t" << info_obj.minorVersion << "\t" << info_obj.build << std::endl;
    
    
    cout << "Please enter a cluster value: ";
    cin >> nClusters;
    printf("num clusters: %d\n", nClusters);
    
//    double time_st = dsecnd();
//    time_st = dsecnd();
    
    /* Get the number of threads that is used by the library by default */
    nThreadsInit = services::Environment::getInstance()->getNumberOfThreads();
    
    /* Set the maximum number of threads to be used by the library */
    services::Environment::getInstance()->setNumberOfThreads(nThreads);
    
    /* Get the number of threads that is used by the library after changing */
    nThreadsNew = services::Environment::getInstance()->getNumberOfThreads();
    
    /* Initialize FileDataSource to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager, float> dataSource(datasetFileName, DataSource::doAllocateNumericTable,
                                                 DataSource::doDictionaryFromContext);
        
    /* Retrieve the data from the input file */
    dataSource.loadDataBlock();
    
    services::SharedPtr<NumericTable> dataTable = dataSource.getNumericTable();
    size_t nRows = dataTable->getNumberOfRows();
    size_t nCols = dataTable->getNumberOfColumns();
    BlockDescriptor<float> block;
    dataTable->getBlockOfRows(0, nRows, readOnly, block);
    float *inputArray = block.getBlockPtr();
    

    
    double time_st = dsecnd();
    time_st = dsecnd();
    
    
    float *centArray = computeCentroids(inputArray, nCols, nRows);
    
    assignment(inputArray, centArray, nCols, nRows);
    
//    data_management::NumericTablePtr centroidTablePtr = NumericTablePtr(new HomogenNumericTable<float>(centArray, nCols, nClusters));
//    
//    
//    kmeans::Batch<float> algorithm2(nClusters, 0);
//    algorithm2.input.set(kmeans::data, dataTable);
//    algorithm2.input.set(kmeans::inputCentroids, centroidTablePtr);
//    algorithm2.compute();
//    
//    double time_end = dsecnd();
//    double time_avg = (time_end - time_st);
//    printf("comp time taken: %f secs \n", time_avg);
//    
//    printNumericTable(algorithm2.getResult()->get(kmeans::assignments), "First 10 cluster assignments:", 10);
//    printNumericTable(algorithm2.getResult()->get(kmeans::centroids  ), "First 10 dimensions of centroids:", 20, 10);
//    printNumericTable(algorithm2.getResult()->get(kmeans::goalFunction), "Goal function value:");

    
    MKL_Free_Buffers();
    
    
    return 0;
}