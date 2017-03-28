#include <daal.h>
#include "service.h"
#include <mkl.h>

using namespace std;
using namespace daal;
using namespace daal::algorithms;
using namespace data_management;

float* computeCentroids(float* sample_users, const int num_clusters,
                        const int num_iters, const int num_cols,
                        const int sample_size) {
  double time_st, time_end, time_avg;

  kmeans::init::Batch<float, kmeans::init::randomDense> init(num_clusters);

  services::SharedPtr<NumericTable> sampleTablePtr =
      services::SharedPtr<NumericTable>(
          new HomogenNumericTable<float>(sample_users, num_cols, sample_size));

  init.input.set(kmeans::init::data, sampleTablePtr);

  time_st = dsecnd();
  time_st = dsecnd();
  init.compute();
  time_end = dsecnd();
  time_avg = (time_end - time_st);

#ifdef DEBUG
  printf("time taken to compute initial centroids: %f secs \n", time_avg);
#endif

  services::SharedPtr<NumericTable> centroids =
      init.getResult()->get(kmeans::init::centroids);

  kmeans::Batch<float> algorithm(num_clusters, num_iters);

  algorithm.input.set(kmeans::data, sampleTablePtr);
  algorithm.input.set(kmeans::inputCentroids, centroids);

  time_st = dsecnd();
  time_st = dsecnd();
  algorithm.compute();
  time_end = dsecnd();
  time_avg = (time_end - time_st);

#ifdef DEBUG
  printf("time taken to compute clusters: %f secs \n", time_avg);
#endif

#ifdef DEBUG
  printNumericTable(algorithm.getResult()->get(kmeans::assignments),
                    "First 10 cluster assignments:", 10);
  printNumericTable(algorithm.getResult()->get(kmeans::centroids),
                    "First 10 dimensions of centroids:", 20, 10);
#endif

  services::SharedPtr<NumericTable> endCentroids =
      algorithm.getResult()->get(kmeans::centroids);
  int nRows = endCentroids->getNumberOfRows();
  if (nRows != num_clusters) {
    std::cout << "ERROR!! first round of kmeans centroids rows don't match num "
                 "clusters" << std::endl;
  }
  BlockDescriptor<float> cent_block;
  endCentroids->getBlockOfRows(0, nRows, readOnly, cent_block);
  float* centArray = cent_block.getBlockPtr();
  float* returnArray =
      (float*)MKL_malloc(sizeof(float) * num_cols * num_clusters, 64);
  cblas_scopy(num_cols * num_clusters, centArray, 1, returnArray, 1);
  return returnArray;
}

int* assignment(float* input_weights, float* centroids, int num_clusters,
                int num_cols, int num_rows) {
  double time_st, time_end, time_avg;

  services::SharedPtr<NumericTable> centroidTablePtr =
      services::SharedPtr<NumericTable>(
          new HomogenNumericTable<float>(centroids, num_cols, num_clusters));

  services::SharedPtr<NumericTable> dataTablePtr =
      services::SharedPtr<NumericTable>(
          new HomogenNumericTable<float>(input_weights, num_cols, num_rows));

#ifdef DEBUG
  printNumericTable(centroidTablePtr, "Input Centroids:", 20, 10);
#endif

  kmeans::Batch<> algorithm2(num_clusters, 0);
  algorithm2.input.set(kmeans::data, dataTablePtr);
  algorithm2.input.set(kmeans::inputCentroids, centroidTablePtr);

  time_st = dsecnd();
  time_st = dsecnd();
  algorithm2.compute();
  time_end = dsecnd();
  time_avg = (time_end - time_st);

#ifdef DEBUG
  printf("time taken to assign clusters: %f secs \n", time_avg);
#endif

#ifdef DEBUG
  printNumericTable(algorithm2.getResult()->get(kmeans::assignments),
                    "First 10 cluster assignments:", 10);
  printNumericTable(algorithm2.getResult()->get(kmeans::centroids),
                    "First 10 dimensions of centroids:", 20, 10);
#endif

  services::SharedPtr<NumericTable> assignments =
      algorithm2.getResult()->get(kmeans::assignments);
  size_t numUsers = assignments->getNumberOfRows();

#ifdef DEBUG
  size_t cols = assignments->getNumberOfColumns();
  cout << "Num Assignments: " << numUsers << "\t" << cols << std::endl;
#endif

  BlockDescriptor<int> assign_block;
  assignments->getBlockOfRows(0, numUsers, readOnly, assign_block);
  int* assignArray = assign_block.getBlockPtr();
  int* returnArray = (int*)MKL_malloc(sizeof(int) * numUsers, 64);
  cblas_scopy(numUsers, (float*)assignArray, 1, (float*)returnArray, 1);

#ifdef DEBUG
  for (int i = 0; i < 50; i++) {
    cout << "user: " << i << " assignment: " << returnArray[i] << std::endl;
  }
#endif

  return returnArray;
}

void kmeans_clustering(float* input_weights, const int num_rows,
                       const int num_cols, const int num_clusters,
                       const int num_iters, const int sample_percentage,
                       const int num_threads, float*& centroids,
                       int*& user_id_cluster_ids) {
  MKL_Free_Buffers();
  services::Environment::getInstance()->setNumberOfThreads(num_threads);
  daal::services::interface1::LibraryVersionInfo info_obj;
#ifdef DEBUG
  cout << info_obj.majorVersion << "\t" << info_obj.minorVersion << "\t"
       << info_obj.build << std::endl;
#endif

  // not sampling at the moment
  int sample_size = num_rows;

  centroids = computeCentroids(input_weights, num_clusters, num_iters, num_cols,
                               sample_size);
  user_id_cluster_ids =
      assignment(input_weights, centroids, num_clusters, num_cols, num_rows);
}