
#include "../utils.hpp"
#include "service.h"

#include <random>

#include <daal.h>
#include <mkl.h>

using namespace daal;
using namespace daal::algorithms;
using namespace data_management;

double* get_random_samples(double* input_weights, const int num_rows,
                           const int num_cols, const int sample_percentage,
                           int* num_samples) {
  std::random_device rd;
  std::mt19937 g(rd());

  std::vector<int> v(num_rows);
  std::iota(v.begin(), v.end(), 0);
  std::shuffle(v.begin(), v.end(), g);

  const int _num_samples = (sample_percentage / 100.0) * num_rows;
  *num_samples = _num_samples;

  double* sample_input_weights =
      (double*)_malloc(sizeof(double) * (_num_samples * num_cols));
  for (int i = 0; i < _num_samples; ++i) {
    std::memcpy(&sample_input_weights[i * num_cols],
                &input_weights[v[i] * num_cols], sizeof(double) * num_cols);
  }

  return sample_input_weights;
}

double* computeCentroids(double* sample_users, const int num_cols,
                         const int num_clusters, const int num_iters,
                         const int num_samples) {
  double time_st, time_end, time_avg;

  kmeans::init::Batch<double, kmeans::init::randomDense> init(num_clusters);

  services::SharedPtr<NumericTable> sampleTablePtr =
      services::SharedPtr<NumericTable>(
          new HomogenNumericTable<double>(sample_users, num_cols, num_samples));

  init.input.set(kmeans::init::data, sampleTablePtr);
#ifdef DEBUG
  time_st = dsecnd();
  time_st = dsecnd();
#endif
  init.compute();
#ifdef DEBUG
  time_end = dsecnd();
  time_avg = (time_end - time_st);
#endif

#ifdef DEBUG
  printf("time taken to compute initial centroids: %f secs \n", time_avg);
#endif

  services::SharedPtr<NumericTable> centroids =
      init.getResult()->get(kmeans::init::centroids);

  kmeans::Batch<double> algorithm(num_clusters, num_iters);

  algorithm.input.set(kmeans::data, sampleTablePtr);
  algorithm.input.set(kmeans::inputCentroids, centroids);

#ifdef DEBUG
  time_st = dsecnd();
  time_st = dsecnd();
#endif
  algorithm.compute();
#ifdef DEBUG
  time_end = dsecnd();
  time_avg = (time_end - time_st);
#endif

#ifdef DEBUG
  printf("time taken to compute clusters: %f secs \n", time_avg);
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
  BlockDescriptor<double> cent_block;
  endCentroids->getBlockOfRows(0, nRows, readOnly, cent_block);
  double* centArray = cent_block.getBlockPtr();
  double* returnArray =
      (double*)_malloc(sizeof(double) * num_cols * num_clusters);
  cblas_dcopy(num_cols * num_clusters, centArray, 1, returnArray, 1);
  return returnArray;
}

int* assignment(double* input_weights, double* centroids, int num_clusters,
                int num_cols, int num_rows) {
#ifdef DEBUG
  double time_st, time_end, time_avg;
#endif

  services::SharedPtr<NumericTable> centroidTablePtr =
      services::SharedPtr<NumericTable>(
          new HomogenNumericTable<double>(centroids, num_cols, num_clusters));

  services::SharedPtr<NumericTable> dataTablePtr =
      services::SharedPtr<NumericTable>(
          new HomogenNumericTable<double>(input_weights, num_cols, num_rows));

  printNumericTable(centroidTablePtr, "Input Centroids:", 20, 10);

  kmeans::Batch<> algorithm2(num_clusters, 0);
  algorithm2.input.set(kmeans::data, dataTablePtr);
  algorithm2.input.set(kmeans::inputCentroids, centroidTablePtr);

#ifdef DEBUG
  time_st = dsecnd();
  time_st = dsecnd();
#endif
  algorithm2.compute();
#ifdef DEBUG
  time_end = dsecnd();
  time_avg = (time_end - time_st);
#endif

#ifdef DEBUG
  printf("time taken to assign clusters: %f secs \n", time_avg);
  printNumericTable(algorithm2.getResult()->get(kmeans::assignments),
                    "First 10 cluster assignments:", 10);
  printNumericTable(algorithm2.getResult()->get(kmeans::centroids),
                    "First 10 dimensions of centroids:", 20, 10);
#endif

  services::SharedPtr<NumericTable> assignments =
      algorithm2.getResult()->get(kmeans::assignments);
  size_t num_users = assignments->getNumberOfRows();

#ifdef DEBUG
  size_t cols = assignments->getNumberOfColumns();
  std::cout << "Num Assignments: " << num_users << "\t" << cols << std::endl;
#endif

  BlockDescriptor<int> assign_block;
  assignments->getBlockOfRows(0, num_users, readOnly, assign_block);
  int* assignArray = assign_block.getBlockPtr();
  int* returnArray = (int*)_malloc(sizeof(int) * num_users);
  cblas_scopy(num_users, (float*)assignArray, 1, (float*)returnArray, 1);

#ifdef DEBUG
  for (int i = 0; i < 50; i++) {
    std::cout << "user: " << i << " assignment: " << returnArray[i]
              << std::endl;
  }
#endif

  return returnArray;
}

void kmeans_clustering(double* input_weights, const int num_rows,
                       const int num_cols, const int num_clusters,
                       const int num_iters, const int sample_percentage,
                       const int num_threads, double*& centroids,
                       int*& user_id_cluster_ids) {
  MKL_Free_Buffers();
  services::Environment::getInstance()->setNumberOfThreads(num_threads);
  daal::services::interface1::LibraryVersionInfo info_obj;
#ifdef DEBUG
  std::cout << info_obj.majorVersion << "\t" << info_obj.minorVersion << "\t"
            << info_obj.build << std::endl;
#endif

  int num_samples = 0;

  double* sampled_input_weights = get_random_samples(
      input_weights, num_rows, num_cols, sample_percentage, &num_samples);

  centroids = computeCentroids(sampled_input_weights, num_cols, num_clusters,
                               num_iters, num_samples);
  user_id_cluster_ids =
      assignment(input_weights, centroids, num_clusters, num_cols, num_rows);
  _free(sampled_input_weights);
}

void random_clustering(double* input_weights, const int num_rows,
                       const int num_cols, const int num_clusters,
                       const int num_threads, double*& centroids,
                       int*& user_id_cluster_ids) {

  MKL_Free_Buffers();
  services::Environment::getInstance()->setNumberOfThreads(num_threads);
  daal::services::interface1::LibraryVersionInfo info_obj;
#ifdef DEBUG
  std::cout << info_obj.majorVersion << "\t" << info_obj.minorVersion << "\t"
            << info_obj.build << std::endl;
#endif

  std::default_random_engine generator;
  std::normal_distribution<double> distribution(0.0, 1.0);

  double* centroids_arr =
      (double*)_malloc(sizeof(double) * num_cols * num_clusters);
  centroids = centroids_arr;
  for (int i = 0; i < (num_clusters * num_cols); i++) {
    centroids_arr[i] = distribution(generator);
  }
  user_id_cluster_ids =
      assignment(input_weights, centroids, num_clusters, num_cols, num_rows);
}