//
//  main.cpp
//  Simdex
//
//  Created by Geet Sethi on 10/24/16.
//  Copyright © 2016 Geet Sethi. All rights reserved.
//

#include "algo.hpp"
#include "parser.hpp"
#include "arith.hpp"
#include "utils.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/stat.h>
#include <functional>
#include <utility>
#include <queue>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <string>
#include <iostream>
#include <fstream>
#include <math.h>

#include <boost/format.hpp>

#include <mkl.h>
#include <mkl_scalapack.h>

extern int numberUsers;
extern int numFeatures;
extern int numItems;
extern int numClusters;

double creationTime;
double sortTime;
double computeKTime;

/**
 * Input: user id -> cluster id mapping (k-means assignments), user weights
 * Output: array of vectors: cluster_id -> vector of user ids, user weights
 * resorted in cluster order (same pointer as before)
 **/
std::vector<size_t> *build_cluster_index(const int *user_id_cluster_id,
                                         float *&user_weights,
                                         const size_t num_users,
                                         const size_t num_latent_factors,
                                         const size_t num_clusters) {

  std::vector<size_t> *cluster_index = new std::vector<size_t>[num_clusters];
  for (size_t user_id = 0; user_id < num_users; ++user_id) {
    cluster_index[user_id_cluster_id[user_id]].push_back(user_id);
  }

  float *user_weights_new =
      (float *)_malloc(sizeof(float) * num_users * num_latent_factors);

  size_t new_user_ind = 0;
  for (size_t i = 0; i < num_clusters; ++i) {
    const std::vector<size_t> user_ids_for_cluster = cluster_index[i];
    const size_t num_users_in_cluster = user_ids_for_cluster.size();

    for (size_t j = 0; j < num_users_in_cluster; ++j) {
      const size_t user_id = user_ids_for_cluster[j];
      std::memcpy(&user_weights_new[new_user_ind * num_latent_factors],
                  &user_weights[user_id * num_latent_factors],
                  sizeof(float) * num_latent_factors);
      ++new_user_ind;
    }
  }
  _free(user_weights);
  user_weights = user_weights_new;
  return cluster_index;
}

int main(int argc, const char *argv[]) {
  if (argc != 13) {
    printf(
        "Usage: Simdex <numItems> <num_latent_factors> <numClusters> <numBins> "
        "<K> "
        "<numClusters to Time> <Weight Directory> <Cluster Directory> "
        "<MaxThreads> <sample_size> <Iters> <num_users>\n");
    exit(1);
  }

  std::vector<std::string> allArgs(argv, argv + argc);

  const size_t num_users = atoi(argv[12]);
  const size_t num_items = atoi(argv[1]);
  const size_t num_latent_factors = atoi(argv[2]);
  const size_t num_clusters = atoi(argv[3]);
  const size_t num_bins = atoi(argv[4]);
  const size_t K = atoi(argv[5]);
  const size_t sample_size = atoi(argv[10]);
  const size_t num_iters = atoi(argv[11]);

  int maxThreads = atoi(argv[9]);
  MKL_Set_Num_Threads(maxThreads);

  const std::string user_weights_path = allArgs[7] + "/user_weights.csv";
  const std::string item_weights_path = allArgs[7] + "/item_weights.csv";

  const std::string centroids_path = allArgs[8] + "/" + allArgs[10] + "/" +
                                     allArgs[11] + "/" + allArgs[3] +
                                     "_centroids.csv";
  const std::string userToClustersPath = allArgs[8] + "/" + allArgs[10] + "/" +
                                         allArgs[11] + "/" + allArgs[3] +
                                         "_user_cluster_ids";

  setNumFeatures(num_latent_factors);
  setNumItems(num_items);
  setNumClusters(num_clusters);

  double time_start, time_end, time_diff1, time_diff2, time_diff3;

  time_start = dsecnd();
  time_start = dsecnd();
  int *user_id_cluster_id = parse_ids_csv(userToClustersPath, num_users);
  float *centroids =
      parse_weights_csv(centroids_path, numClusters, numFeatures);
  float *user_weights =
      parse_weights_csv(user_weights_path, num_users, numFeatures);
  std::vector<size_t> *cluster_index =
      build_cluster_index(user_id_cluster_id, user_weights, num_users,
                          num_latent_factors, num_clusters);
  // user_weights is correct--sorted correctly, matches cluster_index

  time_end = dsecnd();
  time_diff3 = (time_end - time_start);

  time_start = dsecnd();
  time_start = dsecnd();

  time_end = dsecnd();
  time_diff1 = (time_end - time_start);

  float *item_weights =
      parse_weights_csv(item_weights_path, numItems, numFeatures);

  unsigned long long call_start, call_stop, call_run_time = 0;
  call_start = _rdtsc2();

  float *theta_ics = compute_theta_ics(item_weights, centroids, num_items,
                                       num_clusters, num_latent_factors);
  // theta_ics are correct
  float *item_norms = computeAllItemNorms(item_weights);
  // item_norms are correct

  call_stop = _rdtsc2();
  call_run_time = call_stop - call_start;

  std::ofstream user_stats_file;
  std::string base_name = "test";
  const std::string fname =
      (boost::format("%1%_bins-%2%_K-%3%_sample-%4%_iters-%5%.csv") %
       base_name % num_bins % K % sample_size % num_iters).str();
  user_stats_file.open(fname);

  user_stats_file << "user_id,cluster_id,theta_uc,num_items_visited"
                  << std::endl;

  call_start = _rdtsc2();

  creationTime = 0;
  sortTime = 0;
  computeKTime = 0;

  time_start = dsecnd();
  time_start = dsecnd();
  size_t num_users_so_far = 0;
  for (size_t i = 0; i < num_clusters; i++) {
    std::cout << "Cluster ID " << i << std::endl;
    computeTopKForCluster(cluster_index, i, num_users_so_far, user_weights,
                          centroids, item_weights, item_norms, num_bins, K,
                          &theta_ics[i * num_items], num_items,
                          num_latent_factors, user_stats_file);
    num_users_so_far += cluster_index[i].size();
  }
  time_end = dsecnd();
  time_diff2 = (time_end - time_start);

  call_stop = _rdtsc2();
  call_run_time = call_stop - call_start;
  printf("comp time: %f secs \n", time_diff2);

  std::ofstream myfile;
  std::string outputFile;
  outputFile = "simdex2_u" + std::to_string(numberUsers) + "_f" +
               std::to_string(num_latent_factors) + "_k" + std::to_string(K) +
               "_c" + std::to_string(numClusters) + ".csv";
  myfile.open(outputFile, std::ofstream::out | std::ofstream::app);
  myfile << numberUsers << "," << numClusters << "," << num_latent_factors
         << "," << K << "," << num_bins << "," << time_diff3 << ","
         << time_diff1 << "," << time_diff2 << "," << creationTime << ","
         << sortTime << "," << computeKTime << "," << sample_size << ","
         << num_iters << std::endl;

  delete[] cluster_index;
  user_stats_file.close();
  myfile.close();

  return 0;
}
