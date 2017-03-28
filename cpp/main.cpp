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
#include "cluster.hpp"

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
#include <boost/program_options.hpp>

#include <mkl.h>
#include <mkl_scalapack.h>

namespace opt = boost::program_options;

double creationTime;
double sortTime;
double computeKTime;

/**
 * Input: user id -> cluster id mapping (k-means assignments), user weights
 * Output: array of vectors: cluster_id -> vector of user ids, user weights
 * resorted in cluster order (same pointer as before)
 **/
std::vector<int>* build_cluster_index(const int* user_id_cluster_id,
                                      float*& user_weights, const int num_users,
                                      const int num_latent_factors,
                                      const int num_clusters, int* num_users_so_far_array) {

  std::vector<int>* cluster_index = new std::vector<int>[num_clusters];
  for (int user_id = 0; user_id < num_users; ++user_id) {
    cluster_index[user_id_cluster_id[user_id]].push_back(user_id);
  }

  float* user_weights_new =
      (float*)_malloc(sizeof(float) * num_users * num_latent_factors);

  int new_user_ind = 0;
    int num_users_so_far_counter = 0;
  for (int i = 0; i < num_clusters; ++i) {
    const std::vector<int> user_ids_for_cluster = cluster_index[i];
    const int num_users_in_cluster = user_ids_for_cluster.size();
      num_users_so_far_array[i] = num_users_so_far_counter;
      num_users_so_far_counter += num_users_in_cluster;
    for (int j = 0; j < num_users_in_cluster; ++j) {
      const int user_id = user_ids_for_cluster[j];
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

int main(int argc, const char* argv[]) {
  opt::options_description description("SimDex");
  description.add_options()("help,h", "Show help")(
      "weights-dir,w", opt::value<std::string>()->required(),
      "weights directory; must contain user_weights.csv and item_weights.csv")(
      "clusters-dir,d", opt::value<std::string>()->required(),
      "clusters directory; must contain "
      "[sample_percentage]/[num_iters]/[num_clusters]_centroids.csv and "
      "[sample_percentage]/[num_iters]/[num_clusters]_user_cluster_ids")(
      "top-k,k", opt::value<int>()->required(),
      "Top K items to return per user")(
      "num-users,m", opt::value<int>()->required(), "Number of users")(
      "num-items,n", opt::value<int>()->required(), "Number of items")(
      "num-latent-factors,f", opt::value<int>()->required(),
      "Nubmer of latent factors")(
      "num-clusters,c", opt::value<int>()->required(), "Number of clusters")(
      "sample-percentage,s", opt::value<int>()->default_value(20),
      "Ratio of users to sample during clustering, between 0. and 1.")(
      "num-iters,i", opt::value<int>()->default_value(10),
      "Number of iterations to run clustering, default: 10")(
      "num-bins,b", opt::value<int>()->default_value(1),
      "Number of bins, default: 1")("num-threads,t",
                                    opt::value<int>()->default_value(1),
                                    "Number of threads, default: 1")(
      "base-name", opt::value<std::string>()->required(),
      "Base name for file output to record stats");

  opt::variables_map args;
  opt::store(opt::command_line_parser(argc, argv).options(description).run(),
             args);

  if (args.count("help")) {
    std::cout << description << std::endl;
    exit(0);
  }

  opt::notify(args);

  const std::string weights_dir = args["weights-dir"].as<std::string>();
  const std::string user_weights_filepath = weights_dir + "/user_weights.csv";
  const std::string item_weights_filepath = weights_dir + "/item_weights.csv";

  const int K = args["top-k"].as<int>();
  const int num_users = args["num-users"].as<int>();
  const int num_items = args["num-items"].as<int>();
  const int num_latent_factors = args["num-latent-factors"].as<int>();
  const int num_clusters = args["num-clusters"].as<int>();
  const int sample_percentage = args["sample-percentage"].as<int>();
  const int num_iters = args["num-iters"].as<int>();
  const int num_bins = args["num-bins"].as<int>();
  const int num_threads = args["num-threads"].as<int>();

  const std::string clusters_dir = args["clusters-dir"].as<std::string>();
  const std::string centroids_filepath =
      clusters_dir + "/" + std::to_string(sample_percentage) + "/" +
      std::to_string(num_iters) + "/" + std::to_string(num_clusters) +
      "_centroids.csv";
  const std::string user_id_cluster_id_filepath =
      clusters_dir + "/" + std::to_string(sample_percentage) + "/" +
      std::to_string(num_iters) + "/" + std::to_string(num_clusters) +
      "_user_cluster_ids";
  const std::string base_name = args["base-name"].as<std::string>();

  MKL_Set_Num_Threads(num_threads);

  double time_start, time_end;

  time_start = dsecnd();
  time_start = dsecnd();
  int* user_id_cluster_id =
      parse_ids_csv(user_id_cluster_id_filepath, num_users);
  float* centroids =
      parse_weights_csv(centroids_filepath, num_clusters, num_latent_factors);
  float* item_weights =
      parse_weights_csv(item_weights_filepath, num_items, num_latent_factors);
  float* user_weights =
      parse_weights_csv(user_weights_filepath, num_users, num_latent_factors);
  // user_weights is correct--sorted correctly, matches cluster_index

  time_end = dsecnd();
  const double parse_time = (time_end - time_start);

  float** centroids_ptr = &centroids;
  int** user_id_cluster_id_ptr = &user_id_cluster_id;

  kmeans_clustering(num_clusters, num_iters, sample_percentage, user_weights, 
        num_latent_factors, num_users, centroids_ptr, user_id_cluster_id_ptr, num_threads);

  exit(0);




  time_start = dsecnd();
  time_start = dsecnd();
    
    int* num_users_so_far_array = (int*)_malloc(sizeof(int)*num_clusters);

  std::vector<int>* cluster_index =
      build_cluster_index(user_id_cluster_id, user_weights, num_users,
                          num_latent_factors, num_clusters, num_users_so_far_array);
  // theta_ics: a num_clusters x num_items matrix, theta_ics[i, j] = angle
  // between centroid i and item j
  float* theta_ics = compute_theta_ics(item_weights, centroids, num_items,
                                       num_latent_factors, num_clusters);
  // theta_ics are correct
  float* item_norms =
      computeAllItemNorms(item_weights, num_items, num_latent_factors);
  // item_norms are correct

  time_end = dsecnd();
  const double index_time = (time_end - time_start);

  std::ofstream user_stats_file;
  const std::string fname =
      (boost::format("%1%_bins-%2%_K-%3%_sample-%4%_iters-%5%.csv") %
       base_name % num_bins % K % sample_percentage % num_iters).str();
  user_stats_file.open(fname);

  user_stats_file << "user_id,cluster_id,theta_uc,num_items_visited"
                  << std::endl;

  creationTime = 0;
  sortTime = 0;
  computeKTime = 0;

  time_start = dsecnd();
  time_start = dsecnd();
  int num_users_so_far = 0;
#pragma omp parallel for
  for (int cluster_id = 0; cluster_id < num_clusters; cluster_id++) {
    std::cout << "Cluster ID " << cluster_id << std::endl;
      num_users_so_far = num_users_so_far_array[cluster_id];
    computeTopKForCluster(
        cluster_id, &centroids[cluster_id * num_latent_factors],
        cluster_index[cluster_id],
        &user_weights[num_users_so_far * num_latent_factors], item_weights,
        item_norms, &theta_ics[cluster_id * num_items], num_items,
        num_latent_factors, num_bins, K, user_stats_file);
  }
  time_end = dsecnd();
  const double compute_time = (time_end - time_start);

  printf("parse time: %f secs \n", parse_time);
  printf("index time: %f secs \n", index_time);
  printf("comp time: %f secs \n", compute_time);

  delete[] cluster_index;
  user_stats_file.close();

  return 0;
}
