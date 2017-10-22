//
//  main.cpp
//  SimDex
//
//

#include "algo.hpp"
#include "arith.hpp"
#include "clustering/cluster.hpp"
#include "parser.hpp"
#include "utils.hpp"

#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <unordered_set>
#include <random>
#include <string>
#include <utility>

#include <boost/algorithm/string/join.hpp>
#include <boost/format.hpp>
#include <boost/program_options.hpp>

#ifdef MKL_ILP64
#include <mkl.h>
#endif

namespace opt = boost::program_options;

static bool sample = true;

std::unordered_set<int> pick_set(int N, int k, std::mt19937& gen) {
  std::uniform_int_distribution<> dis(1, N);
  std::unordered_set<int> elems;

  while (elems.size() < k) {
    elems.insert(dis(gen));
  }

  return elems;
}

/**
 * Generate k random numbers from range [0, N)
 **/
std::vector<int> get_random_users(int N, int k) {
  std::random_device rd;
  std::mt19937 gen(rd());

  std::unordered_set<int> elems = pick_set(N, k, gen);

  std::vector<int> result(elems.begin(), elems.end());
  std::shuffle(result.begin(), result.end(), gen);
  return result;
}

/**
 * Input: user id -> cluster id mapping (k-means assignments), user weights
 * Output: array of vectors: cluster_id -> vector of user ids, user weights
 * resorted in cluster order (same pointer as before)
 **/
std::vector<int>* build_sample_cluster_index(const uint32_t* user_id_cluster_ids,
    const std::vector<int> & random_user_ids,
    double*& user_weights,
    const int num_latent_factors,
    const int num_clusters,
    int* num_users_so_far_arr) {

  std::unordered_set<uint32_t> sample_cluster_ids;
  std::vector<int>* cluster_index = new std::vector<int>[num_clusters];
  for (auto &user_id : random_user_ids) {
    cluster_index[user_id_cluster_ids[user_id]].push_back(user_id);
  }

  double* user_weights_sample =
    (double*)_malloc(sizeof(double) * random_user_ids.size() * num_latent_factors);

  int new_user_ind = 0;
  int num_users_so_far = 0;
  for (int i = 0; i < num_clusters; ++i) {
    const std::vector<int> user_ids_for_cluster = cluster_index[i];
    const int num_users_in_cluster = user_ids_for_cluster.size();

    num_users_so_far_arr[i] = num_users_so_far;
    num_users_so_far += num_users_in_cluster;
    for (int j = 0; j < num_users_in_cluster; ++j) {
      const int user_id = user_ids_for_cluster[j];
      std::memcpy(&user_weights_sample[new_user_ind * num_latent_factors],
          &user_weights[user_id * num_latent_factors],
          sizeof(double) * num_latent_factors);
      ++new_user_ind;
    }
  }
  _free(user_weights);
  user_weights = user_weights_sample;
  return cluster_index;
}


/**
 * Input: user id -> cluster id mapping (k-means assignments), user weights
 * Output: array of vectors: cluster_id -> vector of user ids, user weights
 * resorted in cluster order (same pointer as before)
 **/
std::vector<int>* build_cluster_index(const uint32_t* user_id_cluster_ids,
    double*& user_weights,
    const int num_users,
    const int num_latent_factors,
    const int num_clusters,
    int* num_users_so_far_arr) {

  std::vector<int>* cluster_index = new std::vector<int>[num_clusters];
  for (int user_id = 0; user_id < num_users; ++user_id) {
    cluster_index[user_id_cluster_ids[user_id]].push_back(user_id);
  }

  double* user_weights_new =
    (double*)_malloc(sizeof(double) * num_users * num_latent_factors);

  int new_user_ind = 0;
  int num_users_so_far = 0;
  for (int i = 0; i < num_clusters; ++i) {
    const std::vector<int> user_ids_for_cluster = cluster_index[i];
    const int num_users_in_cluster = user_ids_for_cluster.size();

    num_users_so_far_arr[i] = num_users_so_far;
    num_users_so_far += num_users_in_cluster;
    for (int j = 0; j < num_users_in_cluster; ++j) {
      const int user_id = user_ids_for_cluster[j];
      std::memcpy(&user_weights_new[new_user_ind * num_latent_factors],
          &user_weights[user_id * num_latent_factors],
          sizeof(double) * num_latent_factors);
      ++new_user_ind;
    }
  }
  _free(user_weights);
  user_weights = user_weights_new;
  return cluster_index;
}

bool is_power_of_two(unsigned int x) {
  return ((x != 0) && ((x & (~x + 1)) == x));
}

int main(int argc, const char* argv[]) {
  opt::options_description description("SimDex");
  description.add_options()("help,h", "Show help")(
      "weights-dir,w", opt::value<std::string>()->required(),
      "weights directory; must contain user_weights.csv and item_weights.csv")(
        "clusters-dir,d", opt::value<std::string>(),
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
              "sample-percentage,s", opt::value<int>()->default_value(10),
              "Ratio of users to sample during clustering, between 0. and 1.")(
                "num-iters,i", opt::value<int>()->default_value(3),
                "Number of iterations to run clustering, default: 10")(
                  "num-bins,b", opt::value<int>()->default_value(1),
                  "Number of bins, default: 1")("batch-size",
                    opt::value<int>()->default_value(256),
                    "Batch size, default: 256")(
                      "num-threads,t", opt::value<int>()->default_value(1),
                      "Number of threads, default: 1")("sample", opt::bool_switch(&sample),
                        "sample users to test index")(
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
  const int num_bins = 1;  // args["num-bins"].as<int>();
  const int batch_size = args["batch-size"].as<int>();
  const int num_threads = args["num-threads"].as<int>();
  const std::string base_name = args["base-name"].as<std::string>();

  if (!is_power_of_two(batch_size)) {
    // batch_size must be a power of 2; exit if not
    std::cout << "Batch size " << batch_size << " is not a power of 2."
      << std::endl;
    exit(1);
  }

  std::string centroids_filepath = "";
  std::string user_id_cluster_ids_filepath = "";
  if (args.count("clusters-dir")) {
    const std::string clusters_dir = args["clusters-dir"].as<std::string>();
    centroids_filepath = clusters_dir + "/" +
      std::to_string(sample_percentage) + "/" +
      std::to_string(num_iters) + "/" +
      std::to_string(num_clusters) + "_centroids.csv";
    user_id_cluster_ids_filepath =
      clusters_dir + "/" + std::to_string(sample_percentage) + "/" +
      std::to_string(num_iters) + "/" + std::to_string(num_clusters) +
      "_user_cluster_ids";
  }

#ifdef MKL_ILP64
#ifdef DEBUG
  MKL_Set_Num_Threads(1);
#else
  MKL_Set_Num_Threads(num_threads);
#endif
#endif


  bench_timer_t parse_time_start = time_start();

  double* item_weights = parse_weights_csv<double>(
      item_weights_filepath, num_items, num_latent_factors);
  double* user_weights = parse_weights_csv<double>(
      user_weights_filepath, num_users, num_latent_factors);

  const double parse_time = time_stop(parse_time_start);

  double* centroids;
  uint32_t* user_id_cluster_ids;
  double cluster_time;
  if (args.count("clusters-dir")) {
    bench_timer_t cluster_start = time_start();
    user_id_cluster_ids =
      parse_ids_csv(user_id_cluster_ids_filepath, num_users);
    centroids = parse_weights_csv<double>(centroids_filepath, num_clusters,
        num_latent_factors);
    cluster_time = time_stop(cluster_start);
  } else {
    cluster_time = kmeans_clustering(
        user_weights, num_users, num_latent_factors, num_clusters, num_iters,
        sample_percentage, centroids, user_id_cluster_ids);
  }

  bench_timer_t index_start = time_start();


  int num_users_so_far_arr[num_clusters];
  std::vector<int>* cluster_index = NULL;
  if (sample) {
    const std::vector<int> random_user_ids = get_random_users(num_users, 0.001*num_users); // 0.1% of users
    cluster_index = build_sample_cluster_index(user_id_cluster_ids, random_user_ids, user_weights,
        num_latent_factors, num_clusters, num_users_so_far_arr);
  } else {
    cluster_index = build_cluster_index(user_id_cluster_ids, user_weights, num_users,
        num_latent_factors, num_clusters, num_users_so_far_arr);
  // user_weights is now sorted correctly, matches cluster_index
  }

  float* item_norms =
    compute_norms_vector(item_weights, num_items, num_latent_factors);
  float* centroid_norms =
    compute_norms_vector(centroids, num_clusters, num_latent_factors);

  // theta_ics: a num_clusters x num_items matrix, theta_ics[i, j] = angle
  // between centroid i and item j
  float* theta_ics =
    compute_theta_ics(item_weights, centroids, num_items, num_latent_factors,
        num_clusters, item_norms, centroid_norms);

  const double index_time = time_stop(index_start);

  std::ofstream user_stats_file;
  const unsigned int curr_time =
    std::chrono::system_clock::now().time_since_epoch().count();
#ifdef STATS
  const std::string user_stats_fname =
    base_name + "_user_stats_K-" + std::to_string(K) + "_" + std::to_string(curr_time) + ".csv";
  user_stats_file.open(user_stats_fname);
  user_stats_file << "cluster_id,theta_uc,theta_b,num_items_visited,query_time"
    << std::endl;
#endif
  bench_timer_t algo_start = time_start();

  // TODO: These buffers are reused across multiple calls to
  // computeTopKForCluster.  For multiple threads, there will be
  // contention--need to allocate a buffer per thread.
  int* top_K_items = (int*)_malloc(num_users * K * sizeof(int));
  float* upper_bounds = (float*)_malloc(num_bins * num_items * sizeof(float));
  int* sorted_upper_bounds_indices = (int*)_malloc(num_items * sizeof(int));
  float* sorted_upper_bounds = (float*)_malloc(num_items * sizeof(float));
  double* sorted_item_weights = (double*)_malloc(sizeof(double) * num_bins *
      num_items * num_latent_factors);

  for (int cluster_id = 0; cluster_id < num_clusters; cluster_id++) {
    if (cluster_index[cluster_id].size() == 0) {
      continue;
    }
#ifdef DEBUG
    std::cout << "Cluster ID " << cluster_id << std::endl;
#endif
    const int num_users_so_far = num_users_so_far_arr[cluster_id];
    computeTopKForCluster(
        &top_K_items[num_users_so_far * K], cluster_id,
        &centroids[cluster_id * num_latent_factors], cluster_index[cluster_id],
        &user_weights[num_users_so_far * num_latent_factors], item_weights,
        item_norms, &theta_ics[cluster_id * num_items],
        centroid_norms[cluster_id], num_items, num_latent_factors, K, batch_size,
        upper_bounds, sorted_upper_bounds_indices, sorted_upper_bounds,
        sorted_item_weights, user_stats_file);
  }

  const double algo_time = time_stop(algo_start);
  const double compute_time = cluster_time + index_time + algo_time;

  std::ofstream timing_stats_file;
#ifdef STATS
  const std::string timing_stats_fname =
    base_name + "_timing_STATS_1_" + std::to_string(curr_time) + ".csv";
#else
  const std::string timing_stats_fname =
    base_name + "_timing_" + std::to_string(curr_time) + ".csv";
#endif
  timing_stats_file.open(timing_stats_fname, std::ios_base::app);
  timing_stats_file
    << "model,K,num_latent_factors,num_threads,num_bins,batch_size,num_"
    "clusters,sample_"
    "percentage,"
    "num_iters,"
    "parse_time,cluster_time,index_time,algo_time,comp_time" << std::endl;
  const std::string timing_stats =
    (boost::format(
                   "%1%,%2%,%3%,%4%,%5%,%6%,%7%,%8%,%9%,%10%,%11%,%12%,%13%,%14%") %
     base_name % K % num_latent_factors % num_threads % num_bins % batch_size %
     num_clusters % sample_percentage % num_iters % parse_time % cluster_time %
     index_time % algo_time % compute_time).str();
  timing_stats_file << timing_stats << std::endl;
  timing_stats_file.close();

  printf("parse time: %f secs\n", parse_time);
  printf("cluster time: %f secs\n", cluster_time);
  printf("index time: %f secs\n", index_time);
  printf("algo time: %f secs\n", algo_time);
  printf("total comp time: %f secs\n", compute_time);

  _free(centroids);
  _free(user_id_cluster_ids);
  _free(top_K_items);
  _free(upper_bounds);
  _free(sorted_upper_bounds_indices);
  _free(sorted_upper_bounds);
  _free(sorted_item_weights);
  delete[] cluster_index;
#ifdef STATS
  user_stats_file.close();
#endif

  return 0;
}
