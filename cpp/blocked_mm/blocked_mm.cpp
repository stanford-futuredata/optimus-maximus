//
//  blocked_mm.cpp
//  SimDex
//
//

#include "../parser.hpp"
#include "../utils.hpp"
#include "blocked_mm.hpp"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>

#include <boost/format.hpp>
#include <boost/program_options.hpp>

#ifdef MKL_ILP64
#include <mkl.h>
#else
#include <cblas.h>
#endif

#define L2_CACHE_SIZE 256000

namespace opt = boost::program_options;

int main(int argc, const char *argv[]) {
  opt::options_description description("SimDex");
  description.add_options()("help,h", "Show help")(
      "user-weights,q", opt::value<std::string>()->required(),
      "user weights file")(
      "item-weights,p", opt::value<std::string>()->required(),
      "item-weights file")("top-k,k", opt::value<int>()->required(),
                           "Top K items to return per user")(
      "num-users,m", opt::value<int>()->required(), "Number of users")(
      "num-items,n", opt::value<int>()->required(), "Number of items")(
      "num-latent-factors,f", opt::value<int>()->required(),
      "Nubmer of latent factors")("num-threads,t",
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

  const std::string user_weights_filepath =
      args["user-weights"].as<std::string>();
  const std::string item_weights_filepath =
      args["item-weights"].as<std::string>();

  const int K = args["top-k"].as<int>();
  const unsigned long num_users = args["num-users"].as<int>();
  const int num_items = args["num-items"].as<int>();
  const int num_latent_factors = args["num-latent-factors"].as<int>();
  const unsigned long num_users_per_block =
      4 * (L2_CACHE_SIZE / (sizeof(double) * num_latent_factors));
  const int num_threads = args["num-threads"].as<int>();
  const std::string base_name = args["base-name"].as<std::string>();

#ifdef MKL_ILP64
  MKL_Set_Num_Threads(1);
  printf("Num threads: %d\n", mkl_get_max_threads());
#endif

  double gemm_time = 0, pr_queue_time = 0, compute_time = 0;

  double *item_weights = parse_weights_csv<double>(
      item_weights_filepath, num_items, num_latent_factors);
  double *user_weights = parse_weights_csv<double>(
      user_weights_filepath, num_users, num_latent_factors);
#ifdef MKL_ILP64
  mkl_free_buffers();
#endif

  // unsigned long available_mem = 1024 * 1024 * 1024;
  // available_mem = available_mem * 800;

  unsigned long needed = num_users_per_block;
  needed *= num_items;
  needed *= sizeof(double);
  double *matrix_product = (double *)_malloc(needed);

  const float alpha = 1.0;
  const float beta = 0.0;
  const int n = num_items;
  const int k = num_latent_factors;

  int *top_K_items = new int[num_users * K];

  for (unsigned long num_users_so_far = 0; num_users_so_far < num_users;
       num_users_so_far += num_users_per_block) {
    std::cout << "Num users so far: " << num_users_so_far << std::endl;
    // Compute blocked matrix product
    bench_timer_t gemm_start = time_start();

    const int m = std::min(num_users_per_block, num_users - num_users_so_far);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, alpha,
                &user_weights[num_users_so_far * num_latent_factors], k,
                item_weights, k, beta, matrix_product, n);
    gemm_time += time_stop(gemm_start);

    // Find Top K
    bench_timer_t top_k_start = time_start();

    if (K == 1) {
      computeTopRating(matrix_product, &top_K_items[num_users_so_far], m,
                       num_items);
    } else {
      computeTopK(matrix_product, &top_K_items[num_users_so_far * K], m,
                  num_items, K);
    }
    pr_queue_time += time_stop(top_k_start);
  }
  compute_time = gemm_time + pr_queue_time;

  delete[] top_K_items;
  _free(item_weights);
  _free(user_weights);
  _free(matrix_product);

  std::ofstream timing_stats_file;
  const unsigned int curr_time =
      std::chrono::system_clock::now().time_since_epoch().count();
  const std::string timing_stats_fname =
      base_name + "_timing_" + std::to_string(curr_time) + ".csv";
  timing_stats_file.open(timing_stats_fname, std::ios_base::app);
  timing_stats_file
      << "model,num_latent_factors,num_threads,K,block_size,gemm_time,pr_"
         "queue_time,comp_time" << std::endl;
  const std::string timing_stats =
      (boost::format("%1%,%2%,%3%,%4%,%5%,%6%,%7%,%8%") % base_name %
       num_latent_factors % num_threads % K % num_users_per_block % gemm_time %
       pr_queue_time % compute_time).str();
  timing_stats_file << timing_stats << std::endl;
  timing_stats_file.close();

  printf("gemm time: %f secs\n", gemm_time);
  printf("priority queue time: %f secs\n", pr_queue_time);
  printf("total comp time: %f secs\n", compute_time);

  return 0;
}