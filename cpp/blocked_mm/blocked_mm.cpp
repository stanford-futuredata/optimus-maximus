//
//  blocked_mm.cpp
//  SimDex
//
//

#include "../parser.hpp"
#include "../utils.hpp"

#include <chrono>
#include <queue>
#include <algorithm>
#include <iostream>
#include <fstream>

#include <boost/format.hpp>
#include <boost/program_options.hpp>

#include <mkl.h>
#include <mkl_scalapack.h>

#include <omp.h>

void computeTopRating(float *ratings_matrix, const int num_users,
                      const int num_items) {
  int *top_K_items = new int[num_users];
#pragma omp parallel for
  for (int user_id = 0; user_id < num_users; user_id++) {

    unsigned long index = user_id;
    index *= num_items;
    float best_rating = ratings_matrix[index];
    int best_item_id = 0;
    for (int item_id = 1; item_id < num_items; ++item_id) {
      const float curr_rating = ratings_matrix[index + item_id];
      if (curr_rating > best_rating) {
        best_rating = curr_rating;
        best_item_id = item_id;
      }
    }
    top_K_items[user_id] = best_item_id;
  }
  delete[] top_K_items;
}

void computeTopK(float *ratings_matrix, const int num_users,
                 const int num_items, const int K) {

  int *top_K_items = new int[num_users * K];
#pragma omp parallel for
  for (int i = 0; i < num_users; i++) {

    // TODO: allocate vector on the stack, reserve the size we need or use the
    // insertion-and-copy array strategy that Matei suggested
    std::priority_queue<std::pair<float, int>,
                        std::vector<std::pair<float, int> >,
                        std::greater<std::pair<float, int> > > q;

    unsigned long index = i;
    index *= num_items;

    for (int j = 0; j < K; j++) {
      q.push(std::make_pair(ratings_matrix[index + j], j));
    }

    for (int j = K; j < num_items; j++) {
      if (ratings_matrix[index + j] > q.top().first) {
        q.pop();
        q.push(std::make_pair(ratings_matrix[index + j], j));
      }
    }

    for (int j = 0; j < K; j++) {
      const std::pair<float, int> p = q.top();
      top_K_items[i * K + K - 1 - j] = p.second;
      q.pop();
    }
  }
  delete[] top_K_items;
}

namespace opt = boost::program_options;

int main(int argc, const char *argv[]) {
  opt::options_description description("SimDex");
  description.add_options()("help,h", "Show help")(
      "weights-dir,w", opt::value<std::string>()->required(),
      "weights directory; must contain user_weights.csv and item_weights.csv")(
      "top-k,k", opt::value<int>()->required(),
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

  const std::string weights_dir = args["weights-dir"].as<std::string>();
  const std::string user_weights_filepath = weights_dir + "/user_weights.csv";
  const std::string item_weights_filepath = weights_dir + "/item_weights.csv";

  const int K = args["top-k"].as<int>();
  const int num_users = args["num-users"].as<int>();
  const int num_items = args["num-items"].as<int>();
  const int num_latent_factors = args["num-latent-factors"].as<int>();
  const int num_threads = args["num-threads"].as<int>();
  const std::string base_name = args["base-name"].as<std::string>();

#ifdef DEBUG
  MKL_Set_Num_Threads(1);
  omp_set_num_threads(1);
#else
  MKL_Set_Num_Threads(num_threads);
  omp_set_num_threads(num_threads);
#endif
  printf("Num threads: %d\n", mkl_get_max_threads());

  double time_st, time_end, gemm_time = 0, pr_queue_time = 0, compute_time = 0;

  const float alpha = 1.0;
  const float beta = 0.0;
  const int m = num_users;
  const int n = num_items;
  const int k = num_latent_factors;

  float *item_weights =
      parse_weights_csv(item_weights_filepath, num_items, num_latent_factors);
  float *user_weights =
      parse_weights_csv(user_weights_filepath, num_users, num_latent_factors);
  mkl_free_buffers();

  unsigned long tb = 1024 * 1024 * 1024;
  tb = tb * 800;

  unsigned long needed = m;
  needed *= n;
  needed *= sizeof(float);

  if (needed > tb) {
    const int per_instance = (int)floor((num_users) / 3.0);
    std::cout << "Blocking user matrix into 3 sub-matrices, " << per_instance
              << " rows per matrix" << std::endl;
    unsigned long mem_needed = 1;
    mem_needed = mem_needed * per_instance;
    mem_needed = mem_needed * n;
    mem_needed = mem_needed * sizeof(float);
    float *matrix_product = (float *)_malloc(mem_needed);

    dsecnd();
    time_st = dsecnd();

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, per_instance, n, k,
                alpha, user_weights, k, item_weights, k, beta, matrix_product,
                n);

    time_end = dsecnd();
    gemm_time += (time_end - time_st);

    dsecnd();
    time_st = dsecnd();
    if (K == 1) {
      computeTopRating(matrix_product, per_instance, num_items);
    } else {
      computeTopK(matrix_product, per_instance, num_items, K);
    }
    time_end = dsecnd();
    pr_queue_time += (time_end - time_st);

    dsecnd();
    time_st = dsecnd();
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, per_instance, n, k,
                alpha, &user_weights[per_instance], k, item_weights, k, beta,
                matrix_product, n);
    time_end = dsecnd();
    gemm_time += (time_end - time_st);

    dsecnd();
    time_st = dsecnd();
    if (K == 1) {
      computeTopRating(matrix_product, per_instance, num_items);
    } else {
      computeTopK(matrix_product, per_instance, num_items, K);
    }
    time_end = dsecnd();
    pr_queue_time += (time_end - time_st);

    dsecnd();
    time_st = dsecnd();
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, per_instance, n, k,
                alpha, &user_weights[per_instance * 2], k, item_weights, k,
                beta, matrix_product, n);
    time_end = dsecnd();
    gemm_time += (time_end - time_st);

    dsecnd();
    time_st = dsecnd();
    if (K == 1) {
      computeTopRating(matrix_product, per_instance, num_items);
    } else {
      computeTopK(matrix_product, per_instance, num_items, K);
    }
    time_end = dsecnd();
    pr_queue_time += (time_end - time_st);

    compute_time = gemm_time + pr_queue_time;

    _free(item_weights);
    _free(user_weights);
    _free(matrix_product);

  } else {
    float *matrix_product = (float *)_malloc(needed);

    dsecnd();
    time_st = dsecnd();
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, alpha,
                user_weights, k, item_weights, k, beta, matrix_product, n);
    time_end = dsecnd();
    gemm_time = (time_end - time_st);

    dsecnd();
    time_st = dsecnd();
    if (K == 1) {
      computeTopRating(matrix_product, num_users, num_items);
    } else {
      computeTopK(matrix_product, num_users, num_items, K);
    }
    time_end = dsecnd();
    pr_queue_time = (time_end - time_st);

    compute_time = gemm_time + pr_queue_time;

    _free(item_weights);
    _free(user_weights);
    _free(matrix_product);
  }

  std::ofstream timing_stats_file;
  const unsigned int curr_time =
      std::chrono::system_clock::now().time_since_epoch().count();
  const std::string timing_stats_fname =
      base_name + "_timing_" + std::to_string(curr_time) + ".csv";
  timing_stats_file.open(timing_stats_fname, std::ios_base::app);
  timing_stats_file << "model,K,num_latent_factors,num_threads,gemm_time,pr_"
                       "queue_time,comp_time" << std::endl;
  const std::string timing_stats =
      (boost::format("%1%,%2%,%3%,%4%,%5%,%6%,%7%") % base_name %
       num_latent_factors % num_threads % K % gemm_time % pr_queue_time %
       compute_time).str();
  timing_stats_file << timing_stats << std::endl;
  timing_stats_file.close();

  printf("gemm time: %f secs\n", gemm_time);
  printf("priority queue time: %f secs\n", pr_queue_time);
  printf("total comp time: %f secs\n", compute_time);

  return 0;
}