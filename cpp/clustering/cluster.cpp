
#include "../utils.hpp"
#include <random>
#include <armadillo>

using namespace arma;

// For debugging purposes in gdb, since we can't use overloaded operators
#ifdef DEBUG
void print(const vec& v) { v.print(); }
void print(const ivec& v) { v.print(); }
void print(const mat& v) { v.print(); }
void print(const urowvec& v) { v.print(); }
double get_elem(const mat& m, int i, int j) { return m(i, j); }
double get_elem(const vec& v, int i) { return v(i); }
int get_elem(const uvec& v, int i) { return v(i); }
double get_norm(const vec& v) { return norm(v, 2); }
double get_dot(const vec& u, const vec& v) { return dot(u, v); }
#endif

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

double kmeans_clustering(double* all_user_weights, const int num_rows,
                         const int num_cols, const int num_clusters,
                         const int num_iters, const int sample_percentage,
                         double*& centroids, uint32_t*& user_id_cluster_ids) {
  int num_samples = 0;
  bench_timer_t random_start = time_start();
  double* sampled_user_weights = get_random_samples(
      all_user_weights, num_rows, num_cols, sample_percentage, &num_samples);
  mat input_mat(sampled_user_weights, num_samples, num_cols, false, true);
  const double random_s = time_stop(random_start);
#ifdef DEBUG
  std::cout << "Random samples time: " << random_s << std::endl;
#endif

#ifdef DEBUG
  bench_timer_t transpose_start = time_start();
#endif
  input_mat = input_mat.t();
#ifdef DEBUG
  const double transpose_input_s = time_stop(transpose_start);
  std::cout << "Transpose input time: " << transpose_input_s << std::endl;
#endif

  bench_timer_t clustering_start = time_start();
  gmm_diag model;
#ifdef DEBUG
  model.learn(input_mat, num_clusters, eucl_dist, static_subset, num_iters, 0,
              0, true);
#else
  model.learn(input_mat, num_clusters, eucl_dist, static_subset, num_iters, 0,
              0, false);
#endif
  const double clustering_s = time_stop(clustering_start);
#ifdef DEBUG
  std::cout << "Clustering time: " << clustering_s << std::endl;
  model.means.head_rows(std::min(num_cols, 5)).print();
  bench_timer_t transpose_centroids_start = time_start();
#endif
  mat means = model.means.t();
#ifdef DEBUG
  const double transpose_centroids_s = time_stop(transpose_centroids_start);
  std::cout << "Transpose centroids time: " << transpose_centroids_s
            << std::endl;
#endif

#ifdef DEBUG
  bench_timer_t copy_centroids_start = time_start();
#endif
  // Copy centroids to separate array; means is on the stack
  centroids = (double*)_malloc(sizeof(double) * means.n_elem);
  std::memcpy(centroids, means.memptr(), sizeof(double) * means.n_elem);
#ifdef DEBUG
  const double copy_centroids_s = time_stop(copy_centroids_start);
  std::cout << "Copy centroids time: " << copy_centroids_s << std::endl;
#endif

#ifdef DEBUG
  bench_timer_t transpose_all_users_start = time_start();
#endif
  mat all_users_mat(all_user_weights, num_rows, num_cols, false, true);
  all_users_mat = all_users_mat.t();
#ifdef DEBUG
  const double transpose_all_users_s = time_stop(transpose_all_users_start);
  std::cout << "Transpose all users time: " << transpose_all_users_s
            << std::endl;
#endif

  bench_timer_t assignments_start = time_start();
  urowvec assignments = model.assign(all_users_mat, eucl_dist);
  const double assignments_s = time_stop(assignments_start);
#ifdef DEBUG
  assignments.head(50).print();
  std::cout << "Assignment time: " << assignments_s << std::endl;
#endif

// we have to copy elements of `assignments` to `user_id_cluster_ids`,
// individually, because `urowvec` in Armadillo is `unsigned long long`
#ifdef DEBUG
  bench_timer_t copy_assignments_start = time_start();
#endif
  user_id_cluster_ids =
      (uint32_t*)_malloc(sizeof(uint32_t) * assignments.n_elem);
  for (uint32_t i = 0; i < assignments.n_elem; ++i) {
    user_id_cluster_ids[i] = assignments[i];
  }
#ifdef DEBUG
  const double copy_assignments_s = time_stop(copy_assignments_start);
  std::cout << "Copy assignments time: " << copy_assignments_s << std::endl;
#endif
  _free(sampled_user_weights);
  return random_s + clustering_s + assignments_s;
}
