
#include "../utils.hpp"
#include <random>
#include <armadillo>
#include <mkl.h>

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
  auto start = Time::now();
  double* sampled_user_weights = get_random_samples(
      all_user_weights, num_rows, num_cols, sample_percentage, &num_samples);
  mat input_mat(sampled_user_weights, num_samples, num_cols, false, true);
  const fsec random_s = Time::now() - start;
#ifdef DEBUG
  std::cout << "Random samples time: " << random_s.count() << std::endl;
#endif

#ifdef DEBUG
  start = Time::now();
#endif
  input_mat = input_mat.t();
#ifdef DEBUG
  const fsec transpose_input_s = Time::now() - start;
  std::cout << "Transpose input time: " << transpose_input_s.count() << std::endl;
#endif

  start = Time::now();
  gmm_diag model;
#ifdef DEBUG
  model.learn(input_mat, num_clusters, eucl_dist, static_subset, num_iters,
              0, 0, true);
#else
  model.learn(input_mat, num_clusters, eucl_dist, static_subset, num_iters,
              0, 0, false);
#endif
  const fsec clustering_s = Time::now() -start;
#ifdef DEBUG
  std::cout << "Clustering time: " << clustering_s.count() << std::endl;
  model.means.head_rows(std::min(num_cols, 5)).print();
  start = Time::now();
#endif
  mat means = model.means.t();
#ifdef DEBUG
  const fsec transpose_centroids_s = Time::now() - start;
  std::cout << "Transpose centroids time: " << transpose_centroids_s.count() << std::endl;
#endif

  centroids = means.memptr();

#ifdef DEBUG
  start = Time::now();
#endif
  mat all_users_mat(all_user_weights, num_rows, num_cols, false, true);
  all_users_mat = all_users_mat.t();
#ifdef DEBUG
  const fsec transpose_all_users_s = Time::now() - start;
  std::cout << "Transpose all users time: " << transpose_all_users_s.count() << std::endl;
#endif

  start = Time::now();
  urowvec assignments = model.assign(all_users_mat, eucl_dist);
  const fsec assignments_s = Time::now() - start;
#ifdef DEBUG
  assignments.head(50).print();
  std::cout << "Assignment time: " << assignments_s.count() << std::endl;
#endif

  // we have to copy elements of `assignments` to `user_id_cluster_ids`,
  // individually, because `urowvec` in Armadillo is `unsigned long long`
#ifdef DEBUG
  start = Time::now();
#endif
  user_id_cluster_ids = (uint32_t *) _malloc(sizeof(uint32_t) * assignments.n_elem);
  for (uint32_t i = 0; i < assignments.n_elem; ++i) {
    user_id_cluster_ids[i] = assignments[i];
  }
#ifdef DEBUG
  const fsec copy_s = Time::now() - start;
  std::cout << "Copy time: " << copy_s.count() << std::endl;
#endif
  _free(sampled_user_weights);
  return random_s.count() + clustering_s.count() + assignments_s.count();
}
