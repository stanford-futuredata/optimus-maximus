#ifndef cluster_hpp
#define cluster_hpp

void random_clustering(float* input_weights, const int num_rows,
                       const int num_cols, const int num_clusters,
                       const int num_iters, const int sample_percentage,
                       const int num_threads, float*& centroids,
                       int*& user_id_cluster_ids);
void kmeans_clustering(float* input_weights, const int num_rows,
                       const int num_cols, const int num_clusters,
                       const int num_iters, const int sample_percentage,
                       const int num_threads, float*& centroids,
                       int*& user_id_cluster_ids);

#endif /* cluster_hpp */