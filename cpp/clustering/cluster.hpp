#ifndef cluster_hpp
#define cluster_hpp

double kmeans_clustering(double* input_weights, const int num_rows,
                         const int num_cols, const int num_clusters,
                         const int num_iters, const int sample_percentage,
                         double*& centroids, uint32_t*& user_id_cluster_ids);

#endif /* cluster_hpp */