#ifndef cluster_hpp
#define cluster_hpp

void kmeans_clustering(int num_clusters, int num_iters, int sample_percentage, float* input_weights, 
        int num_cols, int num_rows, float** centroids_ptr, int** user_id_cluster_id_ptr, int num_threads);