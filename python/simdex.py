#!/usr/bin/env python

from __future__ import division
from __future__ import print_function
from builtins import zip

import numpy as np
import pandas as pd
import heapq
import argparse
import os
from collections import defaultdict


def read_weight_csv(fname):
    return np.loadtxt(fname, delimiter=',')


def read_all_weights(weights_dir):
    return (read_weight_csv(os.path.join(weights_dir, 'user_weights.csv')),
            read_weight_csv(os.path.join(weights_dir, 'item_weights.csv')))


def read_centroids_and_assignments(clusters_dir, num_clusters,
                                   sample_percentage, num_iters):
    return (read_weight_csv(
        os.path.join(clusters_dir,
                     str(sample_percentage),
                     str(num_iters), '%d_centroids.csv' % num_clusters)),
            np.loadtxt(
                os.path.join(clusters_dir,
                             str(sample_percentage),
                             str(num_iters),
                             '%d_user_cluster_ids' % num_clusters),
                dtype='int'))


def invert_cluster_assignments(assignments):
    inverted_dict = defaultdict(list)
    for user_id, cluster_id in enumerate(assignments):
        inverted_dict[cluster_id].append(user_id)
    return dict(inverted_dict).values()


def angle_between_vectors_w_norms(v1, v2, normv1, normv2):
    theta = np.arccos(np.dot(v1, v2) / (normv1 * normv2))
    if np.isnan(theta):
        return 0
    return theta


def angular_upper_bound(item_norm, theta_ic, theta_uc):
    if (theta_ic - theta_uc < 0):
        return item_norm
    else:
        return item_norm * np.cos(theta_ic - theta_uc)


# find the smallest theta_b greater than theta_uc, return
# the corresponding list of upper bounds
def find_upper_bounds(theta_uc, upper_bounds_list):
    for theta_b, upper_bounds in upper_bounds_list:
        if theta_uc <= theta_b:
            return (theta_b, upper_bounds)
    # return the very last upper bounds list, which corresponds
    # to theta max
    return upper_bounds_list[-1]


# create min heap and walk through the sorted item list using Matei's idea:
#   add items to min heap ordered by true rating (i.e., compute the actual
#   user-item dot product)
#   if min(heap) > upper bound of the item
#       done; return heap (also need to count the actual number of items visited)
#   else if np.dot(u, i) > min(heap)
#       pop and add <u, i>
def compute_top_K_with_bounds(user_weight,
                              item_weights,
                              upper_bounds,
                              user_norm,
                              K=100):
    top_K = []
    # First K items
    for _, item_id in upper_bounds[:K]:
        item_weight = item_weights[item_id]
        predicted_rating = np.dot(user_weight, item_weight)
        heapq.heappush(top_K, (predicted_rating, item_id))
    # Remaining N-K items
    num_items_visited = K
    for upper_bound, item_id in upper_bounds[K:]:

        # top_K[0] is the min element of the heap
        if top_K[0][0] > user_norm * upper_bound:
            break
        item_weight = item_weights[item_id]
        predicted_rating = np.dot(user_weight, item_weight)
        num_items_visited += 1
        if top_K[0][0] < predicted_rating:
            heapq.heappushpop(top_K, (predicted_rating, item_id))
    return (heapq.nlargest(K, top_K), num_items_visited)


## compute matrix-vector product for all items for a given user. Sort
## the ratings and return the top K
def compute_naive_top_K(user_weight, item_weights, K=100):
    ratings = np.sum(user_weight * item_weights, axis=1)
    return sorted(zip(ratings, range(len(ratings))), reverse=True)[:K]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_dir', required=True)
    parser.add_argument('--clusters_dir', required=True)
    parser.add_argument('--num_clusters', required=True, type=int)
    parser.add_argument('--num_iters', type=int, required=True)
    parser.add_argument('--sample_percentage', type=int, required=True)
    parser.add_argument('--num_bins', type=int, required=True)
    parser.add_argument(
        '--K', type=int, required=True, choices=[1, 5, 10, 25, 50])
    parser.add_argument('--base_name', required=True)
    args = parser.parse_args()

    weights_dir = args.weights_dir
    clusters_dir = args.clusters_dir
    num_clusters = args.num_clusters
    num_iters = args.num_iters
    sample_percentage = args.sample_percentage
    num_bins = args.num_bins
    K = args.K
    #headers = [
    #    'user_id', 'cluster_id', 'theta_uc', 'num_items_visited_optimal',
    #    'theta_b', 'num_items_visited'
    #]
    headers = ['user_id', 'cluster_id', 'theta_uc', 'num_items_visited']
    vals = []

    user_weights, item_weights = read_all_weights(weights_dir)
    centroids, assignments = read_centroids_and_assignments(
        clusters_dir, num_clusters, sample_percentage, num_iters)
    cluster_id_user_ids = invert_cluster_assignments(assignments)
    item_norms = np.array(
        [np.linalg.norm(item_weight) for item_weight in item_weights])

    # For each cluster + users assigned to that cluster
    for cluster_id, user_id_list in enumerate(cluster_id_user_ids):
        print('Cluster ID %d' % cluster_id)
        centroid = centroids[cluster_id]
        centroid_norm = np.linalg.norm(centroid)

        user_norms = np.array([
            np.linalg.norm(user_weights[user_id]) for user_id in user_id_list
        ])
        # Compute angle between each user and centroid
        theta_ucs = np.array([
            angle_between_vectors_w_norms(centroid, user_weights[user_id],
                                          centroid_norm, user_norms[i])
            for i, user_id in enumerate(user_id_list)
        ])
        # Compute angle between each item and centroid
        theta_ics = np.array([
            angle_between_vectors_w_norms(centroid, item_weight, centroid_norm,
                                          item_norms[item_id])
            for item_id, item_weight in enumerate(item_weights)
        ])
        # Descretize theta_ucs into `num_bins` bins
        theta_max = np.max(theta_ucs)
        theta_bins = list(np.linspace(0.0, theta_max, num_bins + 1))
        theta_bins = np.array(theta_bins[1:])  # drop 0th bin, since it's 0.0

        # For each bin, calculate the upper bound for the rating on all the
        # items. The upper bounds list consists of [(rating, item_id),...] and
        # should be sorted from highest to lowest
        upper_bounds_list = [(theta_b, sorted(
            [(angular_upper_bound(item_norm, theta_ics[item_id], theta_b),
              item_id) for item_id, item_norm in enumerate(item_norms)],
            reverse=True)) for theta_b in theta_bins]

        #if len(user_id_list) > 10:
        #    rand_indices = np.random.choice(len(user_id_list), 10)
        #    user_id_list = np.array(user_id_list)
        #    user_id_list = user_id_list[rand_indices]
        #else:
        #    rand_indices = range(len(user_id_list))

        # For each user id:
        # 1) Find bin for user; look up corresponding upper bounds list for
        #    that bin
        # 2) Compute top K using upper bounds
        # 3) Compute top K using upper bounds with theta_uc instead of theta_b
        # 4) Check that both match true (i.e., brute-force) top-K

        # for i, user_id in enumerate(user_id_list):
        # for i, user_id in zip(rand_indices, user_id_list):
        for i, user_id in zip(range(10), user_id_list[:10]):
            theta_uc = theta_ucs[i]
            theta_b, upper_bounds = find_upper_bounds(theta_uc,
                                                      upper_bounds_list)
            user_weight = user_weights[user_id]
            user_norm = user_norms[i]
            top_K, num_items_visited = compute_top_K_with_bounds(
                user_weight, item_weights, upper_bounds, user_norm, K=K)

            #user_upper_bounds = sorted(
            #    [(angular_upper_bound(item_norm, theta_ics[item_id], theta_uc),
            #      item_id) for item_id, item_norm in enumerate(item_norms)],
            #    reverse=True)
            #top_K_optimal, num_items_visited_optimal = compute_top_K_with_bounds(
            #    user_weight, item_weights, user_upper_bounds, user_norm, K=K)

            true_top_K = compute_naive_top_K(user_weight, item_weights, K=K)
            #top_K_optimal_items = list(zip(*top_K_optimal))[1]
            top_K_items = list(zip(*top_K))[1]
            true_top_K_items = list(zip(*true_top_K))[1]
            assert top_K_items == true_top_K_items
            #assert top_K_optimal_items == true_top_K_items
            #vals.append([
            #    user_id, cluster_id, theta_uc, num_items_visited_optimal,
            #    theta_b, num_items_visited
            #])
            vals.append([user_id, cluster_id, theta_uc, num_items_visited])

    df = pd.DataFrame(vals, columns=headers)
    df.to_csv(
        '%s_bins-%d_K-%d_sample-%d_iters-%d.csv' %
        (args.base_name, num_bins, K, sample_percentage, num_iters),
        index=False)


if __name__ == '__main__':
    main()
