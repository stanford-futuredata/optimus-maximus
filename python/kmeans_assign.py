#!/usr/bin/env python

from daal.data_management import (
    readOnly,
    BlockDescriptor,
    DataSourceIface,
    FileDataSource, )

import daal.algorithms.kmeans as kmeans
from daal.algorithms.kmeans import init

from daal_utils import printNumericTable
from kmeans_sample import set_num_threads
import os
import argparse
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_dir', required=True)
    parser.add_argument('--num_clusters', required=True, type=int)
    parser.add_argument('--num_iters', type=int, required=True)
    parser.add_argument(
        '--sample_size',
        type=int,
        choices=range(1, 100),
        metavar='INT[1,100]',
        required=True)
    parser.add_argument('--num_threads', type=int, default=1)
    parser.add_argument('--output_dir_base', required=True)
    args = parser.parse_args()

    weights_dir = args.weights_dir
    num_clusters = args.num_clusters
    num_iters = args.num_iters
    sample_size = args.sample_size
    set_num_threads(args.num_threads)
    clusters_dir = os.path.join(args.output_dir_base,
                                str(sample_size), str(num_iters))
    if not os.path.exists(clusters_dir): os.makedirs(clusters_dir)

    datasetFileName = os.path.join(weights_dir, 'user_weights.csv')

    centroidsFileName = os.path.join(clusters_dir,
                                     '%d_centroids.csv' % num_clusters)

    centroidSource = FileDataSource(centroidsFileName,
                                    DataSourceIface.doAllocateNumericTable,
                                    DataSourceIface.doDictionaryFromContext)
    centroidSource.loadDataBlock()

    print(weights_dir)

    t0 = time.time()
    dataSource = FileDataSource(datasetFileName,
                                DataSourceIface.doAllocateNumericTable,
                                DataSourceIface.doDictionaryFromContext)

    dataSource.loadDataBlock()

    initAlg = init.Batch_Float32DeterministicDense(num_clusters)
    initAlg.input.set(init.data, centroidSource.getNumericTable())

    t1 = time.time()
    init_time = t1 - t0
    print('init time: %f' % init_time)

    t0 = time.time()
    res = initAlg.compute()
    t1 = time.time()
    centroid_time = t1 - t0
    print('centroid time: %f' % centroid_time)

    centroidsResult = res.get(init.centroids)

    algorithm = kmeans.Batch_Float32LloydDense(num_clusters, 0)
    algorithm.input.set(kmeans.data, dataSource.getNumericTable())
    algorithm.input.set(kmeans.inputCentroids, centroidsResult)

    t0 = time.time()
    res2 = algorithm.compute()
    t1 = time.time()
    cluster_time = t1 - t0
    print('cluster time: %f' % cluster_time)

    printNumericTable(
        res2.get(kmeans.centroids), 'First 10 dimensions of centroids:', 20,
        10)

    assignments_table = res2.get(kmeans.assignments)
    assignment_num_rows = assignments_table.getNumberOfRows()

    assignments_block = BlockDescriptor()
    assignments_table.getBlockOfRows(0, assignment_num_rows, readOnly,
                                     assignments_block)
    # assignments numpy array
    assignments_array = assignments_block.getArray()

    centroids_table = res2.get(kmeans.centroids)
    centroids_num_rows = centroids_table.getNumberOfRows()

    centroids_block = BlockDescriptor()
    centroids_table.getBlockOfRows(0, centroids_num_rows, readOnly,
                                   centroids_block)

    t0 = time.time()

    user_to_clusters_fname = os.path.join(clusters_dir,
                                          '%d_user_cluster_ids' % num_clusters)
    with open(user_to_clusters_fname, 'w') as f:
        for i in range(assignments_array.shape[0]):
            print('%d' % int(assignments_array[i][0]), file=f)

    t1 = time.time()
    output_time = t1 - t0
    print('output time: %f' % output_time)


if __name__ == '__main__':
    main()
