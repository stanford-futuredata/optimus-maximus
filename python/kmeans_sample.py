#!/usr/bin/env python

from daal.data_management import (
    readOnly,
    BlockDescriptor,
    HomogenNumericTable_Float32, )
from daal.services import Environment
from daal.algorithms.kmeans import init
from daal.algorithms import kmeans

from daal_utils import printNumericTable
import numpy as np
import argparse
import time
import math
import os


# Set the maximum number of threads to be used by the library
def set_num_threads(num_threads):
    Environment.getInstance().setNumberOfThreads(num_threads)
    num_threads_new = Environment.getInstance().getNumberOfThreads()
    print('max number of threads: %d' % num_threads_new)


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

    input_dir = args.weights_dir
    num_clusters = args.num_clusters
    num_iters = args.num_iters
    sample_size = args.sample_size
    set_num_threads(args.num_threads)
    output_dir = os.path.join(args.output_dir_base,
                              str(sample_size), str(num_iters))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    input_data = np.loadtxt(
        os.path.join(input_dir, 'user_weights.csv'),
        dtype=np.float32,
        delimiter=',')

    samples = int(math.floor(input_data.shape[0] * (sample_size / 100)))
    random_indices = np.random.choice(input_data.shape[0], samples)
    sampled_data = input_data[random_indices]

    t0 = time.time()
    full_data_table = HomogenNumericTable_Float32(input_data)
    sample_data_table = HomogenNumericTable_Float32(sampled_data)
    initAlg = init.Batch_Float32RandomDense(num_clusters)
    initAlg.input.set(init.data, sample_data_table)
    t1 = time.time()
    init_time = t1 - t0
    print('init time: %f' % init_time)

    t0 = time.time()
    res = initAlg.compute()
    t1 = time.time()
    centroid_time = t1 - t0
    print('centroid time: %f' % centroid_time)

    centroidsResult = res.get(init.centroids)
    algorithm = kmeans.Batch_Float32LloydDense(num_clusters, num_iters)
    algorithm.input.set(kmeans.data, sample_data_table)
    algorithm.input.set(kmeans.inputCentroids, centroidsResult)

    t0 = time.time()
    res2 = algorithm.compute()
    t1 = time.time()
    cluster_time = t1 - t0
    print('cluster time: %f' % cluster_time)

    printNumericTable(
        res2.get(kmeans.centroids), 'First 10 dimensions of centroids:', 20,
        10)

    centroids_table = res2.get(kmeans.centroids)
    centroids_num_rows = centroids_table.getNumberOfRows()

    centroids_block = BlockDescriptor()
    centroids_table.getBlockOfRows(0, centroids_num_rows, readOnly,
                                   centroids_block)
    centroids = centroids_block.getArray()

    initAlg.input.set(init.data, full_data_table)

    assignments_table = res2.get(kmeans.assignments)
    assignment_num_rows = assignments_table.getNumberOfRows()

    assignments_block = BlockDescriptor()
    assignments_table.getBlockOfRows(0, assignment_num_rows, readOnly,
                                     assignments_block)
    assignments = assignments_block.getArray()

    np.savetxt(
        os.path.join(output_dir, '%d_centroids.csv' % num_clusters),
        centroids,
        delimiter=',')
    np.savetxt(
        os.path.join(output_dir, '%d_user_cluster_ids' % num_clusters),
        assignments,
        fmt='%d',
        delimiter='\n')
    cluster_time_fname = os.path.join(
        output_dir, 'cluster_time_u%d_f%d_c%d.csv' %
        (input_data.shape[0], input_data.shape[1], num_clusters))

    with open(cluster_time_fname, 'w') as f:
        header = 'FirstPassClusterTime,TotalTime,Sample%,Samples\n'
        f.write(header)
        outlist = [
            str(cluster_time)[:5],
            str(init_time + centroid_time + cluster_time)[:5],
            str(sample_size),
            str(samples)
        ]
        outstring = ','.join(outlist)
        outstring = outstring + '\n'
        f.write(outstring)


if __name__ == '__main__':
    main()
