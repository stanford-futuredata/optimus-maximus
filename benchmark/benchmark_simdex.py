#! /usr/bin/env python

from consts import MODEL_DIR_BASE, TO_RUN, NUM_NUMA_NODES, NUMA_QUEUE
from pathos import multiprocessing
from itertools import product
import os
import subprocess


def run(run_args):
    numa_queue, num_factors, num_users, num_items, K, num_clusters, sample_percentage, \
            num_iters, num_threads, num_bins, input_dir, base_name, output_dir, runner = run_args

    if not os.path.isdir(input_dir):
        print("Can't find %s" % input_dir)
        return

    base_name = os.path.join(output_dir, base_name)

    # Fetch corresponding cpu ids for available NUMA node
    cpu_ids = numa_queue.get()
    cmd = [
        'taskset', '-c', cpu_ids, runner, '-w', input_dir, '-k', str(K), '-m',
        str(num_users), '-n', str(num_items), '-f', str(num_factors), '-c',
        str(num_clusters), '-s', str(sample_percentage), '-i', str(num_iters),
        '-b', str(num_bins), '-t', str(num_threads), '--base-name', base_name
    ]
    print('Running ' + str(cmd))
    subprocess.call(cmd)
    # Add cpu ids for NUMA node back to queue
    numa_queue.put(cpu_ids)


def main():
    TOP_K = [1, 5, 10, 50]
    NUM_THREADS = [1]
    NUM_BINS = [1, 3, 5, 10]
    NUM_CLUSTERS = [64, 128, 256, 512, 1024, 2048, 4096]
    SAMPLE_PERCENTAGES = [10]
    NUM_ITERS = [3]

    runner = '../cpp/simdex'

    output_dir = 'simdex-timing'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    run_args = []

    for (model_dir, (num_factors, num_users, num_items)) in TO_RUN:
        input_dir = os.path.join(MODEL_DIR_BASE, model_dir)
        base_name = model_dir.replace('/', '-')
        for K, num_threads, num_bins, num_clusters, sample_percentage, num_iters in product(
                TOP_K, NUM_THREADS, NUM_BINS, NUM_CLUSTERS, SAMPLE_PERCENTAGES,
                NUM_ITERS):
            run_args.append(
                (NUMA_QUEUE, num_factors, num_users, num_items, K,
                 num_clusters, sample_percentage, num_iters, num_threads,
                 num_bins, input_dir, base_name, output_dir, runner))

    pool = multiprocessing.ProcessPool(
        NUM_NUMA_NODES)  # Only run 4 jobs at once, since we have 4 NUMA nodes
    pool.map(run, run_args)


if __name__ == '__main__':
    main()
