#! /usr/bin/env python

from consts import MODEL_DIR_BASE, TO_RUN, NUM_NUMA_NODES, get_numa_queue
from pathos import multiprocessing
from itertools import product
import argparse
import os
import subprocess


def run(run_args):
    numa_queue, num_factors, num_users, num_items, K, num_clusters, sample_percentage, \
            num_iters, num_threads, batch_size, input_dir, base_name, output_dir, runner = run_args

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
        '-b', '1', '-t', str(num_threads), '--batch-size', str(batch_size),
        '--base-name', base_name
    ]
    print('Running ' + str(cmd))
    subprocess.call(cmd)
    # Add cpu ids for NUMA node back to queue
    numa_queue.put(cpu_ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--sweep', dest='sweep', action='store_true')
    parser.add_argument('--no-sweep', dest='sweep', action='store_false')
    parser.set_defaults(sweep=False)
    parser.add_argument('--stats', dest='stats', action='store_true')
    parser.add_argument('--no-stats', dest='stats', action='store_false')
    parser.set_defaults(stats=False)
    parser.add_argument(
        '--top_K', help='list of comma-separated integers, e.g., 1,5,10,50')
    args = parser.parse_args()

    TOP_K = [int(val) for val in args.top_K.split(',')] if args.top_K else [
        1, 5, 10, 50
    ]
    NUM_THREADS = [1]
    BATCH_SIZES = [256, 512, 1024, 2048, 4096] if args.sweep else [4096]
    NUM_CLUSTERS = [1, 8, 64, 128, 256, 512, 1024, 2048, 4096]
    SAMPLE_PERCENTAGES = [10]
    NUM_ITERS = [3]

    runner = '../cpp/simdex_stats' if args.stats else '../cpp/simdex'

    BUILD_COMMAND = 'cd ../cpp/ && make clean && make -j5'
    if args.stats:
        BUILD_COMMAND += ' STATS=1'
    BUILD_COMMAND += ' && cd -'
    subprocess.call(BUILD_COMMAND, shell=True)

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    run_args = []
    numa_queue = get_numa_queue()

    for (model_dir, (num_factors, num_users, num_items, best_num_clusters),
         _) in TO_RUN:
        input_dir = os.path.join(MODEL_DIR_BASE, model_dir)
        base_name = model_dir.replace('/', '-')
        num_clusters_to_try = NUM_CLUSTERS if args.sweep else best_num_clusters
        for K, num_threads, batch_size, num_clusters, sample_percentage, num_iters in product(
                TOP_K, NUM_THREADS, BATCH_SIZES, num_clusters_to_try,
                SAMPLE_PERCENTAGES, NUM_ITERS):
            run_args.append(
                (numa_queue, num_factors, num_users, num_items, K,
                 num_clusters, sample_percentage, num_iters, num_threads,
                 batch_size, input_dir, base_name, output_dir, runner))

    pool = multiprocessing.ProcessPool(
        NUM_NUMA_NODES)  # Only run 4 jobs at once, since we have 4 NUMA nodes
    pool.map(run, run_args)


if __name__ == '__main__':
    main()
