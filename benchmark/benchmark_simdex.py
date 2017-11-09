#! /usr/bin/env python

from consts import MODEL_DIR_BASE, TO_RUN, NUM_NUMA_NODES, get_numa_queue
from pathos import multiprocessing
from itertools import product
import argparse
import os
import subprocess


def run(run_args):
    numa_queue, num_factors, num_users, num_items, K, num_clusters, sample_percentage, \
            num_iters, num_threads, batch_size, sample, user_sample_ratio, input_dir, base_name, output_dir, runner = run_args

    if not os.path.isdir(input_dir):
        print("Can't find %s" % input_dir)
        return

    base_name = os.path.join(output_dir, base_name)

    # Fetch corresponding cpu ids for available NUMA node
    cpu_ids = numa_queue.get()
    cmd = [
        'taskset', '-c', cpu_ids, runner, '-w', input_dir, '-k',
        str(K), '-m',
        str(num_users), '-n',
        str(num_items), '-f',
        str(num_factors), '-c',
        str(num_clusters), '-s',
        str(sample_percentage), '-i',
        str(num_iters), '-b', '1', '-t',
        str(num_threads), '--batch-size',
        str(batch_size), '-x',
        str(user_sample_ratio), '--base-name', base_name
    ]
    if sample:
        cmd += ['--sample']
    print('Running ' + str(cmd))
    subprocess.call(cmd)
    # Add cpu ids for NUMA node back to queue
    numa_queue.put(cpu_ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--sweep', dest='sweep', action='store_true')
    parser.add_argument('--no-sweep', dest='sweep', action='store_false')
    parser.set_defaults(sweep=False)
    parser.add_argument('--stats', dest='stats', action='store_true')
    parser.add_argument('--no-stats', dest='stats', action='store_false')
    parser.set_defaults(stats=False)
    parser.add_argument('--sample', dest='sample', action='store_true')
    parser.add_argument('--no-sample', dest='sample', action='store_false')
    parser.set_defaults(sample=False)
    parser.add_argument('--icc', dest='icc', action='store_true')
    parser.add_argument('--no-icc', dest='icc', action='store_false')
    parser.set_defaults(icc=True)
    parser.add_argument('--mkl', dest='mkl', action='store_true')
    parser.add_argument('--no-mkl', dest='mkl', action='store_false')
    parser.set_defaults(mkl=True)
    parser.add_argument('--naive', dest='naive', action='store_true')
    parser.add_argument('--no-naive', dest='naive', action='store_false')
    parser.set_defaults(naive=False)
    parser.add_argument(
        '--decision-rule', dest='decision_rule', action='store_true')
    parser.add_argument(
        '--no-decision-rule', dest='decision_rule', action='store_false')
    parser.set_defaults(decision_rule=False)
    parser.add_argument('--test-only', dest='test_only', action='store_true')
    parser.add_argument(
        '--no-test-only', dest='test_only', action='store_false')
    parser.set_defaults(test_only=False)
    parser.add_argument(
        '--top-K', help='list of comma-separated integers, e.g., 1,5,10,50')
    parser.add_argument(
        '--batch-sizes',
        help='list of comma-separated integers, e.g., 256,512,1024,2048')
    parser.add_argument(
        '--num-clusters',
        help=
        'list of comma-separated integers, e.g., 1,8,64,128,256,512,1024,2048,4096'
    )
    parser.add_argument(
        '--user-sample-ratios',
        help='list of comma-separated integers, e.g., 0.001,0.005,0.01,0.05,0.1'
    )
    args = parser.parse_args()

    TOP_K = [int(val) for val in args.top_K.split(',')] if args.top_K else [
        1, 5, 10, 50
    ]
    NUM_THREADS = [1]
    USER_SAMPLE_RATIOS = [
        float(val) for val in args.user_sample_ratios.split(',')
    ] if args.user_sample_ratios else [0.0]
    BATCH_SIZES = [
        int(val) for val in args.batch_sizes.split(',')
    ] if args.batch_sizes else [512, 2048] if args.sweep else [4096]
    NUM_CLUSTERS = [
        int(val) for val in args.num_clusters.split(',')
    ] if args.num_clusters else [8, 4096] if args.sweep else None
    SAMPLE_PERCENTAGES = [10]
    NUM_ITERS = [3]

    runner = '../cpp/simdex_stats' if args.stats else '../cpp/simdex'

    BUILD_COMMAND = 'cd ../cpp/ && make clean && make -j5'
    if args.icc:
        BUILD_COMMAND += ' ICC=1'
    if args.mkl:
        BUILD_COMMAND += ' MKL=1'
    if args.naive:
        BUILD_COMMAND += ' NAIVE=1'
    if args.stats:
        BUILD_COMMAND += ' STATS=1'
    if args.decision_rule:
        BUILD_COMMAND += ' RULE=1'
    if args.test_only:
        BUILD_COMMAND += ' TEST_ONLY=1'
    BUILD_COMMAND += ' && cd -'
    subprocess.call(BUILD_COMMAND, shell=True)

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    run_args = []
    numa_queue = get_numa_queue()

    for (model_dir, (num_factors, num_users, num_items, best_num_clusters, _),
         _) in TO_RUN:
        input_dir = os.path.join(MODEL_DIR_BASE, model_dir)
        base_name = model_dir.replace('/', '-')
        for K, num_threads, batch_size, num_clusters, sample_percentage, num_iters, user_sample_ratio in product(
                TOP_K, NUM_THREADS, BATCH_SIZES, NUM_CLUSTERS,
                SAMPLE_PERCENTAGES, NUM_ITERS, USER_SAMPLE_RATIOS):
            run_args.append(
                (numa_queue, num_factors, num_users, num_items, K,
                 num_clusters, sample_percentage, num_iters, num_threads,
                 batch_size, args.sample, user_sample_ratio, input_dir,
                 base_name, output_dir, runner))

    pool = multiprocessing.ProcessPool(
        NUM_NUMA_NODES)  # Only run 4 jobs at once, since we have 4 NUMA nodes
    pool.map(run, run_args)


if __name__ == '__main__':
    main()
