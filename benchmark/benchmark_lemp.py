#! /usr/bin/env python

from consts import MODEL_DIR_BASE, TO_RUN, NUM_NUMA_NODES, get_numa_queue
from pathos import multiprocessing
from itertools import product
import argparse
import os
import time
import subprocess


def run(run_args):
    numa_queue, num_factors, num_users, num_items, K, num_threads, input_dir, \
    base_name, output_dir, runner = run_args

    if not os.path.isdir(input_dir):
        print("Can't find %s" % input_dir)
        return

    user_weights_fname = os.path.join(input_dir, 'user_weights.csv')
    item_weights_fname = os.path.join(input_dir, 'item_weights.csv')

    # Fetch corresponding cpu ids for available NUMA node
    cpu_ids = numa_queue.get()
    cmd = [
        'taskset',
        '-c',
        cpu_ids,
        runner,
        '--method=LEMP_LI',
        '--cacheSizeinKB=2560',
        '--Q^T',
        user_weights_fname,
        '--P',
        item_weights_fname,
        '--r=%d' % num_factors,
        '--m=%d' % num_users,
        '--n=%d' % num_items,
        '--k=%d' % K,
        '--t=%d' % num_threads,
        '--logFile=%s' % os.path.join(output_dir, '%s_timing_%d.csv' %
                                      (base_name, int(time.time() * 1000))),
    ]
    print('Running ' + str(cmd))
    subprocess.call(cmd)
    # Add cpu ids for NUMA node back to queue
    numa_queue.put(cpu_ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', required=True)
    # These flags determine what version of Lemp to use: with/without SIMD,
    # with/without ICC compiler (defaults to g++-4.8)
    parser.add_argument('--simd', dest='simd', action='store_true')
    parser.add_argument('--no-simd', dest='simd', action='store_false')
    parser.set_defaults(simd=True)
    parser.add_argument('--icc', dest='icc', action='store_true')
    parser.add_argument('--no-icc', dest='icc', action='store_false')
    parser.set_defaults(icc=False)
    parser.add_argument(
        '--top_K', help='list of comma-separated integers, e.g., 1,5,10,50')
    args = parser.parse_args()

    TOP_K = [int(val) for val in args.top_K.split(',')] if args.top_K else [
        1, 5, 10, 50
    ]
    NUM_THREADS = [1]

    output_suffix = 'lemp-%s-%s' % (('icc' if args.icc else 'no-icc'),
                                    ('simd' if args.simd else 'no-simd'))
    runner = '../%s/tools/runLemp' % output_suffix

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    run_args = []
    numa_queue = get_numa_queue()

    for (model_dir, (num_factors, num_users, num_items, _, _), _) in TO_RUN:
        input_dir = os.path.join(MODEL_DIR_BASE, model_dir)
        base_name = model_dir.replace('/', '-')
        for K, num_threads in product(TOP_K, NUM_THREADS):
            run_args.append(
                (numa_queue, num_factors, num_users, num_items, K, num_threads,
                 input_dir, base_name, output_dir, runner))

    pool = multiprocessing.Pool(
        NUM_NUMA_NODES)  # Only run 4 jobs at once, since we have 4 NUMA nodes
    pool.map(run, run_args)


if __name__ == '__main__':
    main()
