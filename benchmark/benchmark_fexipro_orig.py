#! /usr/bin/env python

from consts import MODEL_DIR_BASE, TO_RUN, NUM_NUMA_NODES, get_numa_queue
from pathos import multiprocessing
from itertools import product
import argparse
import os
import time
import subprocess


def run(run_args):
    numa_queue, K, alg, scaling_value, sigma, input_dir, base_name, output_dir, runner = run_args

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
        '--alg',
        alg,
        '--scalingValue',
        str(scaling_value),
        '--SIGMA',
        str(sigma),
        '--logPathPrefix',
        output_dir,
        '--resultPathPrefix',
        output_dir + 'result',
        '--dataset',
        base_name,
        '--q',
        user_weights_fname,
        '--p',
        item_weights_fname,
        '--k',
        str(K),
    ]
    print('Running ' + str(cmd))
    subprocess.call(cmd)
    # Add cpu ids for NUMA node back to queue
    numa_queue.put(cpu_ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', required=True)
    parser.add_argument(
        '--scaling_value',
        type=int,
        help='maximum value for scaling in FEXIPRO')
    parser.add_argument(
        '--sigma',
        type=float,
        help='percentage of SIGMA for SVD incremental prune')
    parser.add_argument(
        '--top_K', help='list of comma-separated integers, e.g., 1,5,10,50')
    args = parser.parse_args()

    scaling_value = args.scaling_value if args.scaling_value else 127
    sigma = args.sigma if args.sigma else 0.8

    TOP_K = [int(val) for val in args.top_K.split(',')] if args.top_K else [
        1, 5, 10, 50
    ]
    ALGS = ['SIR', 'SI']

    runner = '../fexipro-orig-build/runFEXIPRO'

    output_dir = args.output_dir
    if output_dir[-1] != '/':
        output_dir += '/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    run_args = []
    numa_queue = get_numa_queue()

    for (model_dir, _, _) in TO_RUN:
        input_dir = os.path.join(MODEL_DIR_BASE, model_dir)
        base_name = model_dir.replace('/', '-')
        for K, alg in product(TOP_K, ALGS):
            run_args.append((numa_queue, K, alg, scaling_value, sigma,
                             input_dir, base_name, output_dir, runner))

    pool = multiprocessing.Pool(
        NUM_NUMA_NODES)  # Only run 4 jobs at once, since we have 4 NUMA nodes
    pool.map(run, run_args)


if __name__ == '__main__':
    main()
