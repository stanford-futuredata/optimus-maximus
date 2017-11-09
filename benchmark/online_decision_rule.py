#! /usr/bin/env python

from consts import MODEL_DIR_BASE, TO_RUN, NUM_NUMA_NODES, get_numa_queue
from pathos import multiprocessing
from itertools import product
import numpy as np
import argparse
import os
import time
import subprocess


def run(run_args):
    numa_queue, K, alg, scaling_value, sigma, num_factors, num_items, sample_size, \
    run_fexipro, input_dir, base_name, output_dir, blocked_mm_runner, other_runner = run_args

    if not os.path.isdir(input_dir):
        print("Can't find %s" % input_dir)
        return

    full_base_name = os.path.join(output_dir, base_name)

    user_weights_fname = os.path.join(input_dir, 'user_weights.csv')
    item_weights_fname = os.path.join(input_dir, 'item_weights.csv')

    curr_time = int(time.time() * 1000)
    user_weights = np.loadtxt(user_weights_fname, delimiter=',')
    num_users = int(sample_size)
    random_user_ids = np.random.choice(
        len(user_weights), num_users, replace=False)
    sampled_user_weights = user_weights[random_user_ids]
    sampled_user_weights_fname = os.path.join(
        input_dir, 'sampled_user_weights_%d-users_%d.csv' % (num_users,
                                                             curr_time))
    np.savetxt(sampled_user_weights_fname, sampled_user_weights, delimiter=',')
    user_weights_fname = sampled_user_weights_fname

    # Fetch corresponding cpu ids for available NUMA node
    cpu_ids = numa_queue.get()
    blocked_mm_cmd = [
        blocked_mm_runner, '-q', user_weights_fname, '-p', item_weights_fname,
        '-k',
        str(K), '-m',
        str(num_users), '-n',
        str(num_items), '-f',
        str(num_factors), '-t', '1', '--base-name', full_base_name
    ]
    print('Running ' + str(blocked_mm_cmd))
    subprocess.call(blocked_mm_cmd)
    other_cmd = [
        'taskset',
        '-c',
        cpu_ids,
        other_runner,
        '--alg',
        alg,
        '--scalingValue',
        str(scaling_value),
        '--SIGMA',
        str(sigma),
        '--logPathPrefix',
        output_dir,
        '--outputResult',
        'false',
        '--dataset',
        base_name,
        '--q',
        user_weights_fname,
        '--p',
        item_weights_fname,
        '--k',
        str(K),
    ] if run_fexipro else [
        'taskset',
        '-c',
        cpu_ids,
        other_runner,
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
        '--t=1',
        '--logFile=%s' % os.path.join(output_dir, '%s_timing_K-%d_%d.csv' %
                                      (base_name, K, curr_time)),
    ]

    print('Running ' + str(other_cmd))
    subprocess.call(other_cmd)
    # Add cpu ids for NUMA node back to queue
    numa_queue.put(cpu_ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True)
    parser.add_argument(
        '--scaling-value',
        type=int,
        help='maximum value for scaling in FEXIPRO')
    parser.add_argument(
        '--sigma',
        type=float,
        help='percentage of SIGMA for SVD incremental prune')
    parser.add_argument(
        '--top-K', help='list of comma-separated integers, e.g., 1,5,10,50')
    parser.add_argument('--sample-size', help='number of users to sample')
    parser.add_argument('--icc', dest='icc', action='store_true')
    parser.add_argument('--no-icc', dest='icc', action='store_false')
    parser.set_defaults(icc=False)
    parser.add_argument('--mkl', dest='mkl', action='store_true')
    parser.add_argument('--no-mkl', dest='mkl', action='store_false')
    parser.set_defaults(mkl=False)
    parser.add_argument('--fexipro', dest='run_fexipro', action='store_true')
    parser.add_argument('--lemp', dest='run_fexipro', action='store_false')
    parser.set_defaults(run_fexipro=True)
    args = parser.parse_args()

    scaling_value = args.scaling_value if args.scaling_value else 127
    sigma = args.sigma if args.sigma else 0.8
    sample_size = args.sample_size if args.sample_size else 1000

    TOP_K = [int(val) for val in args.top_K.split(',')] if args.top_K else [
        1, 5, 10, 50
    ]
    ALGS = ['SIR', 'SI']

    blocked_mm_runner = '../cpp/blocked_mm/blocked_mm'
    BUILD_COMMAND = 'cd ../cpp/blocked_mm && make clean && make -j2'
    if args.icc:
        BUILD_COMMAND += ' ICC=1'
    if args.mkl:
        BUILD_COMMAND += ' MKL=1'
    BUILD_COMMAND += ' && cd -'
    subprocess.call(BUILD_COMMAND, shell=True)

    other_runner = '../fexipro-orig-build/runFEXIPRO' if args.run_fexipro else '../lemp-no-icc-simd/tools/runLemp'

    output_dir = args.output_dir
    if output_dir[-1] != '/': output_dir += '/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    run_args = []
    numa_queue = get_numa_queue()

    for (model_dir, (num_factors, _, num_items, _, _), _) in TO_RUN:
        input_dir = os.path.join(MODEL_DIR_BASE, model_dir)
        base_name = model_dir.replace('/', '-')
        for K, alg in product(TOP_K, ALGS):
            run_args.append(
                (numa_queue, K, alg, scaling_value, sigma, num_factors,
                 num_items, sample_size, args.run_fexipro, input_dir,
                 base_name, output_dir, blocked_mm_runner, other_runner))

    pool = multiprocessing.Pool(
        NUM_NUMA_NODES)  # Only run 4 jobs at once, since we have 4 NUMA nodes
    pool.map(run, run_args)


if __name__ == '__main__':
    main()
