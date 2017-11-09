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
    numa_queue, K, alg, scaling_value, sigma, sample, user_sample_ratio, input_dir,\
            base_name, output_dir, runner = run_args

    if not os.path.isdir(input_dir):
        print("Can't find %s" % input_dir)
        return

    user_weights_fname = os.path.join(input_dir, 'user_weights.csv')
    item_weights_fname = os.path.join(input_dir, 'item_weights.csv')

    curr_time = int(time.time() * 1000)
    if sample:
        user_weights = np.loadtxt(user_weights_fname, delimiter=',')
        num_users = int(len(user_weights) * 0.001)
        random_user_ids = np.random.choice(
            len(user_weights), num_users, replace=False)
        sampled_user_weights = user_weights[random_user_ids]
        sampled_user_weights_fname = os.path.join(
            input_dir, 'sampled_user_weights_%d.csv' % curr_time)
        np.savetxt(
            sampled_user_weights_fname, sampled_user_weights, delimiter=',')
        user_weights_fname = sampled_user_weights_fname

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
        '--x',
        str(user_sample_ratio),
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
    ]
    print('Running ' + str(cmd))
    subprocess.call(cmd)
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
    parser.add_argument('--sample', dest='sample', action='store_true')
    parser.add_argument('--no-sample', dest='sample', action='store_false')
    parser.set_defaults(sample=False)
    parser.add_argument(
        '--decision-rule', dest='decision_rule', action='store_true')
    parser.add_argument(
        '--no-decision-rule', dest='decision_rule', action='store_false')
    parser.set_defaults(decision_rule=False)
    parser.add_argument('--test-only', dest='test_only', action='store_true')
    parser.add_argument(
        '--no-test-only', dest='test_only', action='store_false')
    parser.add_argument(
        '--user-sample-ratios',
        help='list of comma-separated integers, e.g., 0.001,0.005,0.01,0.05,0.1'
    )
    parser.set_defaults(test_only=False)
    args = parser.parse_args()

    scaling_value = args.scaling_value if args.scaling_value else 127
    sigma = args.sigma if args.sigma else 0.8

    TOP_K = [int(val) for val in args.top_K.split(',')] if args.top_K else [
        1, 5, 10, 50
    ]
    USER_SAMPLE_RATIOS = [
        float(val) for val in args.user_sample_ratios.split(',')
    ] if args.user_sample_ratios else [0.001, 0.005, 0.01, 0.05, 0.1]
    ALGS = ['SI', 'SIR']

    runner_dir = 'fexipro-orig-build'
    if args.decision_rule:
        runner_dir += '-decision-rule'
    if args.test_only:
        runner_dir += '-test-only'
    runner = '../%s/runFEXIPRO' % runner_dir

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
        for K, alg, user_sample_ratio in product(TOP_K, ALGS,
                                                 USER_SAMPLE_RATIOS):
            run_args.append(
                (numa_queue, K, alg, scaling_value, sigma, args.sample,
                 user_sample_ratio, input_dir, base_name, output_dir, runner))

    pool = multiprocessing.Pool(
        NUM_NUMA_NODES)  # Only run 4 jobs at once, since we have 4 NUMA nodes
    pool.map(run, run_args)


if __name__ == '__main__':
    main()
