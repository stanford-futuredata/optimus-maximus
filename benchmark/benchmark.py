#! /usr/bin/env python

from consts import *
from itertools import product
import os
import subprocess
import argparse


def lemp(arg):
    num_factors, num_users, num_items, K, num_threads, input_dir, \
    output_dir, runner = arg

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.isdir(input_dir):
        print("Can't find %s" % input_dir)
        return

    users_in_fname = os.path.join(input_dir, 'user_weights.csv')
    items_in_fname = os.path.join(input_dir, 'item_weights.csv')

    cmd = [
        runner,
        '--method=LEMP_LI',
        '--cacheSizeinKB=2560',
        '--Q^T',
        users_in_fname,
        '--P',
        items_in_fname,
        '--r=%d' % num_factors,
        '--m=%d' % num_users,
        '--n=%d' % num_items,
        '--k=%d' % K,
        '--t=%d' % num_threads,
        '--logFile=%s' % os.path.join(output_dir, 'lemp_timing_stats.txt'),
        '--resultsFile=%s' % os.path.join(output_dir, 'lemp_user_results.csv'),
    ]
    print('Running ' + str(cmd))
    process = subprocess.Popen(cmd)
    process.wait()
    if process.returncode == 1:
        print(str(cmd) + ' failed')
    return process.returncode


def simdex(arg):
    num_factors, num_users, num_items, K, num_clusters, sample_percentage, \
            num_iters, num_threads, num_bins, input_dir, base_name, runner = arg
    output_dir = 'simdex-timing'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    base_name = os.path.join(output_dir, base_name)

    cmd = [
        runner, '-w', input_dir, '-k', str(K), '-m', str(num_users), '-n',
        str(num_items), '-f', str(num_factors), '-c', str(num_clusters), '-s',
        str(sample_percentage), '-i', str(num_iters), '-b', str(num_bins),
        '-t', str(num_threads), '--base-name', base_name
    ]
    print('Running ' + str(cmd))
    process = subprocess.Popen(cmd)
    process.wait()
    if process.returncode == 1:
        print(str(cmd) + ' failed')
    return process.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--simd', dest='simd', action='store_true')
    parser.add_argument('--no-simd', dest='simd', action='store_false')
    parser.set_defaults(simd=True)
    parser.add_argument('--icc', dest='icc', action='store_true')
    parser.add_argument('--no-icc', dest='icc', action='store_false')
    parser.set_defaults(icc=False)
    args = parser.parse_args()

    TOP_K = [1, 50]
    NUM_THREADS = [8, 1]
    NUM_BINS = [1, 3, 5, 10]
    NUM_CLUSTERS = [256, 1024, 4096]
    SAMPLE_PERCENTAGES = [1, 20, 100]
    NUM_ITERS = [1, 10, 25]
    simdex_runner = '../cpp/simdex'

    output_suffix = 'lemp-%s-%s' % (('icc' if args.icc else 'no-icc'),
                                     ('simd' if args.simd else 'no-simd'))
    output_dir_base = os.path.join(OUTPUT_DIR_BASE, output_suffix)
    lemp_runner = '../%s/tools/runLemp' % output_suffix

    for (model_dir, (num_factors, num_users, num_items)) in TO_RUN:
        for K, num_threads in product(TOP_K, NUM_THREADS):
            result = lemp((num_factors, num_users, num_items, K, num_threads,
                           os.path.join(MODEL_DIR_BASE, model_dir), output_dir_base.format(
                               model_dir=model_dir, K=K), lemp_runner))
            print(result)
            for num_bins, num_clusters, sample_percentage, num_iters in product(
                    NUM_BINS, NUM_CLUSTERS, SAMPLE_PERCENTAGES, NUM_ITERS):
                result = simdex(
                    (num_factors, num_users, num_items, K, num_clusters,
                     sample_percentage, num_iters, num_threads,
                     num_bins, os.path.join(MODEL_DIR_BASE, model_dir), model_dir.replace(
                         '/', '-'), simdex_runner))
                print(result)


if __name__ == '__main__':
    main()
