#! /usr/bin/env python

from consts import *
import subprocess
from itertools import product


def fn(arg):
    num_factors, num_users, num_items, K, num_clusters, num_threads, num_bins, input_dir, \
    base_name, runner = arg

    cmd = [
        runner, '-w', input_dir, '-d', input_dir + '/clustering', '-k', str(K),
        '-m', str(num_users), '-n', str(num_items), '-f', str(num_factors),
        '-c', str(num_clusters), '-s', '20', '-i', '10', '-b', str(num_bins),
        '-t', str(num_threads), '--base-name', base_name.replace('/', '-')
    ]
    print('Running ' + str(cmd))
    process = subprocess.Popen(cmd)
    process.wait()
    if process.returncode == 1:
        print(str(cmd) + ' failed')
    return process.returncode


def main():
    TOP_K = [1, 50]
    NUM_THREADS = [8, 1]
    NUM_BINS = [1, 10]
    NUM_CLUSTERS = [256, 512, 1024, 2048, 4096]
    runner = '../cpp/simdex'

    for (model_dir, (num_factors, num_users, num_items)) in TO_RUN:
        for K, num_threads, num_bins, num_clusters in product(
                TOP_K, NUM_THREADS, NUM_BINS, NUM_CLUSTERS):
            result = fn((num_factors, num_users, num_items, K, num_clusters,
                         num_threads, num_bins, MODEL_DIR_BASE + model_dir,
                         model_dir, runner))
            print(result)


if __name__ == '__main__':
    main()
