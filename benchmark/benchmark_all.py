#! /usr/bin/env python

from consts import MODEL_DIR_BASE, TO_RUN, OUTPUT_DIR_BASE
from pathos import multiprocessing
from itertools import product
import os
import time
import subprocess
import argparse


def run_simdex(run_args):
    cpu_ids, num_factors, num_users, num_items, K, num_clusters, sample_percentage, \
            num_iters, num_threads, num_bins, input_dir, base_name, runner = run_args

    if not os.path.isdir(input_dir):
        print("Can't find %s" % input_dir)
        return

    output_dir = 'simdex-timing'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    base_name = os.path.join(output_dir, base_name)

    cmd = [
        'taskset', '-c', cpu_ids, runner, '-w', input_dir, '-k', str(K), '-m',
        str(num_users), '-n', str(num_items), '-f', str(num_factors), '-c',
        str(num_clusters), '-s', str(sample_percentage), '-i', str(num_iters),
        '-b', str(num_bins), '-t', str(num_threads), '--base-name', base_name
    ]
    print('Running ' + str(cmd))
    process = subprocess.Popen(cmd)
    process.wait()
    if process.returncode == 1:
        print(str(cmd) + ' failed')
    return process.returncode


def run_blocked_mm(run_args):
    cpu_ids, num_factors, num_users, num_items, K, num_threads, input_dir, \
    base_name, runner = run_args

    if not os.path.isdir(input_dir):
        print("Can't find %s" % input_dir)
        return

    output_dir = 'blocked_mm-timing'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    base_name = os.path.join(output_dir, base_name)

    cmd = [
        'taskset', '-c', cpu_ids, runner, '-w', input_dir, '-k', str(K), '-m',
        str(num_users), '-n', str(num_items), '-f', str(num_factors), '-t',
        str(num_threads), '--base-name', base_name
    ]
    print('Running ' + str(cmd))
    process = subprocess.Popen(cmd)
    process.wait()
    if process.returncode == 1:
        print(str(cmd) + ' failed')
    return process.returncode


def run_lemp(run_args):
    cpu_ids, num_factors, num_users, num_items, K, num_threads, input_dir, \
    output_dir, runner = run_args

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.isdir(input_dir):
        print("Can't find %s" % input_dir)
        return

    users_in_fname = os.path.join(input_dir, 'user_weights.csv')
    items_in_fname = os.path.join(input_dir, 'item_weights.csv')

    cmd = [
        'taskset',
        '-c',
        cpu_ids,
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


# We want only physical cores, not virtual cores. This means the size of our
# pool is NUM_VIRTUAL_CORES_PER_POOL / 2
NUM_VIRTUAL_CORES_PER_POOL = 14


# Since we want only physical cores, we grab every other cpu id in the range
# [cpu_id_offset, cpu_id_offset + num_cores)
def get_cpu_assignments(cpu_id_offset, num_cores):
    return ','.join(
        str(v) for v in range(cpu_id_offset, cpu_id_offset + num_cores, 2))


def main():
    parser = argparse.ArgumentParser()
    # These flags determine what version of Lemp to use: with/without SIMD,
    # with/without ICC compiler (defaults to g++-4.8)
    parser.add_argument('--simd', dest='simd', action='store_true')
    parser.add_argument('--no-simd', dest='simd', action='store_false')
    parser.set_defaults(simd=True)
    parser.add_argument('--icc', dest='icc', action='store_true')
    parser.add_argument('--no-icc', dest='icc', action='store_false')
    parser.set_defaults(icc=False)
    args = parser.parse_args()

    # From `lscpu` on raiders6:
    # CPU(s):                112
    # On-line CPU(s) list:   0-111
    # Thread(s) per core:    2
    # Core(s) per socket:    14
    # Socket(s):             4
    # NUMA node(s):          4
    # NUMA node0 CPU(s):     0-13,56-69
    # NUMA node1 CPU(s):     14-27,70-83
    # NUMA node2 CPU(s):     28-41,84-97
    # NUMA node3 CPU(s):     42-55,98-111

    # We'll use NUMA node 2 for Simdex, NUMA node 3 for Lemp
    # Use the upper half of CPU IDs for each
    SIMDEX_CPU_ID_OFFSET = 70  # Simdex has CPU IDs 70-83
    BLOCKED_MM_CPU_ID_OFFSET = 84  # Blocked MM has CPU IDs 84-97
    LEMP_CPU_ID_OFFSET = 98  # Lemp has CPU IDs 98-111

    simdex_cpu_ids = get_cpu_assignments(SIMDEX_CPU_ID_OFFSET,
                                         NUM_VIRTUAL_CORES_PER_POOL)
    blocked_mm_cpu_ids = get_cpu_assignments(BLOCKED_MM_CPU_ID_OFFSET,
                                             NUM_VIRTUAL_CORES_PER_POOL)
    lemp_cpu_ids = get_cpu_assignments(LEMP_CPU_ID_OFFSET,
                                       NUM_VIRTUAL_CORES_PER_POOL)

    TOP_K = [1, 50]
    NUM_THREADS = [8, 1]
    NUM_BINS = [1, 3, 5, 10]
    NUM_CLUSTERS = [256, 1024, 4096]
    SAMPLE_PERCENTAGES = [1, 20, 100]
    NUM_ITERS = [1, 10, 25]

    simdex_runner = '../cpp/simdex'

    blocked_mm_runner = '../cpp/blocked_mm/blocked_mm'

    output_suffix = 'lemp-%s-%s' % (('icc' if args.icc else 'no-icc'),
                                    ('simd' if args.simd else 'no-simd'))
    output_dir_base = os.path.join(OUTPUT_DIR_BASE, output_suffix)
    lemp_runner = '../%s/tools/runLemp' % output_suffix

    simdex_run_args, blocked_mm_run_args, lemp_run_args = [], [], []
    for (model_dir, (num_factors, num_users, num_items)) in TO_RUN:
        for K, num_threads in product(TOP_K, NUM_THREADS):
            lemp_run_args.append(
                (lemp_cpu_ids, num_factors, num_users,
                 num_items, K, num_threads, os.path.join(
                     MODEL_DIR_BASE, model_dir), output_dir_base.format(
                         model_dir=model_dir, K=K), lemp_runner))
            blocked_mm_run_args.append(
                (blocked_mm_cpu_ids, num_factors, num_users, num_items, K,
                 num_threads, os.path.join(MODEL_DIR_BASE,
                                           model_dir), model_dir.replace(
                                               '/', '-'), blocked_mm_runner))
            for num_bins, num_clusters, sample_percentage, num_iters in product(
                    NUM_BINS, NUM_CLUSTERS, SAMPLE_PERCENTAGES, NUM_ITERS):
                simdex_run_args.append(
                    (simdex_cpu_ids, num_factors, num_users, num_items, K,
                     num_clusters, sample_percentage, num_iters,
                     num_threads, num_bins, os.path.join(
                         MODEL_DIR_BASE, model_dir), model_dir.replace(
                             '/', '-'), simdex_runner))

    simdex_pool = multiprocessing.ProcessPool(int(
        NUM_VIRTUAL_CORES_PER_POOL / 2))  # 7 cores for SimDex
    blocked_mm_pool = multiprocessing.ProcessPool(
        int(NUM_VIRTUAL_CORES_PER_POOL / 2))  # 7 cores for Blocked MM
    lemp_pool = multiprocessing.ProcessPool(int(
        NUM_VIRTUAL_CORES_PER_POOL / 2))  # 7 cores for Lemp

    simdex_results = simdex_pool.amap(run_simdex, simdex_run_args)
    blocked_mm_results = blocked_mm_pool.amap(run_blocked_mm,
                                              blocked_mm_run_args)
    lemp_results = lemp_pool.amap(run_lemp, lemp_run_args)
    while not (simdex_results.ready() and blocked_mm_results.ready() and
               lemp_results.ready()):
        time.sleep(5)
        print('.', end='')


if __name__ == '__main__':
    main()
