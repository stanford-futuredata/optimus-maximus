#! /usr/bin/env python

from consts import MODEL_DIR_BASE, TO_RUN, NUM_NUMA_NODES, get_numa_queue
from pathos import multiprocessing
import argparse
import os
import subprocess


def run(run_args):
    numa_queue, num_factors, num_users, num_items, K, scale, \
            input_dir, base_name, output_dir, runner = run_args

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
        str(num_factors), '-s',
        str(scale), '--base-name', base_name
    ]
    print('Running ' + str(cmd))
    subprocess.call(cmd)
    # Add cpu ids for NUMA node back to queue
    numa_queue.put(cpu_ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--stats', dest='stats', action='store_true')
    parser.add_argument('--no-stats', dest='stats', action='store_false')
    parser.set_defaults(stats=False)
    parser.add_argument(
        '--top-K', help='list of comma-separated integers, e.g., 1,5,10,50')
    args = parser.parse_args()

    TOP_K = [int(val) for val in args.top_K.split(',')] if args.top_K else [
        1, 5, 10, 50
    ]

    runner = '../fexipro/fexipro'

    BUILD_COMMAND = 'cd ../fexipro/ && make clean && make && cd -'
    subprocess.call(BUILD_COMMAND, shell=True)

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    run_args = []
    numa_queue = get_numa_queue()

    for (model_dir, (num_factors, num_users, num_items, _, scale),
         _) in TO_RUN:
        input_dir = os.path.join(MODEL_DIR_BASE, model_dir)
        base_name = model_dir.replace('/', '-')
        for K in TOP_K:
            run_args.append((numa_queue, num_factors, num_users, num_items, K,
                             scale, input_dir, base_name, output_dir, runner))

    pool = multiprocessing.ProcessPool(
        NUM_NUMA_NODES)  # Only run 4 jobs at once, since we have 4 NUMA nodes
    pool.map(run, run_args)


if __name__ == '__main__':
    main()
