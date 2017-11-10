#! /usr/bin/env python

from consts import MODEL_DIR_BASE, TO_RUN, NUM_NUMA_NODES, get_numa_queue
from pathos import multiprocessing
import argparse
import os
import subprocess


def run(run_args):
    numa_queue, training_file, test_file, input_dir, base_name, output_dir, runner = run_args

    if not os.path.isdir(input_dir):
        print("Can't find %s" % input_dir)
        return

    # Fetch corresponding cpu ids for available NUMA node
    cpu_ids = numa_queue.get()
    cmd = [
        'taskset', '-c', cpu_ids, runner, '--input-dir', input_dir,
        '--training-file', training_file, '--test-file', test_file,
        '--output-dir', output_dir, '--base-name', base_name
    ]
    print('Running ' + str(cmd))
    subprocess.call(cmd)
    # Add cpu ids for NUMA node back to queue
    numa_queue.put(cpu_ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    runner = './rmse.py'

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    run_args = []
    numa_queue = get_numa_queue(2)

    for (model_dir, _, (training_file, test_file)) in TO_RUN:
        input_dir = os.path.join(MODEL_DIR_BASE, model_dir)
        base_name = model_dir.replace('/', '-')
        run_args.append((numa_queue, training_file, test_file, input_dir,
                         base_name, output_dir, runner))

    pool = multiprocessing.Pool(NUM_NUMA_NODES * 2)
    pool.map(run, run_args)


if __name__ == '__main__':
    main()
