#! /usr/bin/env python

from consts import *
import os
import subprocess
import argparse


def mkdir_p(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


#LEMP/release/tools/runNaive \
#--Q^T=/lfs/raiders6/ssd/geet/lemp/datasets/netflix/f_10/s2_userWeights.csv \
#--P=/lfs/raiders6/ssd/geet/lemp/datasets/netflix/f_10/s_itemWeights.csv \
#--logFile="../../results/log_netflix_${f}_${k}.txt"
#--k=1 --r=10 --m=480189 --n=17770


def fn(arg):
    num_factors, num_users, num_items, K, num_threads, users_in_fname, \
    items_in_fname, output_dir, runner = arg
    mkdir_p(output_dir)
    if not os.path.isfile(users_in_fname):
        print("Can't find %s" % (users_in_fname))
        return
    if not os.path.isfile(items_in_fname):
        print("Can't find %s" % (items_in_fname))
        return

    # ../lemp[-no]-simd/tools/runLemp
    # --Q^T=/dfs/scratch0/fabuzaid/simdex/models/lemp-paper/Netflix-noav-10/user_weights.csv
    # --P=/dfs/scratch0/fabuzaid/simdex/models/lemp-paper/Netflix-noav-10/item_weights.csv
    # --logFile=/dfs/scratch0/fabuzaid/simdex/experiments/lemp-paper/Netflix-noav-10/1/lemp-simd/lemp_timing_stats.txt
    # --resultsFile=/dfs/scratch0/fabuzaid/simdex/experiments/lemp-paper/model-Netflix-noav-10/K-1/lemp-simd/lemp_user_results.csv
    # --k=1 --method=LEMP_LI --cacheSizeinKB=2560 --r=10 --m=480189 --n=17770
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
        '--logFile=%s' % output_dir + 'lemp_timing_stats.txt',
        '--resultsFile=%s' % output_dir + 'lemp_user_results.csv',
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

    #TOP_K = [1, 5, 10, 50]
    TOP_K = [5]
    output_suffix = 'lemp-%s-%s/' % (('icc' if args.icc else 'no-icc'),
                                     ('simd' if args.simd else 'no-simd'))
    output_dir_base = OUTPUT_DIR_BASE + output_suffix
    runner = '../%s/tools/runLemp' % output_suffix

    #run_args = []
    for (model_dir, (num_factors, num_users, num_items)) in TO_RUN:
        for K in TOP_K:
            for num_threads in [1, 2, 4, 8]:
                #run_args.append(
                result = fn((num_factors, num_users, num_items, K, num_threads,
                             MODEL_DIR_BASE + model_dir + 'user_weights.csv',
                             MODEL_DIR_BASE + model_dir + 'item_weights.csv',
                             output_dir_base.format(model_dir=model_dir,
                                                    K=K), runner))
                print(result)

    #from pathos import multiprocessing
    #pool = multiprocessing.Pool(13)
    #results = pool.map(fn, run_args)
    #print(results)


if __name__ == '__main__':
    main()
