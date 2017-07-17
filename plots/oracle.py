#! /usr/bin/env python

from __future__ import print_function, division
import pprint
import numpy as np
import pandas as pd
from itertools import product
from models import GOLD_STANDARD_MODELS


pp = pprint.PrettyPrinter(indent=4, width=160)

def load_dataframes():
    sampling_df = pd.read_csv('timing-results/sampling-time.csv')

    simdex_netflix_df = pd.read_csv('timing-results/netflix-simdex-timing.csv')
    lemp_netflix_df = pd.read_csv('timing-results/netflix-lemp-timing.csv')
    blocked_mm_netflix_df = pd.read_csv(
        'timing-results/netflix-blocked_mm-timing.csv')

    simdex_kdd_df = pd.read_csv('timing-results/kdd-simdex-timing.csv')
    lemp_kdd_df = pd.read_csv('timing-results/kdd-lemp-timing.csv')
    blocked_mm_kdd_df = pd.read_csv('timing-results/kdd-blocked_mm-timing.csv')

    simdex_r2_df = pd.read_csv('timing-results/r2-simdex-timing.csv')
    lemp_r2_df = pd.read_csv('timing-results/r2-lemp-timing.csv')
    blocked_mm_r2_df = pd.read_csv('timing-results/r2-blocked_mm-timing.csv')

    simdex_df = pd.concat([simdex_netflix_df, simdex_kdd_df, simdex_r2_df])
    lemp_df = pd.concat([lemp_netflix_df, lemp_kdd_df, lemp_r2_df])
    blocked_mm_df = pd.concat(
        [blocked_mm_netflix_df, blocked_mm_kdd_df, blocked_mm_r2_df])

    return (simdex_df, sampling_df, blocked_mm_df, lemp_df)


def analyze_oracle(simdex_df,
                   sampling_df,
                   blocked_mm_df,
                   lemp_df,
                   models,
                   num_clusters=8,
                   batch_size=4096,
                   K=[1, 5, 10, 50]):
    simdex_df = simdex_df.query('num_clusters == %d and batch_size == %d' %
                                (num_clusters, batch_size))
    blocked_mm_df = blocked_mm_df[['model', 'K', 'comp_time']]
    lemp_df = lemp_df[['model', 'K', 'comp_time']]

    speedups_dict = {}
    simdex_vs_lemp_dict = {}
    for model, K in product(models, K):
        lemp_model_K_df = lemp_df.query('model == "%s" and K == "%d"' % (model,
                                                                         K))
        simdex_model_K_df = simdex_df.query('model == "%s" and K == "%d"' %
                                            (model, K))
        blocked_mm_model_K_df = blocked_mm_df.query(
            'model == "%s" and K == "%d"' % (model, K))
        sampling_model_K_df = sampling_df.query('model == "%s" and K == "%d"' % (model, K))

        simdex_rt = simdex_model_K_df['comp_time'].min()
        lemp_rt = lemp_model_K_df['comp_time'].min()
        blocked_mm_rt = blocked_mm_model_K_df['comp_time'].min()

        # add sampling overhead
        overhead = sampling_model_K_df['comp_time'].min()

        blocked_mm_lemp_rt = min(lemp_rt, blocked_mm_rt)
        blocked_mm_simdex_rt = min(simdex_rt, blocked_mm_rt)

        global_speed_up = blocked_mm_lemp_rt / blocked_mm_simdex_rt
        simdex_speed_up = lemp_rt / simdex_rt
        speedups_dict[(model, K)] = global_speed_up
        simdex_vs_lemp_dict[(model, K)] = simdex_speed_up

    all_speedups = list(speedups_dict.values())
    simdex_speedups = list(simdex_vs_lemp_dict.values())

    print('Global Speedups: Min: %f, Max: %f, Avg: %f' %
          (np.min(all_speedups), np.max(all_speedups), np.mean(all_speedups)))
    print('SimDex Speedups: Min: %f, Max: %f, Avg: %f' %
          (np.min(simdex_speedups), np.max(simdex_speedups), np.mean(simdex_speedups)))


def main():
    simdex_df, sampling_df, blocked_mm_df, lemp_df = load_dataframes()
    analyze_oracle(simdex_df, sampling_df, blocked_mm_df, lemp_df, GOLD_STANDARD_MODELS)

if __name__ == '__main__':
    main()
