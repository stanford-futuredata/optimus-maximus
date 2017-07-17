#! /usr/bin/env python

from __future__ import print_function, division
import numpy as np
import pandas as pd
from itertools import product

def load_dataframes():
    sampling_df = pd.read_csv('timing-results/sampling-time.csv')

    simdex_netflix_df = pd.read_csv('timing-results/netflix-simdex-timing.csv')
    lemp_netflix_df = pd.read_csv('timing-results/netflix-lemp-timing.csv')
    blocked_mm_netflix_df = pd.read_csv('timing-results/netflix-blocked_mm-timing.csv')

    simdex_kdd_df = pd.read_csv('timing-results/kdd-simdex-timing.csv')
    lemp_kdd_df = pd.read_csv('timing-results/kdd-lemp-timing.csv')
    blocked_mm_kdd_df = pd.read_csv('timing-results/kdd-blocked_mm-timing.csv')

    simdex_r2_df = pd.read_csv('timing-results/r2-simdex-timing.csv')
    lemp_r2_df = pd.read_csv('timing-results/r2-lemp-timing.csv')
    blocked_mm_r2_df = pd.read_csv('timing-results/r2-blocked_mm-timing.csv')

    simdex_df = pd.concat([simdex_netflix_df, simdex_kdd_df, simdex_r2_df])
    lemp_df = pd.concat([lemp_netflix_df, lemp_kdd_df, lemp_r2_df])
    blocked_mm_df = pd.concat([blocked_mm_netflix_df, blocked_mm_kdd_df, blocked_mm_r2_df])

    return (simdex_df, sampling_df, blocked_mm_df, lemp_df)

def analyze_oracle(simdex_df, sampling_df, blocked_mm_df, lemp_df, models, num_clusters=8,
        batch_size=4096, K=[1, 5, 10, 50]):
    simdex_rt = simdex_df.query('num_clusters == %d and batch_size == %d' %
            (num_clusters, batch_size))
    blocked_mm_rt = blocked_mm_df[['model', 'K', 'comp_time']]
    lemp_rt = lemp_df[['model', 'K', 'comp_time']]

    all_speedups = []
    for model, K in product(models, K):
        lemp_model_K_rt = lemp_rt.query('model == "%s" and K == "%d"' % (model, K))
        simdex_model_K_rt = simdex_rt.query('model == "%s" and K == "%d"' % (model, K))
        blocked_mm_model_K_rt = blocked_mm_rt.query('model == "%s" and K == "%d"' % (model, K))
        # sampling_model_rt = sampling_df.query('model == "%s"' % model).sort_values(by='K')

        # add sampling overhead
        #overheads = simdex_model_rt['cluster_time'].min() + \
        #    simdex_model_rt['index_time'].min() + \
        #    sampling_model_rt['comp_time'].min()

        blocked_mm_lemp = min(lemp_model_K_rt['comp_time'].min(), blocked_mm_model_K_rt['comp_time'].min())
        blocked_mm_simdex = min(simdex_model_K_rt['comp_time'].min(), blocked_mm_model_K_rt['comp_time'].min())

        speed_up = blocked_mm_simdex / blocked_mm_lemp
        all_speedups.append(speed_up)

    print('Min: %f, Max: %f, Avg: %f' % (np.min(all_speedups), np.max(all_speedups), np.mean(all_speedups)))

def main():
    simdex_df, sampling_df, blocked_mm_df, lemp_df = load_dataframes()
    analyze_oracle(simdex_df, sampling_df, blocked_mm_df, lemp_df)
