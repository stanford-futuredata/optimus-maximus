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
                   K=[1, 5, 10, 50]):
    #simdex_df = simdex_df.query('num_clusters == 8 and batch_size == 4096')
    simdex_df = simdex_df.sort_values(by='comp_time').groupby(
        ['model', 'K'], as_index=False).first()

    blocked_mm_df = blocked_mm_df[['model', 'K', 'comp_time']]
    lemp_df = lemp_df[['model', 'K', 'comp_time']]

    simdex_mm_vs_lemp_dict = {}
    simdex_mm_vs_lemp_mm_dict = {}
    simdex_vs_lemp_mm_dict = {}
    simdex_vs_lemp_dict = {}
    for model, K in product(models, K):
        lemp_model_K_df = lemp_df.query('model == "%s" and K == "%d"' % (model,
                                                                         K))
        simdex_model_K_df = simdex_df.query('model == "%s" and K == "%d"' %
                                            (model, K))
        blocked_mm_model_K_df = blocked_mm_df.query(
            'model == "%s" and K == "%d"' % (model, K))
        sampling_model_K_df = sampling_df.query('model == "%s" and K == "%d"' %
                                                (model, K))

        simdex_rt = simdex_model_K_df['comp_time'].min()
        lemp_rt = lemp_model_K_df['comp_time'].min()
        blocked_mm_rt = blocked_mm_model_K_df['comp_time'].min()

        # add sampling overhead
        overhead = sampling_model_K_df['comp_time'].min()

        blocked_mm_lemp_rt = min(lemp_rt, blocked_mm_rt)
        blocked_mm_simdex_rt = min(simdex_rt, blocked_mm_rt) + overhead

        global_speed_up = blocked_mm_lemp_rt / blocked_mm_simdex_rt
        simdex_vs_lemp_mm_speed_up = blocked_mm_lemp_rt / simdex_rt
        simdex_mm_speed_up = lemp_rt / blocked_mm_simdex_rt
        simdex_speed_up = lemp_rt / simdex_rt

        simdex_mm_vs_lemp_mm_dict[(model, K)] = global_speed_up
        simdex_vs_lemp_mm_dict[(model, K)] = simdex_vs_lemp_mm_speed_up
        simdex_mm_vs_lemp_dict[(model, K)] = simdex_mm_speed_up
        simdex_vs_lemp_dict[(model, K)] = simdex_speed_up

    pp.pprint(simdex_mm_vs_lemp_mm_dict)

    simdex_mm_vs_lemp_mm = list(simdex_mm_vs_lemp_mm_dict.values())
    simdex_mm_vs_lemp = list(simdex_mm_vs_lemp_dict.values())
    simdex_vs_lemp_mm = list(simdex_vs_lemp_mm_dict.values())
    simdex_vs_lemp = list(simdex_vs_lemp_dict.values())

    print('SimDex + MM vs Lemp: Min: %f, Max: %f, Avg: %f, Median: %f' %
          (np.min(simdex_mm_vs_lemp), np.max(simdex_mm_vs_lemp),
           np.mean(simdex_mm_vs_lemp), np.median(simdex_mm_vs_lemp)))
    print('SimDex + MM vs Lemp + MM: Min: %f, Max: %f, Avg: %f, Median: %f' %
          (np.min(simdex_mm_vs_lemp_mm), np.max(simdex_mm_vs_lemp_mm),
           np.mean(simdex_mm_vs_lemp_mm), np.median(simdex_mm_vs_lemp_mm)))
    print('SimDex vs Lemp + MM: Min: %f, Max: %f, Avg: %f, Median: %f' %
          (np.min(simdex_vs_lemp_mm), np.max(simdex_vs_lemp_mm),
           np.mean(simdex_vs_lemp_mm), np.median(simdex_vs_lemp_mm)))
    print('SimDex vs Lemp: Min: %f, Max: %f, Avg: %f, Median: %f' %
          (np.min(simdex_vs_lemp), np.max(simdex_vs_lemp),
           np.mean(simdex_vs_lemp), np.median(simdex_vs_lemp)))


def main():
    simdex_df, sampling_df, blocked_mm_df, lemp_df = load_dataframes()
    analyze_oracle(simdex_df, sampling_df, blocked_mm_df, lemp_df,
                   GOLD_STANDARD_MODELS)


if __name__ == '__main__':
    main()
