#!/usr/bin/env python

import pandas as pd
import numpy as np


def compute_preproc_perc_stats(df, preproc_cols, comp_cols):

    preproc_perc = []
    for _, row in df.iterrows():

        preproc_time, comp_time = 0, 0
        for col in preproc_cols:
            preproc_time += row[col]
        for col in comp_cols:
            comp_time += row[col]

        preproc_perc.append(preproc_time / comp_time)
    return np.mean(preproc_perc), np.std(preproc_perc)


fexipro_truth = pd.read_csv('timing-results/fexipro-orig-timing.csv')

lemp_truth = pd.read_csv('timing-results/lemp-gold-standard-timing.csv')

simdex_truth_netflix = pd.read_csv('timing-results/netflix-simdex-timing.csv')
simdex_truth_kdd = pd.read_csv('timing-results/kdd-simdex-timing.csv')
simdex_truth_r2 = pd.read_csv('timing-results/r2-simdex-timing.csv')
simdex_truth = pd.concat(
    [simdex_truth_netflix, simdex_truth_kdd,
     simdex_truth_r2]).query('num_clusters <= 8')

fexipro_si_stats = compute_preproc_perc_stats(
    fexipro_truth.query('alg == "SI"'), ['preproc_time'], ['comp_time'])
print('FEXIPRO-SI average/std dev. preproc percentage: ', fexipro_si_stats)

fexipro_sir_stats = compute_preproc_perc_stats(
    fexipro_truth.query('alg == "SIR"'), ['preproc_time'], ['comp_time'])
print('FEXIPRO-SIR average/std dev. preproc percentage: ', fexipro_sir_stats)

lemp_stats = compute_preproc_perc_stats(
    lemp_truth, ['preproc_time', 'index_time'], ['comp_time'])
print('LEMP average preproc/std dev. percentage: ', lemp_stats)

simdex_stats = compute_preproc_perc_stats(
    simdex_truth, ['cluster_time', 'index_time'], ['comp_time'])
print('SimDex average preproc/std dev. percentage: ', simdex_stats)

print('Overal average: ',
      np.mean([
          fexipro_si_stats[0], fexipro_sir_stats[0], lemp_stats[0],
          simdex_stats[0]
      ]))
