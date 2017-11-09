#!/usr/bin/env python

import pandas as pd
import numpy as np

#L2_CACHE_SIZE = 256000
#lemp_dec_rule = pd.read_csv('lemp-decision-rule.csv') # only to query for number of users

simdex_dec_rule = pd.read_csv('simdex-decision-rule.csv')

simdex_truth_netflix = pd.read_csv(
    '../plots/timing-results/netflix-simdex-timing.csv')
simdex_truth_kdd = pd.read_csv('../plots/timing-results/kdd-simdex-timing.csv')
simdex_truth_r2 = pd.read_csv('../plots/timing-results/r2-simdex-timing.csv')
simdex_truth = pd.concat(
    [simdex_truth_netflix, simdex_truth_kdd, simdex_truth_r2])

blocked_mm_truth_netflix = pd.read_csv(
    '../plots/timing-results/netflix-blocked_mm-timing.csv')
blocked_mm_truth_kdd = pd.read_csv(
    '../plots/timing-results/kdd-blocked_mm-timing.csv')
blocked_mm_truth_r2 = pd.read_csv(
    '../plots/timing-results/r2-blocked_mm-timing.csv')
blocked_mm_truth = pd.concat(
    [blocked_mm_truth_netflix, blocked_mm_truth_kdd, blocked_mm_truth_r2])

results = {
    'model': [],
    'K': [],
    'blocked_mm_sample_time': [],
    'simdex_sample_time': [],
    'true_rt_delta': [],
    'true_rt_delta_percentage': [],
    'correct': [],
    'overhead': [],
}

optimizer_vs_optimal = []
optimizer_vs_simdex = []
optimal_vs_simdex = []

for _, row in simdex_dec_rule.iterrows():
    model, K = row['model'], row['K']

    blocked_mm_sample_time, simdex_sample_time = row[
        'blocked_mm_sample_time'], row['simdex_sample_time']
    simdex_index_time, simdex_clustering_time = row['index_time'], row[
        'cluster_time']

    simdex_true_rt = simdex_truth.query('model == "%s" and K == %d' %
                                        (model, K))['comp_time'].min()

    blocked_mm_true_rt = blocked_mm_truth.query('model == "%s" and K == %d' %
                                                (model, K))['comp_time'].min()

    #sample_num_users = int(4*L2_CACHE_SIZE / (8 * row['num_latent_factors']))
    #num_users = lemp_dec_rule.query('model == "%s"' % model)['num_users'].min()
    #simdex_estimate = (simdex_sample_time / sample_num_users)*num_users

    correct = (blocked_mm_true_rt > simdex_true_rt) == row['simdex_wins']
    optimizer_rt = simdex_true_rt if row['simdex_wins'] else blocked_mm_true_rt
    overhead_rt = blocked_mm_sample_time if row[
        'simdex_wins'] else simdex_index_time + simdex_clustering_time + simdex_sample_time
    optimizer_rt += overhead_rt

    optimal_rt = min(simdex_true_rt, blocked_mm_true_rt)

    optimizer_vs_simdex.append(simdex_true_rt / optimizer_rt)
    optimizer_vs_optimal.append(optimal_rt / optimizer_rt)
    optimal_vs_simdex.append(simdex_true_rt / optimal_rt)

    true_rt_delta = (simdex_true_rt - blocked_mm_true_rt)
    true_rt_delta_percentage = true_rt_delta / max(simdex_true_rt,
                                                   blocked_mm_true_rt)
    overhead = overhead_rt / optimizer_rt
    results['model'].append(model)
    results['K'].append(K)
    results['blocked_mm_sample_time'].append(blocked_mm_sample_time)
    results['simdex_sample_time'].append(simdex_sample_time)
    results['true_rt_delta'].append(true_rt_delta)
    results['true_rt_delta_percentage'].append(true_rt_delta_percentage)
    results['correct'].append(correct)
    results['overhead'].append(overhead)

results = pd.DataFrame.from_dict(results)
print(results.query('correct == False'))
print('Accuracy', np.mean(results['correct']))
print('Percent Overhead Avg/Max',
      np.mean(results['overhead']), np.max(results['overhead']))
print('Optimizer vs SimDex: ', np.mean(optimizer_vs_simdex))
print('Optimizer vs Optimal: ', np.mean(optimizer_vs_optimal))
print('Optimal vs SimDex: ', np.mean(optimal_vs_simdex))
