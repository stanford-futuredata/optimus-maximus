#!/usr/bin/env python

import pandas as pd
import numpy as np

simdex_dec_rule = pd.read_csv('simdex-decision-rule.csv')
lemp_dec_rule = pd.read_csv('lemp-decision-rule.csv')

lemp_truth = pd.read_csv(
    '../../plots/timing-results/lemp-gold-standard-timing.csv')

simdex_truth_netflix = pd.read_csv(
    '../../plots/timing-results/netflix-simdex-timing.csv')
simdex_truth_kdd = pd.read_csv(
    '../../plots/timing-results/kdd-simdex-timing.csv')
simdex_truth_r2 = pd.read_csv(
    '../../plots/timing-results/r2-simdex-timing.csv')
simdex_truth = pd.concat(
    [simdex_truth_netflix, simdex_truth_kdd, simdex_truth_r2])

blocked_mm_truth_netflix = pd.read_csv(
    '../../plots/timing-results/netflix-blocked_mm-timing.csv')
blocked_mm_truth_kdd = pd.read_csv(
    '../../plots/timing-results/kdd-blocked_mm-timing.csv')
blocked_mm_truth_r2 = pd.read_csv(
    '../../plots/timing-results/r2-blocked_mm-timing.csv')
blocked_mm_truth = pd.concat(
    [blocked_mm_truth_netflix, blocked_mm_truth_kdd, blocked_mm_truth_r2])

results = {
    'model': [],
    'K': [],
    'blocked_mm_sample_time': [],
    'simdex_sample_time': [],
    'correct': [],
    'overhead': [],
}

BLOCKED_MM = 0
LEMP = 1
SIMDEX = 2

optimizer_vs_optimal = []
optimizer_vs_lemp = []
optimal_vs_lemp = []
simdex_vs_lemp = []

for _, row in simdex_dec_rule.iterrows():
    model, K = row['model'], row['K']
    lemp_row = lemp_dec_rule.query('model == "%s" and K == %d' % (model, K))
    assert (len(lemp_row) == 1)
    lemp_row = lemp_row.iloc[0]
    assert (lemp_row['num_users'] == row['num_sampled_users'])
    num_sampled_users = row['num_sampled_users']

    blocked_mm_sample_time, simdex_sample_time = row[
        'blocked_mm_sample_time'], row['simdex_sample_time']
    simdex_index_time, simdex_clustering_time = row['index_time'], row[
        'cluster_time']
    lemp_sample_time = lemp_row['lemp_sample_time']
    lemp_index_time, lemp_preproc_time = lemp_row['preproc_time'], lemp_row[
        'index_time']

    simdex_true_rt = simdex_truth.query('model == "%s" and K == %d' %
                                        (model, K))['comp_time'].min()

    blocked_mm_true_rt = blocked_mm_truth.query('model == "%s" and K == %d' %
                                                (model, K))['comp_time'].min()
    lemp_true_row = lemp_truth.query('model == "%s" and K == %d' % (model, K))
    lemp_true_rt = lemp_true_row['comp_time'].min()
    num_users = lemp_true_row['num_users'].min()
    lemp_estimate_rt = lemp_sample_time * num_users

    winner = BLOCKED_MM if min(blocked_mm_sample_time, simdex_sample_time, lemp_sample_time) == blocked_mm_sample_time else \
             LEMP if min(blocked_mm_sample_time, simdex_sample_time, lemp_sample_time) == lemp_sample_time else \
             SIMDEX
    true_winner = BLOCKED_MM if min(blocked_mm_true_rt, simdex_true_rt, lemp_true_rt) == blocked_mm_true_rt else \
                  LEMP if min(blocked_mm_true_rt, simdex_true_rt, lemp_true_rt) == lemp_true_rt else \
                  SIMDEX
    correct = (winner == true_winner)

    optimizer_rt = simdex_true_rt if winner == SIMDEX else blocked_mm_true_rt if winner == BLOCKED_MM else lemp_true_rt
    overhead_rt = simdex_sample_time*num_sampled_users + simdex_index_time + simdex_clustering_time + \
                      lemp_sample_time*num_sampled_users + lemp_preproc_time + lemp_index_time if winner == BLOCKED_MM \
                  else \
                       blocked_mm_sample_time*num_sampled_users + \
                          lemp_sample_time*num_sampled_users + lemp_preproc_time + lemp_index_time if winner == SIMDEX \
                  else \
                       simdex_sample_time*num_sampled_users + simdex_index_time + simdex_clustering_time + \
                          blocked_mm_sample_time*num_sampled_users
    optimizer_rt += overhead_rt

    optimal_rt = min(simdex_true_rt, lemp_true_rt, blocked_mm_true_rt)

    optimizer_vs_lemp.append(lemp_true_rt / optimizer_rt)
    optimizer_vs_optimal.append(optimal_rt / optimizer_rt)
    optimal_vs_lemp.append(lemp_true_rt / optimal_rt)
    simdex_vs_lemp.append(lemp_true_rt / simdex_true_rt)

    overhead = overhead_rt / optimizer_rt
    results['model'].append(model)
    results['K'].append(K)
    results['blocked_mm_sample_time'].append(blocked_mm_sample_time)
    results['simdex_sample_time'].append(simdex_sample_time)
    results['correct'].append(correct)
    results['overhead'].append(overhead)

results = pd.DataFrame.from_dict(results)
print(results.query('correct == False'))
print('Accuracy', np.mean(results['correct']))
print('Percent Overhead Avg/Max',
      np.mean(results['overhead']),
      np.std(results['overhead']), np.max(results['overhead']))
print('Optimizer vs LEMP: ', np.mean(optimizer_vs_lemp))
print('Optimizer vs Optimal: ', np.mean(optimizer_vs_optimal))
print(optimal_vs_lemp)
print('Optimal vs LEMP: ', np.mean(optimal_vs_lemp))
print('SB vs LEMP: ',
      np.mean(simdex_vs_lemp), np.min(simdex_vs_lemp), np.max(simdex_vs_lemp))
