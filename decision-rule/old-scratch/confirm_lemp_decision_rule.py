#!/usr/bin/env python

from __future__ import print_function
import pandas as pd
import numpy as np
import sys

lemp_dec_rule = pd.read_csv(sys.argv[1])

lemp_truth = pd.read_csv(
    '../plots/timing-results/lemp-gold-standard-timing.csv')

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
    'lemp_sample_time': [],
    'true_rt_delta': [],
    'true_rt_delta_percentage': [],
    'lemp_true_rt': [],
    'lemp_estimate_rt': [],
    'blocked_mm_true_rt': [],
    'blocked_mm_estimate_rt': [],
    'correct': [],
    'overhead': [],
}

optimizer_vs_optimal = []
optimizer_vs_lemp = []
optimal_vs_lemp = []

for _, row in lemp_dec_rule.iterrows():
    model, K = row['model'], row['K']
    blocked_mm_sample_time, lemp_sample_time = row[
        'blocked_mm_sample_time'], row['lemp_sample_time']
    lemp_index_time, lemp_preproc_time = row['preproc_time'], row['index_time']

    lemp_true_row = lemp_truth.query('model == "%s" and K == %d' % (model, K))
    lemp_true_rt = lemp_true_row['comp_time'].min()
    num_users = lemp_true_row['num_users'].min()
    lemp_estimate_rt = lemp_sample_time * num_users

    blocked_mm_true_rt = blocked_mm_truth.query('model == "%s" and K == %d' %
                                                (model, K))['comp_time'].min()
    blocked_mm_estimate_rt = blocked_mm_sample_time * num_users

    correct = (blocked_mm_true_rt > lemp_true_rt) == row['lemp_wins']
    optimizer_rt = lemp_true_rt if row['lemp_wins'] else blocked_mm_true_rt
    overhead_rt = blocked_mm_sample_time if row[
        'lemp_wins'] else lemp_index_time + lemp_preproc_time + lemp_sample_time
    optimizer_rt += overhead_rt

    optimal_rt = min(lemp_true_rt, blocked_mm_true_rt)

    optimizer_vs_lemp.append(lemp_true_rt / optimizer_rt)
    optimizer_vs_optimal.append(optimal_rt / optimizer_rt)
    optimal_vs_lemp.append(lemp_true_rt / optimal_rt)

    true_rt_delta = (lemp_true_rt - blocked_mm_true_rt)
    true_rt_delta_percentage = true_rt_delta / max(lemp_true_rt,
                                                   blocked_mm_true_rt)
    overhead = overhead_rt / optimizer_rt
    results['model'].append(model)
    results['K'].append(K)
    results['blocked_mm_sample_time'].append(blocked_mm_sample_time)
    results['lemp_sample_time'].append(lemp_sample_time)
    results['true_rt_delta'].append(true_rt_delta)
    results['true_rt_delta_percentage'].append(true_rt_delta_percentage)
    results['correct'].append(correct)
    results['overhead'].append(overhead)
    results['lemp_true_rt'].append(lemp_true_rt)
    results['lemp_estimate_rt'].append(lemp_estimate_rt)
    results['blocked_mm_true_rt'].append(blocked_mm_true_rt)
    results['blocked_mm_estimate_rt'].append(blocked_mm_estimate_rt)

results = pd.DataFrame.from_dict(results)
print(results.query('correct == False'))
#print(results[['model', 'K', 'lemp_estimate_rt', 'lemp_true_rt']])
print('Accuracy', np.mean(results['correct']))
print('Percent Overhead Avg/Max',
      np.mean(results['overhead']), np.max(results['overhead']))
print('Optimizer vs Lemp: ', np.mean(optimizer_vs_lemp))
print('Optimizer vs Optimal: ', np.mean(optimizer_vs_optimal))
print('Optimal vs Lemp: ', np.mean(optimal_vs_lemp))
