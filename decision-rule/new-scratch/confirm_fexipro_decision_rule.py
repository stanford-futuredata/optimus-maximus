#!/usr/bin/env python

import pandas as pd
import numpy as np
import sys

fexipro_dec_rule = pd.read_csv('fexipro-decision-rule.csv').query(
    'alg == "%s"' % sys.argv[1])

fexipro_truth = pd.read_csv('../plots/timing-results/fexipro-orig-timing.csv')

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
    'fexipro_sample_time': [],
    'true_rt_delta': [],
    'true_rt_delta_percentage': [],
    'correct': [],
    'overhead': [],
}

optimizer_vs_optimal = []
optimizer_vs_fexipro = []
optimal_vs_fexipro = []

for _, row in fexipro_dec_rule.iterrows():
    model, K, alg = row['model'], row['K'], row['alg']
    blocked_mm_sample_time, fexipro_sample_time = row[
        'blocked_mm_sample_time'], row['fexipro_sample_time']
    fexipro_preproc_time = row['preproc_time']

    fexipro_true_rt = fexipro_truth.query(
        'model == "%s" and K == %d and alg == "%s"' %
        (model, K, alg))['comp_time'].min()
    blocked_mm_true_rt = blocked_mm_truth.query('model == "%s" and K == %d' %
                                                (model, K))['comp_time'].min()

    correct = (blocked_mm_true_rt > fexipro_true_rt) == row['fexipro_wins']
    optimizer_rt = fexipro_true_rt if row[
        'fexipro_wins'] else blocked_mm_true_rt
    overhead_rt = blocked_mm_sample_time if row[
        'fexipro_wins'] else fexipro_preproc_time + fexipro_sample_time
    optimizer_rt += overhead_rt

    optimal_rt = min(fexipro_true_rt, blocked_mm_true_rt)

    optimizer_vs_fexipro.append(fexipro_true_rt / optimizer_rt)
    optimizer_vs_optimal.append(optimal_rt / optimizer_rt)
    optimal_vs_fexipro.append(fexipro_true_rt / optimal_rt)

    true_rt_delta = (fexipro_true_rt - blocked_mm_true_rt)
    true_rt_delta_percentage = true_rt_delta / max(fexipro_true_rt,
                                                   blocked_mm_true_rt)
    overhead = overhead_rt / optimizer_rt
    results['model'].append(model)
    results['K'].append(K)
    results['blocked_mm_sample_time'].append(blocked_mm_sample_time)
    results['fexipro_sample_time'].append(fexipro_sample_time)
    results['true_rt_delta'].append(true_rt_delta)
    results['true_rt_delta_percentage'].append(true_rt_delta_percentage)
    results['correct'].append(correct)
    results['overhead'].append(overhead)

results = pd.DataFrame.from_dict(results)
print(results.query('correct == False'))
print('Accuracy', np.mean(results['correct']))
print('Percent Overhead Avg/Max',
      np.mean(results['overhead']), np.max(results['overhead']))
print('Optimizer vs FEXIPRO-%s: ' % sys.argv[1], np.mean(optimizer_vs_fexipro))
print('Optimizer vs Optimal: ', np.mean(optimizer_vs_optimal))
print('Optimal vs FEXIPRO-%s: ' % sys.argv[1], np.mean(optimal_vs_fexipro))
