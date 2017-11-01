#! /usr/bin/env python

import pandas as pd

L2_CACHE_SIZE = 256000
MAX_MEM_SIZE = 257840 * 1024 * 1024


def fexipro_truth_query_fn(row):
    return 'model == "%s" and K == %d and alg == "%s" and scaling_value == %f and sigma == %f' % (
        row['model'], row['K'], row['alg'], row['scaling_value'], row['sigma'])


def lemp_truth_query_fn(row):
    return 'model == "%s" and K == %d' % (row['model'], row['K'])


def print_estimates(estimates_df, truth_df, sample_time_col, truth_query_fn):
    for _, row in estimates_df.iterrows():
        model, K = row['model'], row['K']

        num_users = 480189 if 'Netflix' in model else 1000990 if 'KDD' in model else 1823179
        num_items = 17770 if 'Netflix' in model else 624961 if 'KDD' in model else 136736
        num_latent_factors = row['num_latent_factors']

        user_sample_ratio = row['user_sample_ratio']
        if user_sample_ratio == 0.0:
            num_users_per_block = 4 * L2_CACHE_SIZE / (8 * num_latent_factors)
            while (num_users_per_block * num_items * 8 > MAX_MEM_SIZE):
                num_users_per_block /= 2
            user_sample_ratio = num_users_per_block / num_users

        estimate_rt = row[sample_time_col] * num_users
        true_rt = truth_df.query(truth_query_fn(row))['comp_time'].min()
        print(model, K, user_sample_ratio, estimate_rt, true_rt)


fexipro_estimates = pd.read_csv('runtime-estimates/fexipro.csv')
fexipro_truth = pd.read_csv('timing-results/fexipro-orig-timing.csv')

print('FEXIPRO-SIR')
print_estimates(
    fexipro_estimates.query('alg == "SIR"'),
    fexipro_truth.query('alg == "SIR"'), 'fexipro_sample_time',
    fexipro_truth_query_fn)

print()
print('FEXIPRO-SI')
print_estimates(
    fexipro_estimates.query('alg == "SI"'),
    fexipro_truth.query('alg == "SI"'), 'fexipro_sample_time',
    fexipro_truth_query_fn)

lemp_estimates = pd.read_csv('runtime-estimates/lemp.csv')
lemp_truth = pd.read_csv('timing-results/lemp-gold-standard-timing.csv')

print()
print('LEMP')
print_estimates(lemp_estimates, lemp_truth, 'lemp_sample_time',
                lemp_truth_query_fn)
