#!/usr/bin/env python

import pandas as pd
import argparse
import numpy as np
import os


def error_single_row_fn(user_weights, item_weights, user_id_map, item_id_map):
    def fn(row):
        return np.dot(
            user_weights[user_id_map[row['user_id']]],
            item_weights[item_id_map[row['item_id']]]) - row['rating']

    return fn


def calculate_rmse(user_weights, item_weights, user_id_map, item_id_map, df):
    apply_fn = error_single_row_fn(user_weights, item_weights, user_id_map,
                                   item_id_map)
    df['error'] = df.apply(apply_fn, axis=1)
    import pdb
    pdb.set_trace()
    return np.sqrt(np.sum(np.square(df['error'])) / len(df['error']))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_dir', required=True)
    parser.add_argument('--training_file', required=True)
    parser.add_argument('--test_file', required=True)
    args = parser.parse_args()

    print('Loading training set...')
    training_df = pd.read_csv(
        args.training_file,
        delimiter='\t',
        header=None,
        names=['user_id', 'item_id', 'rating'],
        usecols=[0, 1, 2],
        nrows=1000)
    print('Loading test set...')
    test_df = pd.read_csv(
        args.test_file,
        delimiter='\t',
        header=None,
        names=['user_id', 'item_id', 'rating'],
        usecols=[0, 1, 2],
        nrows=1000)

    print('Creating user id map....')
    all_user_ids = sorted(
        np.union1d(training_df['user_id'].unique(), test_df['user_id']
                   .unique()))
    user_id_map = dict(zip(all_user_ids, range(len(all_user_ids))))
    print('Creating item id map....')
    all_item_ids = sorted(
        np.union1d(training_df['item_id'].unique(), test_df['item_id']
                   .unique()))
    item_id_map = dict(zip(all_item_ids, range(len(all_item_ids))))

    print('Loading item weights...')
    item_weights = np.loadtxt(
        '%s/item_weights.csv' % args.weights_dir, delimiter=',')
    print('Loading user weights...')
    user_weights = np.loadtxt(
        '%s/user_weights.csv' % args.weights_dir, delimiter=',')

    training_rmse = calculate_rmse(user_weights, item_weights, user_id_map,
                                   item_id_map, training_df)
    print('Training RMSE: %f' % training_rmse)
    test_rmse = calculate_rmse(user_weights, item_weights, user_id_map,
                               item_id_map, test_df)
    print('Test RMSE: %f' % test_rmse)
    


if __name__ == '__main__':
    main()
