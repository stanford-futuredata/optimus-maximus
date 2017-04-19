#!/usr/bin/env python

from __future__ import print_function

import numpy as np
from scipy.io import mmread
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--user_weights_file', required=True)
    parser.add_argument('--item_weights_file', required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()

    print('Saving item weights....')
    item_mat = mmread(args.item_weights_file)
    item_mat = item_mat.T  # transpose item weights
    item_mat[item_mat < 1e-9] = 0.0
    np.savetxt(
        '%s/item_weights.csv' % args.output_dir, item_mat.T, delimiter=',')

    print('Saving user weights....')
    user_mat = mmread(args.user_weights_file)
    user_mat[user_mat < 1e-9] = 0.0
    np.savetxt(
        '%s/user_weights.csv' % args.output_dir, user_mat, delimiter=',')


if __name__ == '__main__':
    main()
