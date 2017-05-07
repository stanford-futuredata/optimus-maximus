#!/usr/bin/env python

from __future__ import print_function
import pandas as pd
import numpy as np
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stats_file', required=True)
    parser.add_argument(
        '--simdex_faster', dest='simdex_faster', action='store_true')
    parser.add_argument(
        '--blocked_mm_faster', dest='simdex_faster', action='store_false')
    parser.set_defaults(simdex_faster=False)
    parser.add_argument('--num_items', type=int, required=True)
    args = parser.parse_args()

    stats_file = args.stats_file
    NUM_ITEMS = args.num_items

    df = pd.read_csv(stats_file)
    for i in range(100):
        if (np.percentile(df.sample(frac=0.0001)['num_items_visited'], 75) <
                NUM_ITEMS * 0.15) != args.simdex_faster:
            print('Trial %d failed' % i)
            return
    print('It passed!')


if __name__ == '__main__':
    main()
