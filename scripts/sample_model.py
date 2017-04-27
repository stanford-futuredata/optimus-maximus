#!/usr/bin/env python

from __future__ import print_function
from __future__ import division
import numpy as np
import argparse
import subprocess

SAMPLE_RATIOS = np.arange(0.1, 1.0, 0.1)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_users', type=int, required=True)
    parser.add_argument('--num_items', type=int, required=True)
    parser.add_argument('--model_dir', required=True)
    args = parser.parse_args()

    #sed '1p;2p;3p;15p;624962q;d' item_weights.csv > foo.csv
    num_users = args.num_users
    num_items = args.num_items
    for ratio in SAMPLE_RATIOS:
        print('Running ' + str(ratio))
        with open('%s/user_weights.csv' % args.model_dir) as infile, open(
                '%s/user_weights_%0.1f.csv' % (args.model_dir, ratio),
                'w') as outfile:
            sampled_user_ids = set(
                np.random.choice(
                    num_users, int(num_users * ratio), replace=False))
            for line_num, line in enumerate(infile):
                if line_num in sampled_user_ids:
                    print(line, end='', file=outfile)
            #for user_id_chunk in chunks(sampled_user_ids, 100):
            #    sed_user_str = 'p;'.join(str(val) for val in user_id_chunk) + 'q;d'
            #    cmd = [
            #            'sed', sed_user_str, '%s/user_weights.csv' % args.model_dir,
            #            ]
            #    print('Running' + str(cmd))
            #    subprocess.call(cmd, stdout=outfile)

        with open('%s/item_weights.csv' % args.model_dir) as infile, open(
                '%s/item_weights_%0.1f.csv' % (args.model_dir, ratio),
                'w') as outfile:
            sampled_item_ids = set(
                np.random.choice(
                    num_items, int(num_items * ratio), replace=False))
            for line_num, line in enumerate(infile):
                if line_num in sampled_item_ids:
                    print(line, end='', file=outfile)


if __name__ == '__main__':
    main()
