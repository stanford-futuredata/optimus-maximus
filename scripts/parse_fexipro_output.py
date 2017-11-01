#! /usr/bin/env python

import argparse
import glob
import re
import pandas as pd

OUTPUT_CSV_FNAME = 'fexipro-orig-timing.csv'
HEADERS = [
    'model', 'K', 'num_latent_factors', 'alg', 'scaling_value', 'sigma',
    'preproc_time', 'comp_time', 'num_items_first_check',
    'num_items_last_check', 'user_sample_ratio', 'num_sampled_users', 'blocked_mm_sample_time',
    'fexipro_sample_time', 'fexipro_wins'
]


def parse(input_dir):
    if input_dir[-1] != '/': input_dir += '/'
    rows = []
    for fname in glob.iglob('%s/*txt' % input_dir):
        if 'result' in fname:
            continue
        with open(fname) as f:
            fname = fname[len(input_dir):]
            model = re.search('(.+)-SI', fname).group(1)
            values_dict = {
                'model': model,
                'preproc_time': '0.0',
                'comp_time': '0.0',
                'num_items_first_check': '0.0',
                'num_items_last_check': '0.0',
                'blocked_mm_sample_time': '0.0',
                'fexipro_sample_time': '0.0',
                'fexipro_wins': True,
                'user_sample_ratio': '0.0',
                'num_sampled_users': '0',
            }
            for line in f:
                line = line.rstrip()

                parts = line.split(': ')
                header = parts[0]

                if line == 'FEXIPRO wins':
                    values_dict['fexipro_wins'] = True
                    continue
                elif line == 'Blocked MM wins':
                    values_dict['fexipro_wins'] = False
                    continue
                # handle case where last line is "FEXIPRO wins" or "Blocked MM wins"

                value = parts[1]
                if header == 'Algorithm':
                    values_dict['alg'] = value
                elif header == 'p':
                    values_dict['num_latent_factors'] = value.split(',')[-1]
                elif header == 'k':
                    values_dict['K'] = value
                elif header == 'SIGMA':
                    values_dict['sigma'] = value
                elif header == 'Scaling Value':
                    values_dict['scaling_value'] = value
                elif header == 'preprocess time':
                    values_dict['preproc_time'] = value.split(' ')[0]
                elif header == 'online time':
                    values_dict['comp_time'] = value.split(' ')[0]
                elif header == 'Avg Num of p which can pass first Cauchy Schwarz inequality check':
                    values_dict['num_items_first_check'] = value
                elif header == 'Avg Num of p need to be calculated exactly (line 16)':
                    values_dict['num_items_last_check'] = value
                elif header == 'Blocked MM time':
                    values_dict['blocked_mm_sample_time'] = value
                elif header == 'FEXIPRO time':
                    values_dict['fexipro_sample_time'] = value
                elif header == 'User sample ratio':
                    values_dict['user_sample_ratio'] = value
                elif header == 'Num sampled users':
                    values_dict['num_sampled_users'] = value

        if len(values_dict) != len(HEADERS):
            continue
        row = [values_dict[key] for key in HEADERS]
        rows.append(row)
    df = pd.DataFrame(rows, columns=HEADERS)
    df.to_csv(OUTPUT_CSV_FNAME, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', required=True)
    args = parser.parse_args()
    parse(args.input_dir)


if __name__ == '__main__':
    main()
