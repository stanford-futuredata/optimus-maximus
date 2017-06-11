#! /usr/bin/env python

import argparse
import glob
import re
import pandas as pd

OUTPUT_CSV_FNAME = 'fexipro-orig-timing.csv'
HEADERS = [
    'model', 'K', 'num_latent_factors', 'alg', 'preproc_time', 'comp_time'
]


def parse(input_dir):
    rows = []
    for fname in glob.iglob('%s/*txt' % input_dir):
        if 'result' in fname:
            continue
        with open(fname) as f:
            fname = fname[len(input_dir):]
            model = re.search('(.+)-SI', fname).group(1)
            values_dict = {'model': model}
            for line in f:
                line = line.rstrip()
                parts = line.split(': ')
                header, value = parts[0], parts[1]
                if header == 'Algorithm':
                    values_dict['alg'] = value
                elif header == 'p':
                    values_dict['num_latent_factors'] = value.split(',')[-1]
                elif header == 'k':
                    values_dict['K'] = value
                elif header == 'preprocess time':
                    values_dict['preproc_time'] = value.split(' ')[0]
                elif header == 'online time':
                    values_dict['comp_time'] = value.split(' ')[0]
        if len(values_dict) != len(HEADERS):
            continue
        row = [values_dict[key] for key in HEADERS]
        rows.append(row)
    df = pd.DataFrame(rows, columns=HEADERS)
    df.to_csv(OUTPUT_CSV_FNAME, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    args = parser.parse_args()
    parse(args.input_dir)


if __name__ == '__main__':
    main()
