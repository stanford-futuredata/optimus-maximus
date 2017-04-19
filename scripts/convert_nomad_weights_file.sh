#!/usr/bin/env bash

if [ "$#" -ne 1 ]; then
    echo "One argument required: [model_prefix]"
    exit 1
fi

MODEL_PREFIX=$1
set -x

grep -h "^row" ${MODEL_PREFIX}[0-9] > users.txt
grep -h "^column" ${MODEL_PREFIX}[0-9] > items.txt
sort -t , -k 2 -g users.txt > sorted_users.txt
sort -t , -k 2 -g items.txt > sorted_items.txt
sed -i.bak -e "s/^row,[0-9]\+,//" sorted_users.txt && mv sorted_users.txt user_weights.csv
sed -i.bak -e "s/^column,[0-9]\+,//" sorted_items.txt && mv sorted_items.txt item_weights.csv

