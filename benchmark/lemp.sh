#!/usr/bin/env bash
set -x

for thread in 1 2 4 8 16 32 64; do
  ../lemp-simd/tools/runLemp --method=LEMP_LI --cacheSizeinKB=2560 \
    --Q^T ~/models-simdex/lemp-paper/Netflix-noav-100/user_weights.csv \
    --P ~/models-simdex/lemp-paper/Netflix-noav-100/item_weights.csv \
    --r=100 --m=480189 --n=17770 --k=1 --t=${thread} --resultsFile results.txt
done
