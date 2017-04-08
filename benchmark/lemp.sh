#!/usr/bin/env bash
set -x

../lemp-no-icc-simd/tools/runLemp --method=LEMP_LI --cacheSizeinKB=2560 \
  --Q^T ~/models-simdex/lemp-paper/Netflix-noav-10/user_weights.csv \
  --P ~/models-simdex/lemp-paper/Netflix-noav-10/item_weights.csv \
  --r=10 --m=480189 --n=17770 --k=1 --t=1
