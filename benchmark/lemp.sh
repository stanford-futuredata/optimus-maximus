#!/usr/bin/env bash
set -x

../lemp-no-icc-simd/tools/runLemp --method=LEMP_LI --cacheSizeinKB=2560 \
  --Q^T ~/models-simdex/pb-new/Netflix-50/user_weights.csv \
  --P ~/models-simdex/pb-new/Netflix-50/item_weights.csv \
  --r=50 --m=480189 --n=17770 --k=1 --t=1
