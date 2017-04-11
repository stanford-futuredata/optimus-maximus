#!/usr/bin/env bash
set -x

../cpp/blocked_mm/blocked_mm \
  -w /lfs/raiders6/ssd/fabuzaid/models-simdex/lemp-paper/Netflix-noav-10 \
  -k 1 -m 480189 -n 17770 -f 10 -t 1 \
  --base-name lemp-paper-Netflix-noav-10
