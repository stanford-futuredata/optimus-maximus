#! /usr/bin/env bash

set -x
../cpp/simdex \
  -w /lfs/raiders6/ssd/fabuzaid/models-simdex/lemp-paper/Netflix-noav-50 \
  -k 50 -m 480189 -n 17770 -f 50 -c 256 -s 20 -i 1 -b 5 -t 1 \
  --base-name lemp-paper-Netflix-noav-50
