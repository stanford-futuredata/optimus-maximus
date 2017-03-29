#!/usr/env bash

set -x
../cpp/simdex \
  -w /lfs/raiders6/ssd/fabuzaid/models-simdex/pb-new/kdd-10 \
  -k 1 -m 1000990 -n 624961 -f 10 -c 256 -s 20 -i 1 -b 1 -t 8 \
  --base-name csv/pb-new-kdd-10
