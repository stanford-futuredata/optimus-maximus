#! /usr/bin/env bash

set -x

if [[ $1 == "--stats" ]]; then
  RUNNER=../cpp/simdex_stats
else
  RUNNER=../cpp/simdex
fi

$RUNNER \
  -w $HOME/models-simdex/lemp-paper/Netflix-noav-50 \
  -k 1 -m 480189 -n 17770 -f 50 -c 256 -s 10 -i 3 -b 5 -t 1 \
  --base-name lemp-paper-Netflix-noav-50
