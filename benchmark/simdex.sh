#! /usr/bin/env bash

set -x
../cpp/simdex \
  -w $HOME/models-simdex/lemp-paper/Netflix-noav-10 \
  -k 1 -m 480189 -n 17770 -f 10 -c 256 -s 20 -i 1 -b 5 -t 1 \
  --base-name lemp-paper-Netflix-noav-10
