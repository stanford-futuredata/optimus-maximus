#! /usr/bin/env bash

if [[ $1 == "--stats" ]]; then
  RUNNER=../cpp/simdex_stats
  FLAGS="ICC=1 STATS=1"
elif [[ $1 == "--debug" ]]; then
  RUNNER=../cpp/simdex_debug
  FLAGS="ICC=1 DEBUG=1"
elif [[ $1 == "--mkl" ]]; then
  RUNNER=../cpp/simdex
  FLAGS="MKL=1"
elif [[ $1 == "--icc" ]]; then
  RUNNER=../cpp/simdex
  FLAGS="ICC=1"
else
  RUNNER=../cpp/simdex
fi

if [ -z "$2" ]; then
  BATCH_SIZE=1024
else
  BATCH_SIZE=$2
fi

set -x

cd ../cpp && make clean && make -j5 $FLAGS && cd -

$RUNNER \
  -w $HOME/models-simdex/lemp-paper/Netflix-noav-50 \
  -k 50 -m 480189 -n 17770 -f 50 -c 256 -s 10 -i 3 -b 3 -t 1 \
  --batch-size $BATCH_SIZE \
  --base-name lemp-paper-Netflix-noav-50
