#! /usr/bin/env bash

RUNNER=../cpp/simdex # release binary by default
FLAGS="" # No MKL or ICC by default

for arg in "$@"
do
  case "$arg" in
    '--stats')
      RUNNER=../cpp/simdex_stats
      FLAGS+=" STATS=1"
      ;;
    '--debug')
      RUNNER=../cpp/simdex_debug
      FLAGS+=" DEBUG=1"
      ;;
    '--mkl')
      FLAGS+=" MKL=1"
      ;;
    '--icc')
      FLAGS+=" ICC=1"
      ;;
    '--naive')
      FLAGS+=" NAIVE=1"
      ;;
    '--decision-rule')
      FLAGS+=" RULE=1"
      ;;
    '--test-only')
      FLAGS+=" TEST_ONLY=1"
      ;;
  esac
done

set -x

cd ../cpp && make clean && make -j5 $FLAGS && cd -

export OPENBLAS_NUM_THREADS=1

for thread in 1 2 4 8 16; do
  $RUNNER \
    -w $HOME/models-simdex/lemp-paper/Netflix-noav-100 \
    -k 1 -m 480189 -n 17770 -f 100 -c 64 -s 10 -i 3 -b 3 -t $thread \
    --batch-size 4096 \
    --base-name simdex-parallelized-lemp-paper-Netflix-noav-100
done
