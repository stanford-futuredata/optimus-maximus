#!/usr/bin/env bash

FLAGS="" # No MKL or ICC by default

for arg in "$@"
do
  case "$arg" in
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
  esac
done

set -x

cd ../cpp/blocked_mm && make clean && make $FLAGS -j2 && cd -

for thread in 1 2 4 8 16 32 64; do
  export OPENBLAS_NUM_THREADS=$thread
  ../cpp/blocked_mm/blocked_mm \
    -q $HOME/models-simdex/lemp-paper/Netflix-noav-100/user_weights.csv \
    -p $HOME/models-simdex/lemp-paper/Netflix-noav-100/item_weights.csv \
    -k 1 -m 480189 -n 17770 -f 100 -t $thread \
    --base-name netflix-blocked_mm-parallelized-num-threads-${thread}
done
