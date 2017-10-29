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

../cpp/blocked_mm/blocked_mm \
  -q $HOME/models-simdex/lemp-paper/Netflix-noav-10/user_weights.csv \
  -p $HOME/models-simdex/lemp-paper/Netflix-noav-10/item_weights.csv \
  -k 1 -m 480189 -n 17770 -f 10 -t 1 \
  --base-name lemp-paper-Netflix-noav-10
