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

$RUNNER \
  -w $HOME/models-simdex/lemp-paper/Netflix-noav-10 \
  -k 1 -m 480189 -n 17770 -f 10 -c 8 -s 10 -i 3 -b 3 -t 1 \
  --batch-size 4096 \
  --base-name lemp-paper-Netflix-noav-10
