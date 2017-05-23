#!/usr/bin/env bash

if [[ $1 == "--debug" ]]; then
  RUNNER=../fexipro/fexipro_debug
  FLAGS="DEBUG=1"
else
  RUNNER=../fexipro/fexipro
fi

set -x

cd ../fexipro && make clean && make $FLAGS && cd -

$RUNNER \
  -w $HOME/models-simdex/lemp-paper/Netflix-noav-50 \
  -k 1 -m 480189 -n 17770 -f 50 \
  --base-name lemp-paper-Netflix-noav-50
