#!/usr/bin/env bash
set -x

cd ../fexipro && make clean && make && cd -

../fexipro/fexipro \
  -w $HOME/models-simdex/lemp-paper/Netflix-noav-10 \
  -k 1 -m 480189 -n 17770 -f 10 \
  --base-name lemp-paper-Netflix-noav-10
