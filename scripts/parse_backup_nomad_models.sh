#! /usr/bin/env bash

set -x

SIMDEX_DIR=$HOME/simdex

cp $SIMDEX_DIR/scripts/nomad_get_rmse.sh .

for arg; do
  ./nomad_get_rmse.sh ${arg}/output.txt
  tail -n1 -v ${arg}/output.txt.csv >> rmse.csv
  cd $arg && cp $SIMDEX_DIR/scripts/convert_nomad_weights_file.sh . && \
  ./convert_nomad_weights_file.sh "" && rm convert_nomad_weights_file.sh && cd - \
  && gsutil -m cp -r $arg/ gs://simdex/vldb-june-2017/nomad/${arg}/
done

rm nomad_get_rmse.sh

