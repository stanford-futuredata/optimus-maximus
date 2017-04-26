#!/usr/bin/env bash

set -x

for arg; do
  ./nomad_get_rmse.sh ${arg}/output.txt
  tail -n1 -v ${arg}/output.txt.csv >> rmse.csv
  cd $arg && cp $HOME/simdex/scripts/convert_nomad_weights_file.sh . && \
  ./convert_nomad_weights_file.sh "" && rm convert_nomad_weights_file.sh && cd - \
  && aws s3 cp $arg s3://ocius-experiments/models/nomad/${arg}/ --recursive
done
aws s3 cp rmse.csv s3://ocius-experiments/models/nomad/rmse.csv

