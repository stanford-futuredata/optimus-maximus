#!/usr/bin/env bash

set -x

../fexipro-orig-build/runFEXIPRO \
  --k 1 --alg SIR --dataset lemp-paper-Netflix-noav-10 \
  --q ~/models-simdex/lemp-paper/Netflix-noav-10/user_weights.csv \
  --p ~/models-simdex/lemp-paper/Netflix-noav-10/item_weights.csv \
  --outputResult false --logPathPrefix fexipro-log/

