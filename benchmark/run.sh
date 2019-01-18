#! /usr/bin/env bash
set -e
set -x

./benchmark_lemp.py --simd --no-decision-rule --user-sample-ratio 0.0 --no-test-only --output-dir lemp-fast-text
./benchmark_blocked_mm.py --output-dir blocked_mm-fast-text --mkl --no-icc 
./benchmark_simdex.py --no-decision-rule --no-test-only --user-sample-ratio 0.0 --num-clusters 1,8,64 --no-icc --output-dir simdex-fast-text
./benchmark_fexipro_orig.py --no-decision-rule --no-test-only --user-sample-ratio 0.0 --output-dir fexipro-fast-text
