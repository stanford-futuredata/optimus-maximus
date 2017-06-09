#!/usr/bin/env bash

# Run from top-level dir, e.g., `scripts/build_lemp.sh --simd --no-icc`
TARGET_DIR="fexipro-orig-build"
C_COMPILER=`which gcc-4.8`
CXX_COMPILER=`which g++-4.8`

set -x

rm -rf $TARGET_DIR && mkdir -p $TARGET_DIR && cd $TARGET_DIR
cmake -D CMAKE_C_COMPILER=$C_COMPILER -D CMAKE_CXX_COMPILER=$CXX_COMPILER ../fexipro-orig/Code/FEXIPRO
make
