#!/usr/bin/env bash

# Run from top-level dir, e.g., `scripts/build_lemp.sh --simd --decision-rule`

TARGET_DIR="lemp"
C_COMPILER=`which gcc-4.8`
CXX_COMPILER=`which g++-4.8`
DEFINITION_FLAGS=""

for arg in "$@"
do
  case "$arg" in
    '--icc')
      TARGET_DIR+="-icc"
      C_COMPILER=`which icc`
      CXX_COMPILER=`which icc`
      ;;
    '--simd')
      TARGET_DIR+="-simd"
      DEFINITION_FLAGS="-D SIMD=1"
      ;;
    '--decision-rule')
      TARGET_DIR+="-decision-rule"
      DEFINITION_FLAGS+=" -D RULE=1"
      ;;
    '--test-only')
      TARGET_DIR+="-test-only"
      DEFINITION_FLAGS+=" -D TEST_ONLY=1"
      ;;
  esac
done

set -x

rm -rf $TARGET_DIR && mkdir -p $TARGET_DIR && cd $TARGET_DIR
cmake -D CMAKE_C_COMPILER=$C_COMPILER -D CMAKE_CXX_COMPILER=$CXX_COMPILER $DEFINITION_FLAGS ../LEMP-benchmarking/
make -j
