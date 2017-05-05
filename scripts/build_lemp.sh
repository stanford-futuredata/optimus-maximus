#!/usr/bin/env bash

# Run from top-level dir, e.g., `scripts/build_lemp.sh --simd --no-icc`

if [ "$#" -ne 2 ]; then
    echo "Two arguments required: [--simd|--no-simd] [--icc|--no-icc]"
    exit 1
fi

TARGET_DIR="lemp"
C_COMPILER=`which gcc-4.8`
CXX_COMPILER=`which g++-4.8`
DEFINITION_FLAGS=""

if [ "$2" = "--icc" ]; then
    TARGET_DIR+="-icc"
    C_COMPILER=`which icc`
    CXX_COMPILER=`which icc`
elif [ "$2" = "--no-icc" ]; then
    TARGET_DIR+="-no-icc"
    C_COMPILER=`which gcc-4.8`
    CXX_COMPILER=`which g++-4.8`
fi
if [ "$1" = "--simd" ]; then 
    TARGET_DIR+="-simd"
    DEFINITION_FLAGS="-D SIMD=1"
elif [ "$1" = "--no-simd" ]; then 
    TARGET_DIR+="-no-simd"
fi

set -x

rm -rf $TARGET_DIR && mkdir -p $TARGET_DIR && cd $TARGET_DIR
cmake -D CMAKE_C_COMPILER=$C_COMPILER -D CMAKE_CXX_COMPILER=$CXX_COMPILER $DEFINITION_FLAGS ../LEMP-benchmarking/
make -j4
