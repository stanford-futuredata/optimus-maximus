#!/usr/bin/env bash

# Run this inside of nomad/Code/nomad after compiling nomad
if [ "$#" -ne 4 ]; then
    echo "Four arguments required: [reg] [dim] [input_dir] [output_dir]"
    exit 1
fi

set -x

REG=$1
DIM=$2
INPUT_DIR=$3
OUTPUT_DIR=$4

mkdir -p $OUTPUT_DIR

mpirun ./nomad_double --nthreads 4 --reg $REG --lrate 0.008 --drate 0.01 --dim $DIM --timeout 1000 \
  --path ${INPUT_DIR}/ --output ${OUTPUT_DIR}/ > ${OUTPUT_DIR}/output.txt
# --lrate 0.012 --drate 0.05
