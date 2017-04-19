#!/usr/bin/env bash

# Run this inside of nomad/Code/nomad after compiling nomad

set -x

mkdir -p $HOME/models/nomad/KDD-100
mkdir -p $HOME/data/yahoo_KDD_cup_2011_dataset/nomad-format

mpirun ./nomad_double --nthreads 4 --reg 1.00 --lrate 0.0005 --drate 0.05 --dim 100 --timeout 500 \
  --output $HOME/models/nomad/KDD-100/ --path $HOME/data/yahoo_KDD_cup_2011_dataset/nomad-format
