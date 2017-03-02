#!/bin/sh

#  Yahoo_args.sh
#  ocius
#
#  Created by Geet Sethi on 10/6/16.
#  Copyright © 2016 Geet Sethi. All rights reserved.

numClusters=$1
numIterations=$2
sampleSize=$3
f=$4
weightspath=/mnt/ramdisk/datasets/netflix/f_${f}/
echo -e "Starting Netflix F=" ${f} " Clusters=" ${numClusters}
python ../python_scripts/kmeans_sample.py ${numClusters} ${numIterations} ${weightspath} ${sampleSize}
python ../python_scripts/kmeans_assign.py ${numClusters} ${numIterations} ${weightspath} ${sampleSize}