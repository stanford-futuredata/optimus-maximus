#!/bin/sh

#  Yahoo_args.sh
#  ocius
#
#  Created by Geet Sethi on 10/6/16.
#  Copyright © 2016 Geet Sethi. All rights reserved.

export PATH=~/anaconda2/bin:$PATH
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

source activate intelpython2

factors=( 50 25 10 )
clusters=( 64 128 256 512 1024 2048 3072 4096 )
iterations=( 10 20 30 )
samples=( 10 20 40 60 80 100 )

for sample in ${samples[*]};
do
    for iteration in ${iterations[*]};
    do
        for factor in ${factors[*]};
        do
            for cluster in ${clusters[*]};
            do
                exec ./netflix_cluster_worker.sh ${cluster} ${iteration} ${sample} ${factor} &
            done
        done
    done
    echo -e "Waiting for current batch of jobs to finish before starting next batch."
    wait
done


#weightspath=/mnt/ramdisk/datasets/netflix/f_${f}/
#echo -e "Starting Netflix F=" ${f} " Clusters=" ${net_clusters}
#python sample.py ${net_clusters} ${niterations} ${weightspath} 0
#python assign.py ${net_clusters} ${niterations} ${weightspath}
