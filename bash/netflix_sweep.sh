#!/bin/sh

#  Yahoo_args.sh
#  ocius
#
#  Created by Geet Sethi on 10/6/16.
#  Copyright © 2016 Geet Sethi. All rights reserved.


topk=( 50 10 5 1 )
factors=( 50 25 10 )
bins=1
clusters=( 64 128 256 512 1024 2048 3072 4096 )
iterations=( 20 30 10 )
samples=( 20 40 60 80 100 10 )

for iteration in ${iterations[*]};
do
    for sample in ${samples[*]};
    do
        for k in ${topk[*]};
        do
            for f in ${factors[*]};
            do
                for cluster in ${clusters[*]};
                do
                    weightspath=/mnt/ramdisk/datasets/netflix/f_${f}/
                    clusterpath=/mnt/ramdisk/datasets/netflix/f_${f}/${sample}/${iteration}/
                    echo -e "Starting Netflix F=" ${f} " K=" ${k} " Bins=" ${bins}
                    ../simdex 17770 ${f} ${cluster} ${bins} ${k} ${cluster} ${weightspath} ${clusterpath} 1 ${sample} ${iteration} &
                done
            done
        done
    echo -e "Waiting for current batch of jobs with sample: " ${sample} " and iters: " ${iteration} " to finish before starting next batch."
    wait
    done
done

