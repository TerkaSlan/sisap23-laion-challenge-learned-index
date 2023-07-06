#!/bin/bash
#0.2,4,9,0.01,33,11331
#0.2,4,11,0.01,33,11331
#0.2,4,10,0.005,33,11331
#0.2,4,10,0.01,33,11331
for EPS in 0.11 0.12
do
    for MIN_SAMPLES in 12 13
    do
        for LEAF_SIZE in 7 8
        do
            for P in 0.6 0.7
            do
                qsub -q elixircz@elixir-pbs.elixir-czech.cz -l walltime=4:00:00 -l select=1:mem=50gb:ncpus=1 \
                    -N dbscan-$EPS-$MIN_SAMPLES-$LEAF_SIZE-$P \
                    -v EPS="$EPS",MIN_SAMPLES="$MIN_SAMPLES",LEAF_SIZE="$LEAF_SIZE",P="$P" \
                    run-single-dbscan-best-setup.sh
            done
        done
    done
done
