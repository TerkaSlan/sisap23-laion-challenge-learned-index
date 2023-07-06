#!/bin/bash

for DATA in pca32v2 #clip768v2 pca32v2 pca96v2 #clip768v2
do
    for LR in 0.1 #0.01 0.001
    do
        for MODEL_TYPE in MLP #Bigger
        do
            for EPOCHS in 10 #50 100
            do
                for N_CATEGORIES in 1000 #2000
                do
                    qsub -q elixircz@elixir-pbs.elixir-czech.cz -l walltime=24:00:00 -l select=1:mem=50gb:ncpus=1 \
                        -N kmeans-$DATA-$N_CATEGORIES-$EPOCHS-$MODEL_TYPE-$LR \
                        -v DATA="$DATA",LR="$LR",PERC_TRAIN="$PERC_TRAIN",MODEL_TYPE="$MODEL_TYPE",EPOCHS="$EPOCHS",N_CATEGORIES="$N_CATEGORIES" \
                        run-single-kmeans-best-setup.sh
                done
            done
        done
    done
done
