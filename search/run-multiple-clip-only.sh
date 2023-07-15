#!/bin/bash

SIZE=10M

for DATASET in clip768v2
do
    for LR in 0.5 0.1 #0.05 #0.01 0.001
    do
        for N_LEVELS in 2 3
        do
            for MODEL_TYPE in MLP Bigger
            do
                for EPOCHS in 50 #100 200
                do
                    for N_CATEGORIES in 50 #100 200
                    do
                        qsub -q elixircz@elixir-pbs.elixir-czech.cz -l walltime=24:00:00 -l select=1:mem=50gb:ncpus=1 \
                        -N sisap-$DATSET-$N_CATEGORIES-$EPOCHS-$MODEL_TYPE-$N_LEVELS-$LR \
                        -v LR="$LR",N_LEVELS="$N_LEVELS",MODEL_TYPE="$MODEL_TYPE",EPOCHS="$EPOCHS",N_CATEGORIES="$N_CATEGORIES",DATASET="$DATASET",SIZE="$SIZE" \
                        run-single-sisap-exp.sh
                    done
                done
            done
        done
    done
done