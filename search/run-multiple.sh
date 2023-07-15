#!/bin/bash

SIZE=10M
#SIZE=10M
EMB=pca96
#PREPROCESS=False
#EMB=emb
PREPROCESS=False
SAVE=True

for PREPROCESS in True #False #True
do
    for DATASET in pca96v2 #clip768 #v2 #pca32v2 #pca96v2 #pca32v2 #clip768v2 #pca32v2 #clip768v2 #pca32v2 # clip768v2
    do
        for LR in 0.09 0.1 0.11 #0.01 #0.001 0.01 0.1 #0.01 0.1 #0.1 0.01 0.001 #0.025 #0.03 0.027 0.025 0.023 0.02 #0.07 0.05 0.03 0.025 0.02 #0.5 0.1 0.05 #0.01 0.001
        do
            for MODEL_TYPE in MLP MLP-2 MLP-3 MLP-4 MLP-5 MLP-7 MLP-8 MLP-6 MLP-9 #MLP-2 #MLP-3 #MLP-6 MLP-8 #MLP-8 MLP-4 MLP-5 MLP-7 #MLP-7 MLP-3 MLP #MLP MLP-6 MLP-8 #MLP MLP-2 MLP-3 MLP-4 MLP-6
            do
                for EPOCHS in 45 50 55 #180 200 250 300 #100 150 200 #40 100 200 #20 30 40 50 60 #80 100 120 #25 26 27 28 29 #30 31 32 33 #10 15 20 25 30 35 #50 100 200
                do
                    for N_CATEGORIES in 110 112 114 116 118 120 122 124 #500 #1000 #200 #45 46 47 48 49 #50 51 52 #10 15 20 30 35 40 45 #50 55 #100 150 200
                    do
                        #for N_BUCKETS in 4 5 10 20
                        #do
                        qsub -q elixircz@elixir-pbs.elixir-czech.cz -l walltime=24:00:00 -l select=1:mem=50gb:ncpus=1 \
                        -N sisap-$DATSET-$EMB-$N_CATEGORIES-$EPOCHS-$PREPROCESS-$LR-$MODEL_TYPE \
                        -v LR="$LR",EMB="$EMB",EPOCHS="$EPOCHS",PREPROCESS="$PREPROCESS",N_CATEGORIES="$N_CATEGORIES",DATASET="$DATASET",SIZE="$SIZE",MODEL_TYPE="$MODEL_TYPE" \
                        run-single.sh
                        #done
                    done
                done
            done
        done
    done
done