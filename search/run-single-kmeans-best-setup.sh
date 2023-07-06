#!/bin/bash

module add pytorch-1.1.0_python-3.6.2_cuda-10.1
source /storage/brno11-elixir/home/tslaninakova/dynamic-lmi-env/bin/activate
cd /auto/brno12-cerit/nfs4/home/tslaninakova/sisap-challenge/repo

python3 search/kmeans-best-setup.py \
--dataset=$DATA --size=10M \
--n-categories=$N_CATEGORIES --epochs=$EPOCHS \
--model-type=$MODEL_TYPE --lr=$LR 2>&1 | tee -a job-logs/$PBS_JOBID.log
