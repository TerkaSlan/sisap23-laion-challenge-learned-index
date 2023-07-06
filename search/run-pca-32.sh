#!/bin/bash
#PBS -q elixircz@elixir-pbs.elixir-czech.cz
#PBS -l select=1:ncpus=1:mem=150gb:cluster=elmo3
#PBS -l walltime=24:00:00
#PBS -m ae
#PBS -N sisap-10M-pca32-ep-500-lr-0.1-mlp

module add pytorch-1.1.0_python-3.6.2_cuda-10.1
source /storage/brno11-elixir/home/tslaninakova/dynamic-lmi-env/bin/activate
cd /auto/brno12-cerit/nfs4/home/tslaninakova/sisap-challenge/repo

#python3 search-dev.py --dataset=pca32v2 --emb=pca32 --size=10M --n-categories=1000 2>&1 | tee -a job-logs/$PBS_JOBID.log
#python3 search/search-dev.py --dataset=pca32v2 --emb=pca32 --size=10M --n-categories=1000 --epochs=1000 \
# 2>&1 | tee -a job-logs/$PBS_JOBID.log

python3 search/search-dev.py --dataset=pca32v2 --emb=pca32 --size=100K --n-categories=500 --epochs=50 \
 --lr=0.1 --model-type=MLP --perc-train=  2>&1 | tee -a job-logs/$PBS_JOBID.log
#python3 search/search-dev.py --dataset=pca32v2 --emb=pca32 --size=10M --n-categories=1000 --epochs=1000 --lr=0.1 2>&1 | tee -a job-logs/$PBS_JOBID.log
