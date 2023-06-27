#!/bin/bash
#PBS -q elixircz@elixir-pbs.elixir-czech.cz
#PBS -l select=1:ncpus=1:mem=50gb:cluster=elmo3
#PBS -l walltime=24:00:00
#PBS -m ae
#PBS -N sisap-10M-96

module add pytorch-1.1.0_python-3.6.2_cuda-10.1
source /storage/brno11-elixir/home/tslaninakova/dynamic-lmi-env/bin/activate
cd /auto/brno12-cerit/nfs4/home/tslaninakova/sisap-challenge/repo

#python3 search-dev.py --dataset=pca32v2 --emb=pca32 --size=10M --n-categories=1000 2>&1 | tee -a job-logs/$PBS_JOBID.log
python3 search/search-dev.py --dataset=pca96v2 --emb=pca96 --size=10M --n-categories=1000 2>&1 | tee -a job-logs/$PBS_JOBID.log
