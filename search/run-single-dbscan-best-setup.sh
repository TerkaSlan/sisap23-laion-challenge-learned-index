#!/bin/bash

module add pytorch-1.1.0_python-3.6.2_cuda-10.1
source /storage/brno11-elixir/home/tslaninakova/dynamic-lmi-env/bin/activate
cd /auto/brno12-cerit/nfs4/home/tslaninakova/sisap-challenge/repo

python3 search/dbscan-best-setup.py \
--eps=$EPS --min-samples=$MIN_SAMPLES \
--leaf-size=$LEAF_SIZE --p=$P 2>&1 | tee -a job-logs/$PBS_JOBID.log
