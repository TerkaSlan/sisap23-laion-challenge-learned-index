#!/bin/bash
#PBS -q elixircz@elixir-pbs.elixir-czech.cz
#PBS -l select=1:ncpus=1:mem=50gb
#PBS -l walltime=4:00:00
#PBS -m ae
#PBS -N eval-search-hyp

module add pytorch-1.1.0_python-3.6.2_cuda-10.1
source /storage/brno11-elixir/home/tslaninakova/dynamic-lmi-env/bin/activate
cd /auto/brno12-cerit/nfs4/home/tslaninakova/sisap-challenge/repo/search

python evaluate-search-hyp.py